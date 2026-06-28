"""
rank.py — Phase 2 of the submission pipeline.

Loads pre-computed artifacts, runs the full ranking pipeline over
the job description, and writes the final submission CSV.

Must run precompute.py before this script.

Usage:
    python rank.py \\
        --jd           data/job_description.docx \\
        --candidates   data/candidates.jsonl \\
        --artifacts    artifacts/ \\
        --output       submission.csv \\
        --retrieval-k  500 \\
        --rerank-k     200

Pipeline:
    1. Load StructuredJD from .docx (or .json cache)
    2. Load Retriever (bi-encoder + FAISS index from artifacts/)
    3. Retrieve top-K candidates (semantic search)
    4. CrossEncoder reranking → top-K'
    5. Feature extraction (experience, education, certs, signals)
    6. Hybrid scoring (formula: 0.40*sem + 0.30*ce + 0.10*exp + ...)
    7. Explanation generation
    8. Write submission.csv (candidate_id, rank, score, reasoning)
    9. Validate with validate_submission.py logic
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loguru import logger


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run full ranking pipeline and write submission CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--jd",
        type=Path,
        default=Path("data/job_description.docx"),
        help="Path to job_description.docx (or .json canonical cache).",
    )
    p.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/candidates.jsonl"),
        help="Path to candidates.jsonl.",
    )
    p.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts"),
        help="Directory containing precomputed embeddings + FAISS index.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("submission.csv"),
        help="Output path for the submission CSV.",
    )
    p.add_argument(
        "--retrieval-k",
        type=int,
        default=500,
        help="Number of candidates to retrieve with FAISS.",
    )
    p.add_argument(
        "--rerank-k",
        type=int,
        default=200,
        help="Number of candidates to keep after cross-encoder reranking.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of candidates in the final submission (must be 100).",
    )
    return p.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.jd.exists():
        logger.error(f"JD file not found: {args.jd}")
        sys.exit(1)
    if not args.candidates.exists():
        logger.error(f"Candidates file not found: {args.candidates}")
        sys.exit(1)
    if not args.artifacts.exists():
        logger.error(
            f"Artifacts directory not found: {args.artifacts}. "
            "Run precompute.py first."
        )
        sys.exit(1)


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    t_total = time.perf_counter()
    logger.info("=== rank.py — AI Talent Intelligence Ranking Pipeline ===")

    # ── Lazy imports ──────────────────────────────────────────────────────────
    from src.explanation.generator import ExplanationGenerator
    from src.features.feature_engineer import FeatureEngineer
    from src.parsers.candidate_parser import CandidateParser, CandidateTextBuilder
    from src.parsers.jd_parser import JDParser
    from src.reranker.cross_encoder import CrossEncoderReranker
    from src.retrieval.retriever import Retriever
    from src.scoring.hybrid_scorer import HybridScorer
    from src.schemas.scoring import SubmissionResult, SubmissionRow

    # ── Step 1: Parse JD ─────────────────────────────────────────────────────
    logger.info(f"Step 1/8 — Parsing JD: {args.jd}")
    jd_parser = JDParser()
    if args.jd.suffix.lower() == ".docx":
        jd = jd_parser.from_docx(args.jd)
    else:
        jd = jd_parser.from_canonical(args.jd)
    logger.info(f"  JD parsed: {jd.title!r} @ {jd.company!r}")
    logger.info(f"  Must-have skills: {[s.name for s in jd.must_have_skills]}")

    # ── Step 2: Load Retriever ────────────────────────────────────────────────
    logger.info(f"Step 2/8 — Loading Retriever from {args.artifacts}")
    retriever = Retriever.from_artifacts(
        args.artifacts,
        default_top_k=args.retrieval_k,
    )
    logger.info(f"  Retriever ready: {retriever.num_candidates:,} candidates indexed.")

    # ── Step 3: FAISS Retrieval ───────────────────────────────────────────────
    logger.info(f"Step 3/8 — Semantic retrieval (top-{args.retrieval_k})")
    retrieval_result = retriever.retrieve(jd, top_k=args.retrieval_k)
    logger.info(
        f"  Retrieved {retrieval_result.num_retrieved} candidates "
        f"in {retrieval_result.retrieval_time_ms:.1f}ms"
    )

    # ── Step 4: Build candidate text map (for cross-encoder pairs) ───────────
    logger.info("Step 4/8 — Loading candidate texts for cross-encoder...")
    parser = CandidateParser(args.candidates)
    builder = CandidateTextBuilder()
    retrieved_set = set(retrieval_result.candidate_ids)

    candidate_texts: dict[str, str] = {}
    candidate_map: dict[str, object] = {}

    for profile in parser.iter_candidates():
        if profile.candidate_id in retrieved_set:
            candidate_texts[profile.candidate_id] = builder.build(profile)
            candidate_map[profile.candidate_id] = profile

    logger.info(
        f"  Loaded {len(candidate_texts):,} candidate texts "
        f"({retrieval_result.num_retrieved - len(candidate_texts)} IDs not found in JSONL)."
    )

    # ── Step 5: Cross-encoder reranking ──────────────────────────────────────
    logger.info(f"Step 5/8 — Cross-encoder reranking (top-{args.rerank_k})")
    reranker = CrossEncoderReranker()
    reranker.load_model()
    reranked = reranker.rerank(
        retrieval_result,
        candidate_texts,
        top_k=args.rerank_k,
    )
    logger.info(
        f"  Reranked to {reranked.num_candidates} candidates "
        f"in {reranked.rerank_time_ms:.0f}ms"
    )

    # ── Step 6: Feature extraction ────────────────────────────────────────────
    logger.info("Step 6/8 — Feature extraction")
    feature_engineer = FeatureEngineer()
    feature_vectors = []
    n_honeypots_detected = 0

    for cid in reranked.candidate_ids:
        profile = candidate_map.get(cid)
        if profile is None:
            continue
        fv = feature_engineer.extract(profile, jd)  # type: ignore[arg-type]
        feature_vectors.append(fv)
        if fv.is_honeypot:
            n_honeypots_detected += 1

    logger.info(
        f"  Extracted features for {len(feature_vectors)} candidates. "
        f"Honeypots detected: {n_honeypots_detected}"
    )

    if not feature_vectors:
        logger.error("No feature vectors computed — aborting.")
        sys.exit(1)

    # ── Step 7: Hybrid scoring ────────────────────────────────────────────────
    logger.info(f"Step 7/8 — Hybrid scoring (top-{args.top_n})")
    scorer = HybridScorer()
    scoring_result = scorer.score_all(reranked, feature_vectors, top_n=args.top_n)

    if scoring_result.ranked:
        logger.info(
            f"  Scored {scoring_result.num_input} candidates. "
            f"Honeypots excluded: {scoring_result.num_honeypots}. "
            f"Top score: {scoring_result.ranked[0].final_score:.2f}/100"
        )
    else:
        logger.error("No clean candidates scored — aborting.")
        sys.exit(1)

    if len(scoring_result.ranked) < args.top_n:
        logger.error(
            f"Only {len(scoring_result.ranked)} clean candidates available "
            f"(need {args.top_n}). Increase --retrieval-k or --rerank-k."
        )
        sys.exit(1)

    # ── Step 8: Explanations + CSV ────────────────────────────────────────────
    logger.info("Step 8/8 — Generating explanations and writing CSV")
    explainer = ExplanationGenerator()
    fv_map = {fv.candidate_id: fv for fv in feature_vectors}

    rows: list[SubmissionRow] = []
    for rank_idx, hs in enumerate(scoring_result.ranked, start=1):
        fv = fv_map.get(hs.candidate_id)
        reasoning = (
            explainer.generate(hs, fv, rank=rank_idx)
            if fv
            else f"Ranked #{rank_idx} with composite score {hs.final_score:.2f}/100."
        )
        rows.append(
            SubmissionRow(
                candidate_id=hs.candidate_id,
                rank=rank_idx,
                score=round(hs.final_score, 4),
                reasoning=reasoning,
            )
        )

    submission = SubmissionResult(rows=rows)
    submission.to_csv(args.output)
    logger.info(f"  Submission CSV written to: {args.output.resolve()}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_total
    logger.info("=== Pipeline Complete ===")
    logger.info(f"  Total time        : {total_elapsed:.1f}s")
    logger.info(f"  Candidates ranked : {args.top_n}")
    logger.info(f"  Honeypots excluded: {scoring_result.num_honeypots}")
    logger.info(f"  Top-1 score       : {scoring_result.ranked[0].final_score:.4f}/100")
    logger.info(f"  Output            : {args.output.resolve()}")
    logger.info("")
    logger.info("Run validation:")
    logger.info(f"  python validate_submission.py {args.output}")


if __name__ == "__main__":
    main()
