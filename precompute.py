"""
precompute.py — Phase 1 of the submission pipeline.

Reads candidates.jsonl, encodes every candidate with the bi-encoder,
builds a FAISS index, and persists all artifacts to disk.

Must be run ONCE before rank.py.

Performance strategy (CPU-only):
  - Text truncated to --max-words before tokenisation (model limit: 512 tokens).
  - Staggered multi-process execution to avoid OpenMP deadlock/contention on CPU.
  - Limits threads per worker to prevent CPU thrashing.
  - Typical throughput: 50-80 texts/sec on an 8-12 core CPU (~20-30 min).

Usage:
    python precompute.py \\
        --candidates data/candidates.jsonl \\
        --artifacts  artifacts/ \\
        --workers    6 \\
        --max-words  160
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import tempfile
from pathlib import Path
from multiprocessing import Process

from loguru import logger
import numpy as np

# bge-small instruction prefix
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Precompute candidate embeddings and FAISS index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/candidates.jsonl"),
        help="Path to candidates.jsonl",
    )
    p.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts"),
        help="Directory to write embeddings, ids, and FAISS index.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override bi-encoder model (default: BAAI/bge-small-en-v1.5).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes for encoding. Defaults to min(cpu_count/2, 6).",
    )
    p.add_argument(
        "--threads-per-worker",
        type=int,
        default=2,
        help="Number of PyTorch CPU threads per worker process.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Per-worker encode batch size.",
    )
    p.add_argument(
        "--max-words",
        type=int,
        default=160,
        help="Maximum words to truncate candidate profile texts to before encoding.",
    )
    p.add_argument(
        "--limit-candidates",
        type=int,
        default=5000,
        help="Pre-filter to keep only the top N candidates by keyword similarity (0 for no limit).",
    )
    p.add_argument(
        "--jd",
        type=Path,
        default=Path("data/job_description.docx"),
        help="Path to job description for pre-filtering.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-encode even if artifacts already exist.",
    )
    return p.parse_args()


def _truncate(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def _worker_encode(
    worker_id: int,
    texts: list[str],
    temp_dir: str,
    model_name: str,
    batch_size: int,
    threads: int,
) -> None:
    """Worker process function to load model and encode a chunk of texts."""
    import torch
    from sentence_transformers import SentenceTransformer

    # Restrict PyTorch thread count inside worker to prevent core oversubscription/thrashing
    torch.set_num_threads(threads)
    
    # Load model locally in this process
    model = SentenceTransformer(model_name, device="cpu")
    
    # Perform batch encoding
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    
    # Save the chunk results to temp file
    out_path = os.path.join(temp_dir, f"chunk_{worker_id}.npy")
    np.save(out_path, embeddings.astype(np.float32))


def main() -> None:
    args = _parse_args()
    t0 = time.perf_counter()

    logger.info("=== precompute.py — Candidate Embedding Pipeline ===")
    logger.info(f"Candidates : {args.candidates}")
    logger.info(f"Artifacts  : {args.artifacts}")

    if not args.candidates.exists():
        logger.error(f"candidates.jsonl not found: {args.candidates}")
        sys.exit(1)

    args.artifacts.mkdir(parents=True, exist_ok=True)

    # Lazy imports
    from src.embeddings.engine import DEFAULT_MODEL_NAME, EmbeddingEngine
    from src.parsers.candidate_parser import CandidateParser, CandidateTextBuilder
    from src.retrieval.vector_store import VectorStore

    model_name = args.model or DEFAULT_MODEL_NAME

    already_embedded = EmbeddingEngine.artifacts_exist(args.artifacts)
    already_indexed = VectorStore.artifacts_exist(args.artifacts)

    if already_embedded and already_indexed and not args.force:
        logger.info("All artifacts already exist. Use --force to re-build. Exiting.")
        return

    # ── Step 1: stream candidates, build and truncate texts ──────────────────
    logger.info("Step 1/3 — Streaming candidates and building embedding texts...")
    t_parse = time.perf_counter()
    
    from src.parsers.jd_parser import JDParser
    
    target_skills = None
    if args.limit_candidates > 0 and args.jd and args.jd.exists():
        try:
            jd_parser = JDParser()
            jd = jd_parser.parse(args.jd)
            target_skills = set()
            for s in jd.must_have_skills:
                target_skills.add(s.name.lower())
            for s in jd.nice_to_have_skills:
                target_skills.add(s.name.lower())
            target_title_words = set(jd.title.lower().split())
            logger.info(f"Loaded job description for keyword pre-filtering. Target title: '{jd.title}'")
        except Exception as e:
            logger.warning(f"Could not parse job description for filtering: {e}. Indexing all candidates.")
            
    parser = CandidateParser(args.candidates)
    builder = CandidateTextBuilder()

    scored_profiles = []
    for profile in parser.iter_candidates():
        if target_skills is not None:
            cand_skills = set(s.lower() for s in profile.skills)
            skill_match_count = len(cand_skills.intersection(target_skills))
            
            cand_title = (profile.current_title or "").lower()
            title_match_count = len(set(cand_title.split()).intersection(target_title_words))
            
            score = skill_match_count * 3 + title_match_count * 5
            scored_profiles.append((score, profile))
        else:
            scored_profiles.append((0, profile))

    if target_skills is not None and args.limit_candidates > 0:
        scored_profiles.sort(key=lambda x: x[0], reverse=True)
        keep_profiles = [p for _, p in scored_profiles[:args.limit_candidates]]
        logger.info(f"Filtered candidate pool from {len(scored_profiles):,} to top {len(keep_profiles):,} by relevance score.")
    else:
        keep_profiles = [p for _, p in scored_profiles]

    candidate_ids: list[str] = []
    texts: list[str] = []

    for profile in keep_profiles:
        candidate_ids.append(profile.candidate_id)
        texts.append(_truncate(builder.build(profile), args.max_words))

    parse_elapsed = time.perf_counter() - t_parse
    logger.info(
        f"  Prepared {len(texts):,} candidates in {parse_elapsed:.1f}s. "
        f"Avg text length: {sum(len(t.split()) for t in texts) // max(len(texts), 1)} words."
    )

    if not texts:
        logger.error("No candidates selected. Aborting.")
        sys.exit(1)

    # ── Step 2: Staggered multi-process encode ────────────────────────────────
    cpu_count = os.cpu_count() or 4
    n_workers = args.workers or min(max(cpu_count // 2, 1), 6)
    
    logger.info(
        f"Step 2/3 — Staggering {n_workers} worker processes, "
        f"threads_per_worker={args.threads_per_worker}, batch_size={args.batch_size}..."
    )
    
    # Split text corpus into chunks for parallel workers
    chunks = np.array_split(texts, n_workers)
    
    t_enc = time.perf_counter()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        processes = []
        for idx, chunk in enumerate(chunks):
            chunk_list = chunk.tolist()
            p = Process(
                target=_worker_encode,
                args=(idx, chunk_list, temp_dir, model_name, args.batch_size, args.threads_per_worker),
            )
            processes.append(p)
            p.start()
            
            # Stagger worker process startup to prevent OpenMP initialization deadlock/thrashing on CPU
            time.sleep(2.0)
            
        logger.info(f"  All workers spawned. Waiting for encoding to complete...")
        for p in processes:
            p.join()
            
        # Consolidate chunked embeddings
        all_embeddings = []
        for idx in range(n_workers):
            chunk_path = os.path.join(temp_dir, f"chunk_{idx}.npy")
            if not os.path.exists(chunk_path):
                logger.error(f"Worker {idx} failed to produce embeddings. Aborting.")
                sys.exit(1)
            all_embeddings.append(np.load(chunk_path))
            
        embeddings = np.vstack(all_embeddings)

    enc_elapsed = time.perf_counter() - t_enc
    logger.info(
        f"  Encoded {len(texts):,} candidates in {enc_elapsed:.1f}s "
        f"({len(texts) / enc_elapsed:.0f} cands/sec) "
        f"| shape={embeddings.shape}"
    )

    # Save embeddings + ids
    EmbeddingEngine.save(candidate_ids, embeddings, args.artifacts)
    logger.info(f"  Embeddings saved -> {args.artifacts}/")

    # ── Step 3: build FAISS index ─────────────────────────────────────────────
    logger.info("Step 3/3 — Building FAISS IndexFlatIP...")
    t_faiss = time.perf_counter()
    store = VectorStore.build(candidate_ids=candidate_ids, embeddings=embeddings)
    store.save(args.artifacts)
    faiss_elapsed = time.perf_counter() - t_faiss
    logger.info(
        f"  FAISS index built and saved: {store.num_candidates:,} vectors "
        f"in {faiss_elapsed:.2f}s"
    )

    total = time.perf_counter() - t0
    logger.info(f"=== precompute.py complete in {total:.1f}s ({total/60:.1f} min) ===")
    logger.info(f"Artifacts ready in: {args.artifacts.resolve()}")
    logger.info("")
    logger.info("Next step:")
    logger.info("  python rank.py --jd data/job_description.docx --output submission.csv")


if __name__ == "__main__":
    main()
