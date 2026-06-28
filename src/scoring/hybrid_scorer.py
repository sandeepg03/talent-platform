"""
Hybrid Scorer — combines all sub-scores into a single final ranking.

Architecture position:
  RerankResult  (ce_scores, semantic_scores)
  FeatureVector (experience, education, certification, redrob_signal, is_honeypot)
        ↓
  HybridScorer.score_all()
        ↓
  list[HybridScore]  — sorted descending, honeypots excluded
        ↓
  ExplanationGenerator

Scoring formula (strictly from specification):
  composite = 0.40 × semantic_similarity
            + 0.30 × cross_encoder_score
            + 0.10 × experience_score
            + 0.10 × redrob_signal_score
            + 0.05 × education_score
            + 0.05 × certification_score

  final_score = composite × 100   (rounded to 4 dp, range [0, 100])

Post-scoring pipeline:
  1. Compute HybridScore for every non-honeypot candidate in the reranked pool.
  2. Sort descending by final_score.
  3. Return the top-100 (required by submission spec).

Honeypot handling:
  - Honeypot candidates are SCORED (their HybridScore is computed) but
    NOT included in the final top-100 list returned to rank.py.
  - This allows post-hoc audit of flagged candidates.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from src.reranker.cross_encoder import RerankResult
from src.schemas.scoring import FeatureVector, HybridScore

# ---------------------------------------------------------------------------
# Score batch result
# ---------------------------------------------------------------------------


@dataclass
class ScoringResult:
    """
    Full output of HybridScorer.score_all().

    Attributes:
        ranked:        Top-100 HybridScore objects, sorted descending by final_score.
                       These are the candidates submitted to the leaderboard.
        all_scored:    Every scored candidate (including honeypots) — for audit / UI.
        honeypots:     Candidates flagged as honeypots (excluded from ``ranked``).
        num_input:     How many candidates entered the scorer.
    """

    ranked: list[HybridScore]  # top-100, honeypot-free
    all_scored: list[HybridScore]  # full set, sorted descending
    honeypots: list[HybridScore]  # flagged but scored
    num_input: int

    @property
    def num_honeypots(self) -> int:
        return len(self.honeypots)

    @property
    def num_clean(self) -> int:
        return self.num_input - self.num_honeypots


# ---------------------------------------------------------------------------
# Hybrid Scorer
# ---------------------------------------------------------------------------


class HybridScorer:
    """
    Applies the hybrid weighted scoring formula to produce the final ranking.

    Stateless — safe to instantiate once and call repeatedly.

    Usage:
        scorer = HybridScorer()
        result = scorer.score_all(
            rerank_result=rerank_result,
            feature_vectors=feature_vectors,
            top_n=100,
        )
        top_100 = result.ranked
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_all(
        self,
        rerank_result: RerankResult,
        feature_vectors: list[FeatureVector],
        top_n: int = 100,
    ) -> ScoringResult:
        """
        Score every candidate and return the top-N ranked list.

        Args:
            rerank_result:   Output of CrossEncoderReranker.rerank() — provides
                             semantic_scores and ce_scores keyed by candidate_id.
            feature_vectors: List of FeatureVector produced by FeatureEngineer
                             for the same set of candidates.
            top_n:           How many top-ranked non-honeypot candidates to return.
                             Must be >= 1.  The submission spec requires exactly 100.

        Returns:
            ScoringResult with:
              - ranked:     top-N clean candidates, descending by final_score
              - all_scored: all candidates scored, descending by final_score
              - honeypots:  flagged candidates (excluded from ranked)
              - num_input:  total number of candidates processed

        Raises:
            ValueError: If top_n < 1 or feature_vectors is empty.
        """
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")
        if not feature_vectors:
            raise ValueError("feature_vectors must not be empty.")

        # Build O(1) lookup maps from rerank result
        sem_dict = rerank_result.as_semantic_score_dict()
        ce_dict = rerank_result.as_ce_score_dict()

        scores: list[HybridScore] = []
        skipped = 0

        for fv in feature_vectors:
            sem = sem_dict.get(fv.candidate_id)
            ce = ce_dict.get(fv.candidate_id)

            if sem is None or ce is None:
                logger.warning(
                    f"candidate_id {fv.candidate_id!r} missing from rerank_result — skipping."
                )
                skipped += 1
                continue

            hs = HybridScore.compute(
                candidate_id=fv.candidate_id,
                semantic_similarity=float(sem),
                cross_encoder_score=float(ce),
                experience_score=fv.experience_score,
                redrob_signal_score=fv.redrob_signal_score,
                education_score=fv.education_score,
                certification_score=fv.certification_score,
                is_honeypot=fv.is_honeypot,
            )
            scores.append(hs)

        if skipped:
            logger.warning(
                f"{skipped} candidate(s) skipped (missing from rerank_result). "
                f"{len(scores)} candidates scored."
            )

        # Sort all scored candidates descending
        scores.sort(key=lambda h: h.final_score, reverse=True)

        honeypots = [h for h in scores if h.is_honeypot]
        clean = [h for h in scores if not h.is_honeypot]
        ranked = clean[:top_n]

        logger.info(
            f"Scoring complete: {len(scores)} scored, "
            f"{len(honeypots)} honeypots excluded, "
            f"{len(ranked)} candidates in final ranking "
            f"(top score={ranked[0].final_score:.2f} if ranked else 'N/A')"
        )

        return ScoringResult(
            ranked=ranked,
            all_scored=scores,
            honeypots=honeypots,
            num_input=len(feature_vectors),
        )

    def score_one(
        self,
        candidate_id: str,
        semantic_similarity: float,
        cross_encoder_score: float,
        feature_vector: FeatureVector,
    ) -> HybridScore:
        """
        Score a single candidate directly (for streaming / incremental pipelines).

        Args:
            candidate_id:       CAND_XXXXXXX string.
            semantic_similarity: Cosine similarity from FAISS retrieval [0, 1].
            cross_encoder_score: Normalised CE score [0, 1].
            feature_vector:      FeatureVector for this candidate.

        Returns:
            HybridScore with all component scores and final_score.
        """
        return HybridScore.compute(
            candidate_id=candidate_id,
            semantic_similarity=semantic_similarity,
            cross_encoder_score=cross_encoder_score,
            experience_score=feature_vector.experience_score,
            redrob_signal_score=feature_vector.redrob_signal_score,
            education_score=feature_vector.education_score,
            certification_score=feature_vector.certification_score,
            is_honeypot=feature_vector.is_honeypot,
        )
