"""
Evaluation module — measures ranking quality against ground truth.

Metrics computed:
  - NDCG@10, NDCG@20, NDCG@100  (primary leaderboard metric family)
  - Precision@10, Precision@20
  - Mean Reciprocal Rank (MRR)
  - Honeypot exclusion rate (what fraction of honeypots were correctly excluded)
  - Score distribution statistics (mean, std, min, max, median)

All metrics are computed without external ML libraries — pure NumPy for
reproducibility and portability.

Usage:
    evaluator = RankingEvaluator()
    report = evaluator.evaluate(
        ranked_ids=["CAND_001", "CAND_002", ...],   # ordered top-100
        scores=[87.3, 85.1, ...],                    # parallel to ranked_ids
        relevant_ids={"CAND_001", "CAND_007", ...},  # ground truth set
    )
    print(report.summary())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class EvaluationReport:
    """
    Full evaluation report for one ranking.

    All metric values are in [0, 1] unless otherwise noted.
    """

    ndcg_at_10: float
    ndcg_at_20: float
    ndcg_at_100: float
    precision_at_10: float
    precision_at_20: float
    mrr: float                    # Mean Reciprocal Rank
    honeypot_exclusion_rate: float  # fraction of honeypots NOT in top-100

    # Score distribution
    score_mean: float
    score_std: float
    score_min: float
    score_max: float
    score_median: float

    # Inputs (stored for reproducibility)
    num_ranked: int
    num_relevant: int
    num_honeypots_excluded: int = 0
    num_total_honeypots: int = 0

    def summary(self) -> str:
        """Return a one-line human-readable summary of key metrics."""
        return (
            f"NDCG@10={self.ndcg_at_10:.4f}  NDCG@20={self.ndcg_at_20:.4f}  "
            f"NDCG@100={self.ndcg_at_100:.4f}  MRR={self.mrr:.4f}  "
            f"P@10={self.precision_at_10:.4f}  P@20={self.precision_at_20:.4f}"
        )

    def as_dict(self) -> dict[str, float]:
        """Return all metrics as a flat dict for logging / CSV export."""
        return {
            "ndcg@10": self.ndcg_at_10,
            "ndcg@20": self.ndcg_at_20,
            "ndcg@100": self.ndcg_at_100,
            "precision@10": self.precision_at_10,
            "precision@20": self.precision_at_20,
            "mrr": self.mrr,
            "honeypot_exclusion_rate": self.honeypot_exclusion_rate,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "score_median": self.score_median,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class RankingEvaluator:
    """
    Stateless ranking evaluator.

    All methods are pure functions — no state stored between calls.
    """

    def evaluate(
        self,
        ranked_ids: list[str],
        scores: list[float],
        relevant_ids: set[str],
        *,
        honeypot_ids: set[str] | None = None,
    ) -> EvaluationReport:
        """
        Compute all ranking metrics for one result set.

        Args:
            ranked_ids:   Candidate IDs in rank order (rank 1 first).
            scores:       Final scores parallel to ranked_ids, in [0, 100].
            relevant_ids: Ground-truth set of relevant candidate IDs.
            honeypot_ids: Set of known honeypot IDs. If provided, computes
                          honeypot_exclusion_rate.

        Returns:
            EvaluationReport with all metrics populated.
        """
        if len(ranked_ids) != len(scores):
            raise ValueError(
                f"ranked_ids ({len(ranked_ids)}) and scores ({len(scores)}) "
                "must have equal length."
            )

        scores_arr = np.array(scores, dtype=np.float64)

        # Binary relevance vector: 1 if relevant, 0 otherwise
        relevance = np.array(
            [1 if cid in relevant_ids else 0 for cid in ranked_ids],
            dtype=np.float64,
        )

        ndcg10 = self._ndcg(relevance, relevant_ids, k=10)
        ndcg20 = self._ndcg(relevance, relevant_ids, k=20)
        ndcg100 = self._ndcg(relevance, relevant_ids, k=100)

        p10 = self._precision_at_k(relevance, k=10)
        p20 = self._precision_at_k(relevance, k=20)
        mrr = self._mrr(relevance)

        # Honeypot exclusion
        hp_ids = honeypot_ids or set()
        total_hp = len(hp_ids)
        ranked_set = set(ranked_ids)
        excluded_hp = sum(1 for h in hp_ids if h not in ranked_set)
        hp_rate = excluded_hp / total_hp if total_hp > 0 else 1.0

        return EvaluationReport(
            ndcg_at_10=ndcg10,
            ndcg_at_20=ndcg20,
            ndcg_at_100=ndcg100,
            precision_at_10=p10,
            precision_at_20=p20,
            mrr=mrr,
            honeypot_exclusion_rate=hp_rate,
            score_mean=float(scores_arr.mean()) if len(scores_arr) else 0.0,
            score_std=float(scores_arr.std()) if len(scores_arr) else 0.0,
            score_min=float(scores_arr.min()) if len(scores_arr) else 0.0,
            score_max=float(scores_arr.max()) if len(scores_arr) else 0.0,
            score_median=float(np.median(scores_arr)) if len(scores_arr) else 0.0,
            num_ranked=len(ranked_ids),
            num_relevant=len(relevant_ids),
            num_honeypots_excluded=excluded_hp,
            num_total_honeypots=total_hp,
        )

    # ------------------------------------------------------------------
    # Metric implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _dcg(relevance: np.ndarray, k: int) -> float:
        """Discounted Cumulative Gain at k."""
        top_k = relevance[:k]
        positions = np.arange(2, len(top_k) + 2, dtype=np.float64)  # ranks 1-indexed → log2(rank+1)
        return float(np.sum(top_k / np.log2(positions)))

    def _ndcg(
        self, relevance: np.ndarray, relevant_ids: set[str], k: int
    ) -> float:
        """Normalised DCG at k. Returns 0 if no relevant items exist."""
        if not relevant_ids:
            return 0.0
        actual_dcg = self._dcg(relevance, k)
        # Ideal: place all relevant items at the top
        ideal_rel = np.ones(min(len(relevant_ids), k), dtype=np.float64)
        ideal_dcg = self._dcg(ideal_rel, k)
        if ideal_dcg < 1e-10:
            return 0.0
        return min(1.0, actual_dcg / ideal_dcg)

    @staticmethod
    def _precision_at_k(relevance: np.ndarray, k: int) -> float:
        """Fraction of top-k results that are relevant."""
        top_k = relevance[:k]
        if len(top_k) == 0:
            return 0.0
        return float(top_k.sum() / len(top_k))

    @staticmethod
    def _mrr(relevance: np.ndarray) -> float:
        """Mean Reciprocal Rank — 1/rank of first relevant result."""
        for i, rel in enumerate(relevance):
            if rel > 0:
                return 1.0 / (i + 1)
        return 0.0
