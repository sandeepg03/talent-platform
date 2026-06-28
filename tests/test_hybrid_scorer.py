"""
Unit tests for src.scoring.hybrid_scorer

All tests are fully synthetic — no model downloads, no file I/O.

Covers:
  - ScoringResult properties
  - HybridScorer.score_all(): formula correctness, sorting, honeypot exclusion,
    top-N slicing, skipped-candidate handling, empty input guard
  - HybridScorer.score_one(): single-candidate path
  - Formula weight invariants: weights sum to 1, perfect/zero inputs
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from src.reranker.cross_encoder import RerankResult
from src.schemas.scoring import FeatureVector, HybridScore
from src.scoring.hybrid_scorer import HybridScorer, ScoringResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TODAY = datetime.date.today()


def _fv(
    cid: str,
    exp: float = 0.8,
    edu: float = 0.7,
    cert: float = 0.2,
    sig: float = 0.6,
    honeypot: bool = False,
) -> FeatureVector:
    return FeatureVector(
        candidate_id=cid,
        experience_score=exp,
        education_score=edu,
        certification_score=cert,
        redrob_signal_score=sig,
        signal_open_to_work=1.0,
        signal_response_rate=0.8,
        signal_interview_completion=0.9,
        signal_profile_completeness=0.8,
        signal_recency=0.95,
        signal_github=0.55,
        signal_assessment_avg=0.6,
        signal_saved_by_recruiters=0.3,
        is_honeypot=honeypot,
        years_of_experience=6.0,
        highest_education_degree="Bachelor of Technology",
        matched_must_have_skills=["python", "faiss"],
        matched_nice_to_have_skills=["kubernetes"],
        cert_names=["AWS ML Specialty"],
    )


def _rerank(
    ids: list[str],
    sem_scores: list[float] | None = None,
    ce_scores: list[float] | None = None,
) -> RerankResult:
    n = len(ids)
    sem = np.array(sem_scores or [0.85] * n, dtype=np.float32)
    ce = np.array(ce_scores or [0.70] * n, dtype=np.float32)
    raw = np.array([1.5] * n, dtype=np.float32)
    return RerankResult(
        candidate_ids=ids,
        semantic_scores=sem,
        ce_scores=ce,
        ce_raw_scores=raw,
        query_text="Senior AI Engineer",
        rerank_time_ms=100.0,
    )


# ---------------------------------------------------------------------------
# Tests — ScoringResult
# ---------------------------------------------------------------------------


class TestScoringResult:
    def _make(self, n_clean: int = 5, n_honey: int = 2) -> ScoringResult:
        clean = [
            HybridScore.compute(f"CAND_{i:07d}", 0.8, 0.7, 0.8, 0.6, 0.7, 0.2)
            for i in range(n_clean)
        ]
        honey = [
            HybridScore.compute(f"CAND_{i+n_clean:07d}", 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, is_honeypot=True)
            for i in range(n_honey)
        ]
        return ScoringResult(
            ranked=clean,
            all_scored=clean + honey,
            honeypots=honey,
            num_input=n_clean + n_honey,
        )

    def test_num_honeypots(self) -> None:
        r = self._make(5, 3)
        assert r.num_honeypots == 3

    def test_num_clean(self) -> None:
        r = self._make(5, 3)
        assert r.num_clean == 5

    def test_ranked_length(self) -> None:
        r = self._make(10, 2)
        assert len(r.ranked) == 10

    def test_honeypots_list(self) -> None:
        r = self._make(5, 2)
        assert all(h.is_honeypot for h in r.honeypots)


# ---------------------------------------------------------------------------
# Tests — HybridScorer.score_all()
# ---------------------------------------------------------------------------


class TestScoreAll:
    def test_returns_scoring_result(self) -> None:
        scorer = HybridScorer()
        ids = [f"CAND_{i:07d}" for i in range(10)]
        rr = _rerank(ids)
        fvs = [_fv(cid) for cid in ids]
        result = scorer.score_all(rr, fvs, top_n=5)
        assert isinstance(result, ScoringResult)

    def test_top_n_respected(self) -> None:
        scorer = HybridScorer()
        ids = [f"CAND_{i:07d}" for i in range(20)]
        rr = _rerank(ids)
        fvs = [_fv(cid) for cid in ids]
        result = scorer.score_all(rr, fvs, top_n=10)
        assert len(result.ranked) == 10

    def test_ranked_sorted_descending(self) -> None:
        scorer = HybridScorer()
        ids = [f"CAND_{i:07d}" for i in range(10)]
        # Give each candidate a different semantic score so ordering is deterministic
        sem = np.linspace(0.5, 0.9, 10, dtype=np.float32).tolist()
        rr = _rerank(ids, sem_scores=sem)
        fvs = [_fv(cid) for cid in ids]
        result = scorer.score_all(rr, fvs, top_n=10)
        for i in range(len(result.ranked) - 1):
            assert result.ranked[i].final_score >= result.ranked[i + 1].final_score

    def test_honeypots_excluded_from_ranked(self) -> None:
        scorer = HybridScorer()
        ids = [f"CAND_{i:07d}" for i in range(10)]
        rr = _rerank(ids, sem_scores=[0.9] * 10, ce_scores=[0.9] * 10)
        # Mark first 3 as honeypots
        fvs = [_fv(cid, honeypot=(i < 3)) for i, cid in enumerate(ids)]
        result = scorer.score_all(rr, fvs, top_n=10)
        ranked_ids = {h.candidate_id for h in result.ranked}
        for i in range(3):
            assert ids[i] not in ranked_ids

    def test_honeypots_counted_separately(self) -> None:
        scorer = HybridScorer()
        ids = [f"CAND_{i:07d}" for i in range(10)]
        rr = _rerank(ids)
        fvs = [_fv(cid, honeypot=(i >= 7)) for i, cid in enumerate(ids)]
        result = scorer.score_all(rr, fvs, top_n=10)
        assert result.num_honeypots == 3

    def test_all_scored_includes_honeypots(self) -> None:
        scorer = HybridScorer()
        ids = [f"CAND_{i:07d}" for i in range(5)]
        rr = _rerank(ids)
        fvs = [_fv(cid, honeypot=(i == 0)) for i, cid in enumerate(ids)]
        result = scorer.score_all(rr, fvs, top_n=5)
        assert len(result.all_scored) == 5

    def test_scores_in_zero_100_range(self) -> None:
        scorer = HybridScorer()
        ids = [f"CAND_{i:07d}" for i in range(15)]
        rr = _rerank(ids)
        fvs = [_fv(cid) for cid in ids]
        result = scorer.score_all(rr, fvs, top_n=15)
        for hs in result.all_scored:
            assert 0.0 <= hs.final_score <= 100.0

    def test_top_n_capped_at_available_clean(self) -> None:
        scorer = HybridScorer()
        ids = [f"CAND_{i:07d}" for i in range(5)]
        rr = _rerank(ids)
        fvs = [_fv(cid) for cid in ids]
        result = scorer.score_all(rr, fvs, top_n=100)
        # Only 5 clean candidates available
        assert len(result.ranked) == 5

    def test_skips_candidate_missing_from_rerank(self) -> None:
        scorer = HybridScorer()
        # rerank has 5, feature_vectors has 7 (2 extra)
        rerank_ids = [f"CAND_{i:07d}" for i in range(5)]
        all_ids = rerank_ids + ["CAND_9999990", "CAND_9999991"]
        rr = _rerank(rerank_ids)
        fvs = [_fv(cid) for cid in all_ids]
        result = scorer.score_all(rr, fvs, top_n=10)
        # Only 5 candidates from rerank can be scored
        assert len(result.all_scored) == 5

    def test_invalid_top_n_raises(self) -> None:
        scorer = HybridScorer()
        rr = _rerank(["CAND_0000001"])
        fvs = [_fv("CAND_0000001")]
        with pytest.raises(ValueError, match="top_n"):
            scorer.score_all(rr, fvs, top_n=0)

    def test_empty_feature_vectors_raises(self) -> None:
        scorer = HybridScorer()
        rr = _rerank(["CAND_0000001"])
        with pytest.raises(ValueError, match="empty"):
            scorer.score_all(rr, [], top_n=100)

    def test_num_input_correct(self) -> None:
        scorer = HybridScorer()
        ids = [f"CAND_{i:07d}" for i in range(12)]
        rr = _rerank(ids)
        fvs = [_fv(cid) for cid in ids]
        result = scorer.score_all(rr, fvs, top_n=12)
        assert result.num_input == 12

    def test_higher_semantic_leads_to_higher_score(self) -> None:
        """With all other scores equal, higher semantic score → higher final score."""
        scorer = HybridScorer()
        ids = ["CAND_0000001", "CAND_0000002"]
        rr = _rerank(ids, sem_scores=[0.95, 0.55], ce_scores=[0.70, 0.70])
        fvs = [_fv("CAND_0000001"), _fv("CAND_0000002")]
        result = scorer.score_all(rr, fvs, top_n=2)
        assert result.ranked[0].candidate_id == "CAND_0000001"

    def test_candidate_id_preserved_in_output(self) -> None:
        scorer = HybridScorer()
        ids = [f"CAND_{i:07d}" for i in range(5)]
        rr = _rerank(ids)
        fvs = [_fv(cid) for cid in ids]
        result = scorer.score_all(rr, fvs, top_n=5)
        result_ids = {h.candidate_id for h in result.ranked}
        assert result_ids.issubset(set(ids))


# ---------------------------------------------------------------------------
# Tests — HybridScorer.score_one()
# ---------------------------------------------------------------------------


class TestScoreOne:
    def test_returns_hybrid_score(self) -> None:
        scorer = HybridScorer()
        fv = _fv("CAND_0000001")
        hs = scorer.score_one("CAND_0000001", 0.85, 0.70, fv)
        assert isinstance(hs, HybridScore)

    def test_candidate_id_matches(self) -> None:
        scorer = HybridScorer()
        fv = _fv("CAND_0000042")
        hs = scorer.score_one("CAND_0000042", 0.85, 0.70, fv)
        assert hs.candidate_id == "CAND_0000042"

    def test_formula_correctness(self) -> None:
        """Verify the exact weighted formula output for known inputs."""
        scorer = HybridScorer()
        sem, ce, exp, sig, edu, cert = 0.8, 0.7, 0.9, 0.6, 0.75, 0.5
        fv = _fv("CAND_0000001", exp=exp, edu=edu, cert=cert, sig=sig)
        hs = scorer.score_one("CAND_0000001", sem, ce, fv)
        expected = (
            0.40 * sem + 0.30 * ce + 0.10 * exp
            + 0.10 * sig + 0.05 * edu + 0.05 * cert
        )
        assert abs(hs.composite_score - expected) < 1e-5

    def test_perfect_candidate_scores_100(self) -> None:
        scorer = HybridScorer()
        fv = _fv("CAND_0000001", exp=1.0, edu=1.0, cert=1.0, sig=1.0)
        hs = scorer.score_one("CAND_0000001", 1.0, 1.0, fv)
        assert abs(hs.final_score - 100.0) < 1e-4

    def test_zero_candidate_scores_zero(self) -> None:
        scorer = HybridScorer()
        fv = _fv("CAND_0000001", exp=0.0, edu=0.0, cert=0.0, sig=0.0)
        hs = scorer.score_one("CAND_0000001", 0.0, 0.0, fv)
        assert abs(hs.final_score) < 1e-4

    def test_honeypot_flag_preserved(self) -> None:
        scorer = HybridScorer()
        fv = _fv("CAND_0000001", honeypot=True)
        hs = scorer.score_one("CAND_0000001", 0.5, 0.5, fv)
        assert hs.is_honeypot is True

    def test_final_score_in_zero_100(self) -> None:
        scorer = HybridScorer()
        fv = _fv("CAND_0000001", exp=0.5, edu=0.5, cert=0.5, sig=0.5)
        hs = scorer.score_one("CAND_0000001", 0.5, 0.5, fv)
        assert 0.0 <= hs.final_score <= 100.0


# ---------------------------------------------------------------------------
# Tests — formula weight invariants
# ---------------------------------------------------------------------------


class TestFormulaInvariants:
    def test_weights_sum_to_one(self) -> None:
        weights = [0.40, 0.30, 0.10, 0.10, 0.05, 0.05]
        assert abs(sum(weights) - 1.0) < 1e-10

    def test_semantic_weight_dominates(self) -> None:
        """Increasing semantic_similarity by 0.1 should move score more than any other."""
        scorer = HybridScorer()
        base_fv = _fv("CAND_0000001", exp=0.5, edu=0.5, cert=0.5, sig=0.5)
        base = scorer.score_one("CAND_0000001", 0.5, 0.5, base_fv).final_score
        delta_sem = scorer.score_one("CAND_0000001", 0.6, 0.5, base_fv).final_score - base
        delta_ce = scorer.score_one("CAND_0000001", 0.5, 0.6, base_fv).final_score - base
        delta_exp = scorer.score_one("CAND_0000001", 0.5, 0.5, _fv("CAND_0000001", exp=0.6, edu=0.5, cert=0.5, sig=0.5)).final_score - base
        assert delta_sem > delta_ce > delta_exp

    def test_ce_weight_second_largest(self) -> None:
        scorer = HybridScorer()
        base_fv = _fv("CAND_0000001", exp=0.5, edu=0.5, cert=0.5, sig=0.5)
        base = scorer.score_one("CAND_0000001", 0.5, 0.5, base_fv).final_score
        delta_ce = scorer.score_one("CAND_0000001", 0.5, 0.6, base_fv).final_score - base
        delta_edu = scorer.score_one("CAND_0000001", 0.5, 0.5, _fv("CAND_0000001", exp=0.5, edu=0.6, cert=0.5, sig=0.5)).final_score - base
        assert delta_ce > delta_edu
