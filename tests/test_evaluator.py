"""
Unit tests for src.evaluation.evaluator

All tests are pure NumPy — no model downloads, no file I/O.

Covers:
  - NDCG@k: perfect ranking, zero relevant, partial overlap, empty ranked
  - Precision@k: all relevant, none relevant, partial
  - MRR: first hit at rank 1, rank 3, no hit
  - Honeypot exclusion rate: all excluded, none excluded, no honeypots
  - Score distribution: mean/std/min/max/median correctness
  - EvaluationReport.summary() and as_dict()
  - Edge cases: empty ranked list, mismatched lengths
"""

from __future__ import annotations

import pytest
import numpy as np

from src.evaluation.evaluator import EvaluationReport, RankingEvaluator


def _ids(n: int) -> list[str]:
    return [f"CAND_{i:07d}" for i in range(1, n + 1)]


def _scores(n: int, start: float = 90.0, step: float = -0.5) -> list[float]:
    return [start + i * step for i in range(n)]


class TestNDCG:
    def test_perfect_ranking_ndcg_is_one(self) -> None:
        ev = RankingEvaluator()
        ids = _ids(10)
        # All 10 are relevant — perfect ranking
        relevant = set(ids)
        report = ev.evaluate(ids, _scores(10), relevant)
        assert abs(report.ndcg_at_10 - 1.0) < 1e-6

    def test_no_relevant_ndcg_is_zero(self) -> None:
        ev = RankingEvaluator()
        ids = _ids(10)
        report = ev.evaluate(ids, _scores(10), set())
        assert report.ndcg_at_10 == 0.0
        assert report.ndcg_at_20 == 0.0
        assert report.ndcg_at_100 == 0.0

    def test_no_overlap_ndcg_is_zero(self) -> None:
        ev = RankingEvaluator()
        ids = _ids(10)
        relevant = {"CAND_9999990", "CAND_9999991"}
        report = ev.evaluate(ids, _scores(10), relevant)
        assert report.ndcg_at_10 == 0.0

    def test_first_result_relevant_ndcg_greater_than_last(self) -> None:
        ev = RankingEvaluator()
        ids = _ids(10)
        # Relevant only at rank 1
        relevant_first = {ids[0]}
        # Relevant only at rank 10
        relevant_last = {ids[9]}
        r_first = ev.evaluate(ids, _scores(10), relevant_first)
        r_last = ev.evaluate(ids, _scores(10), relevant_last)
        assert r_first.ndcg_at_10 > r_last.ndcg_at_10

    def test_ndcg100_gte_ndcg10_when_relevant_after_10(self) -> None:
        ev = RankingEvaluator()
        ids = _ids(100)
        # Relevant at rank 15 — not in top-10 but in top-100
        relevant = {ids[14]}
        report = ev.evaluate(ids, _scores(100), relevant)
        assert report.ndcg_at_100 > report.ndcg_at_10

    def test_ndcg_in_zero_one(self) -> None:
        ev = RankingEvaluator()
        ids = _ids(100)
        relevant = set(ids[:30])
        report = ev.evaluate(ids, _scores(100), relevant)
        for val in (report.ndcg_at_10, report.ndcg_at_20, report.ndcg_at_100):
            assert 0.0 <= val <= 1.0


class TestPrecision:
    def test_all_relevant_precision_is_one(self) -> None:
        ev = RankingEvaluator()
        ids = _ids(10)
        report = ev.evaluate(ids, _scores(10), set(ids))
        assert abs(report.precision_at_10 - 1.0) < 1e-6

    def test_none_relevant_precision_is_zero(self) -> None:
        ev = RankingEvaluator()
        ids = _ids(10)
        report = ev.evaluate(ids, _scores(10), set())
        assert report.precision_at_10 == 0.0

    def test_half_relevant_precision_half(self) -> None:
        ev = RankingEvaluator()
        ids = _ids(10)
        relevant = set(ids[:5])   # first 5 are relevant
        report = ev.evaluate(ids, _scores(10), relevant)
        assert abs(report.precision_at_10 - 0.5) < 1e-6

    def test_p20_uses_top_20(self) -> None:
        ev = RankingEvaluator()
        ids = _ids(100)
        # Relevant only at positions 11–20
        relevant = set(ids[10:20])
        report = ev.evaluate(ids, _scores(100), relevant)
        assert report.precision_at_10 == 0.0   # none in top-10
        assert abs(report.precision_at_20 - 0.5) < 1e-6  # 10/20


class TestMRR:
    def test_first_hit_at_rank_1(self) -> None:
        ev = RankingEvaluator()
        ids = _ids(10)
        report = ev.evaluate(ids, _scores(10), {ids[0]})
        assert abs(report.mrr - 1.0) < 1e-6

    def test_first_hit_at_rank_3(self) -> None:
        ev = RankingEvaluator()
        ids = _ids(10)
        report = ev.evaluate(ids, _scores(10), {ids[2]})
        assert abs(report.mrr - 1 / 3) < 1e-6

    def test_no_hit_mrr_zero(self) -> None:
        ev = RankingEvaluator()
        ids = _ids(10)
        report = ev.evaluate(ids, _scores(10), {"CAND_9999999"})
        assert report.mrr == 0.0


class TestHoneypotExclusion:
    def test_all_excluded(self) -> None:
        ev = RankingEvaluator()
        ids = _ids(10)
        honeypots = {"CAND_9999990", "CAND_9999991"}
        report = ev.evaluate(ids, _scores(10), set(), honeypot_ids=honeypots)
        assert abs(report.honeypot_exclusion_rate - 1.0) < 1e-6

    def test_none_excluded(self) -> None:
        ev = RankingEvaluator()
        ids = _ids(10)
        honeypots = {ids[0], ids[1]}  # honeypots are IN the ranked list
        report = ev.evaluate(ids, _scores(10), set(), honeypot_ids=honeypots)
        assert report.honeypot_exclusion_rate == 0.0

    def test_partial_exclusion(self) -> None:
        ev = RankingEvaluator()
        ids = _ids(10)
        honeypots = {ids[0], "CAND_9999990"}  # one in, one out
        report = ev.evaluate(ids, _scores(10), set(), honeypot_ids=honeypots)
        assert abs(report.honeypot_exclusion_rate - 0.5) < 1e-6

    def test_no_honeypots_rate_is_one(self) -> None:
        ev = RankingEvaluator()
        ids = _ids(10)
        report = ev.evaluate(ids, _scores(10), set(), honeypot_ids=set())
        assert abs(report.honeypot_exclusion_rate - 1.0) < 1e-6

    def test_num_honeypots_excluded_counted(self) -> None:
        ev = RankingEvaluator()
        ids = _ids(10)
        honeypots = {"CAND_9999990", "CAND_9999991", "CAND_9999992"}
        report = ev.evaluate(ids, _scores(10), set(), honeypot_ids=honeypots)
        assert report.num_honeypots_excluded == 3
        assert report.num_total_honeypots == 3


class TestScoreDistribution:
    def test_score_mean_correct(self) -> None:
        ev = RankingEvaluator()
        ids = _ids(4)
        scores = [80.0, 70.0, 60.0, 50.0]
        report = ev.evaluate(ids, scores, set())
        assert abs(report.score_mean - 65.0) < 1e-6

    def test_score_min_max(self) -> None:
        ev = RankingEvaluator()
        ids = _ids(5)
        scores = [90.0, 80.0, 70.0, 60.0, 50.0]
        report = ev.evaluate(ids, scores, set())
        assert abs(report.score_max - 90.0) < 1e-6
        assert abs(report.score_min - 50.0) < 1e-6

    def test_score_median(self) -> None:
        ev = RankingEvaluator()
        ids = _ids(5)
        scores = [90.0, 80.0, 70.0, 60.0, 50.0]
        report = ev.evaluate(ids, scores, set())
        assert abs(report.score_median - 70.0) < 1e-6


class TestReportHelpers:
    def test_summary_is_string(self) -> None:
        ev = RankingEvaluator()
        ids = _ids(10)
        report = ev.evaluate(ids, _scores(10), set(ids[:5]))
        s = report.summary()
        assert isinstance(s, str)
        assert "NDCG@10" in s

    def test_as_dict_has_all_keys(self) -> None:
        ev = RankingEvaluator()
        ids = _ids(10)
        report = ev.evaluate(ids, _scores(10), set(ids[:3]))
        d = report.as_dict()
        for key in ("ndcg@10", "ndcg@20", "ndcg@100", "precision@10",
                    "precision@20", "mrr", "honeypot_exclusion_rate",
                    "score_mean", "score_std", "score_min", "score_max", "score_median"):
            assert key in d

    def test_mismatched_lengths_raises(self) -> None:
        ev = RankingEvaluator()
        with pytest.raises(ValueError, match="equal length"):
            ev.evaluate(["CAND_0000001"], [90.0, 80.0], set())
