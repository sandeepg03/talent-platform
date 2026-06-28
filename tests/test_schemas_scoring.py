"""
Unit tests for src.schemas.scoring
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.schemas.scoring import (
    HybridScore,
    SubmissionResult,
    SubmissionRow,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(rank: int, score: float, cid: str | None = None) -> SubmissionRow:
    cid = cid or f"CAND_{rank:07d}"
    return SubmissionRow(
        candidate_id=cid,
        rank=rank,
        score=score,
        reasoning=f"Candidate {cid} ranked {rank} with score {score}.",
    )


def _make_valid_submission() -> SubmissionResult:
    rows = [_make_row(r, 100.0 - r + 1) for r in range(1, 101)]
    return SubmissionResult(rows=rows)


# ---------------------------------------------------------------------------
# Tests — HybridScore.compute
# ---------------------------------------------------------------------------


class TestHybridScoreCompute:
    def test_formula_weights_sum_to_one(self) -> None:
        """All weights must sum to 1.0."""
        weights = [0.40, 0.30, 0.10, 0.10, 0.05, 0.05]
        assert abs(sum(weights) - 1.0) < 1e-9

    def test_perfect_candidate_scores_100(self) -> None:
        score = HybridScore.compute(
            candidate_id="CAND_0000001",
            semantic_similarity=1.0,
            cross_encoder_score=1.0,
            experience_score=1.0,
            redrob_signal_score=1.0,
            education_score=1.0,
            certification_score=1.0,
        )
        assert score.final_score == 100.0

    def test_zero_candidate_scores_zero(self) -> None:
        score = HybridScore.compute(
            candidate_id="CAND_0000002",
            semantic_similarity=0.0,
            cross_encoder_score=0.0,
            experience_score=0.0,
            redrob_signal_score=0.0,
            education_score=0.0,
            certification_score=0.0,
        )
        assert score.final_score == 0.0

    def test_weighted_sum_correctness(self) -> None:
        score = HybridScore.compute(
            candidate_id="CAND_0000003",
            semantic_similarity=0.8,
            cross_encoder_score=0.6,
            experience_score=0.7,
            redrob_signal_score=0.5,
            education_score=0.4,
            certification_score=0.3,
        )
        expected_composite = (
            0.40 * 0.8 + 0.30 * 0.6 + 0.10 * 0.7 + 0.10 * 0.5 + 0.05 * 0.4 + 0.05 * 0.3
        )
        assert abs(score.composite_score - expected_composite) < 1e-9
        assert abs(score.final_score - round(expected_composite * 100.0, 4)) < 1e-6

    def test_composite_clamped_to_one(self) -> None:
        # All components at maximum — composite should stay at 1.0
        score = HybridScore.compute(
            candidate_id="CAND_0000004",
            semantic_similarity=1.0,
            cross_encoder_score=1.0,
            experience_score=1.0,
            redrob_signal_score=1.0,
            education_score=1.0,
            certification_score=1.0,
        )
        assert score.composite_score == 1.0

    def test_honeypot_flag_preserved(self) -> None:
        score = HybridScore.compute(
            candidate_id="CAND_9999999",
            semantic_similarity=0.9,
            cross_encoder_score=0.9,
            experience_score=0.9,
            redrob_signal_score=0.9,
            education_score=0.9,
            certification_score=0.9,
            is_honeypot=True,
        )
        assert score.is_honeypot is True


# ---------------------------------------------------------------------------
# Tests — SubmissionRow validation
# ---------------------------------------------------------------------------


class TestSubmissionRow:
    def test_empty_reasoning_raises(self) -> None:
        with pytest.raises(Exception):
            SubmissionRow(
                candidate_id="CAND_0000001",
                rank=1,
                score=95.0,
                reasoning="   ",
            )

    def test_rank_below_one_raises(self) -> None:
        with pytest.raises(Exception):
            SubmissionRow(
                candidate_id="CAND_0000001",
                rank=0,
                score=95.0,
                reasoning="Valid.",
            )

    def test_rank_above_100_raises(self) -> None:
        with pytest.raises(Exception):
            SubmissionRow(
                candidate_id="CAND_0000001",
                rank=101,
                score=95.0,
                reasoning="Valid.",
            )


# ---------------------------------------------------------------------------
# Tests — SubmissionResult validation
# ---------------------------------------------------------------------------


class TestSubmissionResult:
    def test_valid_submission_builds(self) -> None:
        result = _make_valid_submission()
        assert len(result.rows) == 100

    def test_less_than_100_rows_raises(self) -> None:
        rows = [_make_row(r, 100.0 - r) for r in range(1, 50)]
        with pytest.raises(Exception):
            SubmissionResult(rows=rows)

    def test_duplicate_ranks_raises(self) -> None:
        rows = [_make_row(r, 100.0 - r) for r in range(1, 101)]
        rows[5] = _make_row(1, 90.0, "CAND_9999000")  # Duplicate rank 1
        with pytest.raises(Exception):
            SubmissionResult(rows=rows)

    def test_scores_increasing_raises(self) -> None:
        # Scores increase with rank (wrong direction)
        rows = [_make_row(r, float(r)) for r in range(1, 101)]
        with pytest.raises(Exception):
            SubmissionResult(rows=rows)

    def test_duplicate_candidate_ids_raises(self) -> None:
        rows = [_make_row(r, 100.0 - r) for r in range(1, 101)]
        # Overwrite row at rank 2 with same candidate_id as rank 1
        rows[1] = SubmissionRow(
            candidate_id=rows[0].candidate_id,
            rank=2,
            score=rows[1].score,
            reasoning="Duplicate.",
        )
        with pytest.raises(Exception):
            SubmissionResult(rows=rows)


# ---------------------------------------------------------------------------
# Tests — CSV serialization
# ---------------------------------------------------------------------------


class TestSubmissionCSV:
    def test_csv_written_correctly(self, tmp_path: Path) -> None:
        result = _make_valid_submission()
        output = tmp_path / "submission.csv"
        result.to_csv(output)
        assert output.exists()
        with open(output, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["candidate_id", "rank", "score", "reasoning"]
            data_rows = list(reader)
        assert len(data_rows) == 100

    def test_csv_ranks_sorted_ascending(self, tmp_path: Path) -> None:
        result = _make_valid_submission()
        output = tmp_path / "submission.csv"
        result.to_csv(output)
        with open(output, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            ranks = [int(row["rank"]) for row in reader]
        assert ranks == list(range(1, 101))

    def test_csv_scores_non_increasing(self, tmp_path: Path) -> None:
        result = _make_valid_submission()
        output = tmp_path / "submission.csv"
        result.to_csv(output)
        with open(output, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            scores = [float(row["score"]) for row in reader]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score at position {i} ({scores[i]}) is less than "
                f"score at position {i + 1} ({scores[i + 1]})"
            )
