"""
Pydantic v2 models for intermediate and final scoring results.

Every stage of the pipeline produces or consumes one of these models.
This guarantees that data flows through the system with full type-safety
and that the final ``SubmissionRow`` can be written directly to CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Per-stage score containers
# ---------------------------------------------------------------------------


class SemanticScore(BaseModel):
    """Output of the FAISS retrieval stage."""

    candidate_id: str
    faiss_rank: int  # Rank within the FAISS top-K result set
    cosine_similarity: Annotated[float, Field(ge=0.0, le=1.0)]

    model_config = {"frozen": True}


class CrossEncoderScore(BaseModel):
    """Output of the cross-encoder reranking stage."""

    candidate_id: str
    raw_score: float  # Logit output from the cross-encoder
    normalized_score: Annotated[float, Field(ge=0.0, le=1.0)]

    model_config = {"frozen": True}


class FeatureVector(BaseModel):
    """
    Feature vector produced by ``FeatureEngineer`` for a single candidate.

    All values are normalized to [0, 1] before this model is populated
    so that the ``HybridScorer`` can apply weights directly.
    """

    candidate_id: str

    # Sub-scores (all in [0, 1])
    experience_score: Annotated[float, Field(ge=0.0, le=1.0)]
    education_score: Annotated[float, Field(ge=0.0, le=1.0)]
    certification_score: Annotated[float, Field(ge=0.0, le=1.0)]
    redrob_signal_score: Annotated[float, Field(ge=0.0, le=1.0)]

    # Decomposed Redrob sub-components (stored for explainability)
    signal_open_to_work: float
    signal_response_rate: float
    signal_interview_completion: float
    signal_profile_completeness: float
    signal_recency: float
    signal_github: float
    signal_assessment_avg: float
    signal_saved_by_recruiters: float

    # Honeypot flag — True means this candidate has impossible profile signals
    is_honeypot: bool = False

    # Raw supporting data for explanation generator
    years_of_experience: float
    highest_education_degree: str
    matched_must_have_skills: list[str] = Field(default_factory=list)
    matched_nice_to_have_skills: list[str] = Field(default_factory=list)
    cert_names: list[str] = Field(default_factory=list)

    model_config = {"frozen": True}


class HybridScore(BaseModel):
    """
    Final composite score for one candidate, before normalization to 0–100.

    Formula:
        composite = 0.40 × semantic + 0.30 × cross_encoder
                  + 0.10 × experience + 0.10 × redrob_signal
                  + 0.05 × education  + 0.05 × certification
    """

    candidate_id: str

    # Component scores (all in [0, 1])
    semantic_similarity: Annotated[float, Field(ge=0.0, le=1.0)]
    cross_encoder_score: Annotated[float, Field(ge=0.0, le=1.0)]
    experience_score: Annotated[float, Field(ge=0.0, le=1.0)]
    redrob_signal_score: Annotated[float, Field(ge=0.0, le=1.0)]
    education_score: Annotated[float, Field(ge=0.0, le=1.0)]
    certification_score: Annotated[float, Field(ge=0.0, le=1.0)]

    # Composite (weighted sum, pre-normalization, in [0, 1])
    composite_score: Annotated[float, Field(ge=0.0, le=1.0)]

    # Final score normalized to [0, 100]
    final_score: Annotated[float, Field(ge=0.0, le=100.0)]

    # Honeypot candidates are scored but excluded from top-100
    is_honeypot: bool = False

    model_config = {"frozen": True}

    @classmethod
    def compute(
        cls,
        candidate_id: str,
        semantic_similarity: float,
        cross_encoder_score: float,
        experience_score: float,
        redrob_signal_score: float,
        education_score: float,
        certification_score: float,
        is_honeypot: bool = False,
    ) -> "HybridScore":
        """
        Factory method that applies the scoring formula and normalizes to [0, 100].

        All input scores must already be in [0, 1].
        """
        composite = (
            0.40 * semantic_similarity
            + 0.30 * cross_encoder_score
            + 0.10 * experience_score
            + 0.10 * redrob_signal_score
            + 0.05 * education_score
            + 0.05 * certification_score
        )
        composite = max(0.0, min(1.0, composite))
        final = round(composite * 100.0, 4)

        return cls(
            candidate_id=candidate_id,
            semantic_similarity=semantic_similarity,
            cross_encoder_score=cross_encoder_score,
            experience_score=experience_score,
            redrob_signal_score=redrob_signal_score,
            education_score=education_score,
            certification_score=certification_score,
            composite_score=composite,
            final_score=final,
            is_honeypot=is_honeypot,
        )


# ---------------------------------------------------------------------------
# Submission row
# ---------------------------------------------------------------------------


class SubmissionRow(BaseModel):
    """
    A single row in the final submission CSV.

    Columns: candidate_id, rank, score, reasoning
    Constraints enforced by validate_submission.py:
      - rank: integer in [1, 100], unique
      - score: float, non-increasing as rank increases
      - reasoning: non-empty string
    """

    candidate_id: str
    rank: Annotated[int, Field(ge=1, le=100)]
    score: Annotated[float, Field(ge=0.0, le=100.0)]
    reasoning: str

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def validate_reasoning_not_empty(self) -> "SubmissionRow":
        if not self.reasoning.strip():
            raise ValueError(f"reasoning must not be empty for candidate {self.candidate_id}")
        return self


class SubmissionResult(BaseModel):
    """
    Container for the full top-100 submission.

    Provides serialization to CSV in the exact format required by
    ``validate_submission.py``.
    """

    rows: Annotated[list[SubmissionRow], Field(min_length=100, max_length=100)]

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def validate_ranks_unique_and_complete(self) -> "SubmissionResult":
        ranks = [r.rank for r in self.rows]
        if sorted(ranks) != list(range(1, 101)):
            raise ValueError("Ranks must be exactly 1 through 100, each appearing once.")
        return self

    @model_validator(mode="after")
    def validate_scores_non_increasing(self) -> "SubmissionResult":
        sorted_rows = sorted(self.rows, key=lambda r: r.rank)
        for i in range(len(sorted_rows) - 1):
            if sorted_rows[i].score < sorted_rows[i + 1].score:
                raise ValueError(
                    f"Score must be non-increasing: rank {sorted_rows[i].rank} "
                    f"({sorted_rows[i].score}) < rank {sorted_rows[i + 1].rank} "
                    f"({sorted_rows[i + 1].score})"
                )
        return self

    @model_validator(mode="after")
    def validate_candidate_ids_unique(self) -> "SubmissionResult":
        ids = [r.candidate_id for r in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("All candidate_ids in the submission must be unique.")
        return self

    def to_csv(self, path: Path) -> None:
        """
        Write the submission to a UTF-8 encoded CSV at ``path``.

        Rows are written sorted by rank (ascending).  This is the only
        public write method — all CSV generation must go through here.
        """
        sorted_rows = sorted(self.rows, key=lambda r: r.rank)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for row in sorted_rows:
                writer.writerow([row.candidate_id, row.rank, f"{row.score:.4f}", row.reasoning])
