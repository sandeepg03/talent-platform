"""
Pydantic v2 models for candidate profiles.

These models mirror ``candidate_schema.json`` exactly and serve as the single
source of truth for data shapes across the entire pipeline.  All downstream
modules import from here — no ad-hoc dict access anywhere else.
"""

from __future__ import annotations

import re
from datetime import date
from enum import Enum
from typing import Annotated

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

CANDIDATE_ID_PATTERN: re.Pattern[str] = re.compile(r"^CAND_[0-9]{7}$")


class CompanySize(str, Enum):
    SIZE_1_10 = "1-10"
    SIZE_11_50 = "11-50"
    SIZE_51_200 = "51-200"
    SIZE_201_500 = "201-500"
    SIZE_501_1000 = "501-1000"
    SIZE_1001_5000 = "1001-5000"
    SIZE_5001_10000 = "5001-10000"
    SIZE_10001_PLUS = "10001+"


class SkillProficiency(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class LanguageProficiency(str, Enum):
    BASIC = "basic"
    CONVERSATIONAL = "conversational"
    PROFESSIONAL = "professional"
    NATIVE = "native"


class EducationTier(str, Enum):
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4 = "tier_4"
    UNKNOWN = "unknown"


class WorkMode(str, Enum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    FLEXIBLE = "flexible"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class Profile(BaseModel):
    """Top-level candidate summary information."""

    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: Annotated[float, Field(ge=0.0, le=50.0)]
    current_title: str
    current_company: str
    current_company_size: CompanySize
    current_industry: str

    model_config = {"frozen": True}


class CareerEntry(BaseModel):
    """A single position in a candidate's work history."""

    company: str
    title: str
    start_date: date
    end_date: date | None
    duration_months: Annotated[int, Field(ge=0)]
    is_current: bool
    industry: str
    company_size: CompanySize
    description: str

    model_config = {"frozen": True}


class Education(BaseModel):
    """An educational qualification."""

    institution: str
    degree: str
    field_of_study: str
    start_year: Annotated[int, Field(ge=1970, le=2030)]
    end_year: Annotated[int, Field(ge=1970, le=2035)]
    grade: str | None = None
    tier: EducationTier = EducationTier.UNKNOWN

    model_config = {"frozen": True}


class Skill(BaseModel):
    """A single skill with proficiency and endorsement metadata."""

    name: str
    proficiency: SkillProficiency
    endorsements: Annotated[int, Field(ge=0)]
    duration_months: Annotated[int, Field(ge=0)] = 0

    model_config = {"frozen": True}


class Certification(BaseModel):
    """A professional certification."""

    name: str
    issuer: str
    year: int

    model_config = {"frozen": True}


class Language(BaseModel):
    """A spoken language with proficiency level."""

    language: str
    proficiency: LanguageProficiency

    model_config = {"frozen": True}


class SalaryRange(BaseModel):
    """Expected salary band in INR Lakhs Per Annum."""

    min: Annotated[float, Field(ge=0.0)]
    max: Annotated[float, Field(ge=0.0)]

    model_config = {"frozen": True}


class RedrobSignals(BaseModel):
    """
    Platform behavioral signals from the Redrob ecosystem.
    These 23 signals measure real engagement and availability,
    not just static profile data.
    """

    profile_completeness_score: Annotated[float, Field(ge=0.0, le=100.0)]
    signup_date: date
    last_active_date: date
    open_to_work_flag: bool
    profile_views_received_30d: Annotated[int, Field(ge=0)]
    applications_submitted_30d: Annotated[int, Field(ge=0)]
    recruiter_response_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    avg_response_time_hours: Annotated[float, Field(ge=0.0)]
    skill_assessment_scores: dict[str, Annotated[float, Field(ge=0.0, le=100.0)]]
    connection_count: Annotated[int, Field(ge=0)]
    endorsements_received: Annotated[int, Field(ge=0)]
    notice_period_days: Annotated[int, Field(ge=0, le=180)]
    expected_salary_range_inr_lpa: SalaryRange
    preferred_work_mode: WorkMode
    willing_to_relocate: bool
    github_activity_score: Annotated[float, Field(ge=-1.0, le=100.0)]
    search_appearance_30d: Annotated[int, Field(ge=0)]
    saved_by_recruiters_30d: Annotated[int, Field(ge=0)]
    interview_completion_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    offer_acceptance_rate: Annotated[float, Field(ge=-1.0, le=1.0)]
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------


class CandidateProfile(BaseModel):
    """
    Full candidate record as stored in ``candidates.jsonl``.

    This is the canonical model used throughout the entire pipeline.
    Downstream modules must accept ``CandidateProfile`` objects — never
    raw dicts.
    """

    candidate_id: str
    profile: Profile
    career_history: Annotated[list[CareerEntry], Field(min_length=1, max_length=10)]
    education: Annotated[list[Education], Field(max_length=5)]
    skills: list[Skill]
    certifications: list[Certification] = Field(default_factory=list)
    languages: list[Language] = Field(default_factory=list)
    redrob_signals: RedrobSignals

    model_config = {"frozen": True}

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id(cls, v: str) -> str:
        if not CANDIDATE_ID_PATTERN.match(v):
            raise ValueError(f"candidate_id must match CAND_XXXXXXX (7 digits), got: {v!r}")
        return v

    @model_validator(mode="after")
    def validate_career_current_consistency(self) -> "CandidateProfile":
        """At most one career entry should have is_current=True."""
        current_entries = [e for e in self.career_history if e.is_current]
        if len(current_entries) > 1:
            raise ValueError(
                f"Multiple career entries marked is_current=True for candidate {self.candidate_id}"
            )
        return self

    # ------------------------------------------------------------------
    # Convenience properties (computed, not stored)
    # ------------------------------------------------------------------

    @property
    def skill_names(self) -> list[str]:
        """Lower-cased skill names for fast membership checks."""
        return [s.name.lower() for s in self.skills]

    @property
    def all_titles(self) -> list[str]:
        """All job titles held across career history."""
        return [e.title for e in self.career_history]

    @property
    def total_career_months(self) -> int:
        """Sum of all career entry durations."""
        return sum(e.duration_months for e in self.career_history)

    @property
    def has_ai_ml_experience(self) -> bool:
        """
        Heuristic: returns True if any career description mentions
        AI/ML-related production keywords.
        """
        ai_keywords = {
            "embedding",
            "vector",
            "faiss",
            "retrieval",
            "ranking",
            "transformer",
            "bert",
            "llm",
            "fine-tun",
            "pytorch",
            "tensorflow",
            "sklearn",
            "machine learning",
            "deep learning",
            "nlp",
            "neural",
            "sentence-transformer",
            "bge",
            "e5",
            "openai",
            "langchain",
            "huggingface",
            "rag",
            "lora",
            "qlora",
            "peft",
        }
        combined = " ".join(e.description.lower() for e in self.career_history)
        combined += " " + " ".join(s.name.lower() for s in self.skills)
        return any(kw in combined for kw in ai_keywords)
