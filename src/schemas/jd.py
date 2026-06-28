"""
Pydantic v2 models for a parsed Job Description.

The JD parser produces a ``StructuredJD`` from the raw docx text.
Every downstream module (embedding engine, feature engineer, scorer)
consumes ``StructuredJD`` — never raw strings.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ExperienceLevel(str, Enum):
    JUNIOR = "junior"        # 0-2 yrs
    MID = "mid"              # 3-5 yrs
    SENIOR = "senior"        # 5-9 yrs
    PRINCIPAL = "principal"  # 9+ yrs


class RequirementPriority(str, Enum):
    MUST_HAVE = "must_have"
    NICE_TO_HAVE = "nice_to_have"
    DISQUALIFIER = "disqualifier"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class SkillRequirement(BaseModel):
    """A single skill requirement extracted from the JD."""

    name: str
    priority: RequirementPriority
    context: str = ""  # Free-text rationale from the JD

    model_config = {"frozen": True}


class ExperienceRequirement(BaseModel):
    """Years-of-experience band stated in the JD."""

    min_years: Annotated[float, Field(ge=0.0)]
    max_years: Annotated[float, Field(ge=0.0)]
    preferred_level: ExperienceLevel
    notes: str = ""

    model_config = {"frozen": True}


class LocationRequirement(BaseModel):
    """Location and work-mode constraints from the JD."""

    cities: list[str] = Field(default_factory=list)
    country: str = ""
    work_modes: list[str] = Field(default_factory=list)
    relocation_open: bool = False

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Root JD model
# ---------------------------------------------------------------------------


class StructuredJD(BaseModel):
    """
    Machine-readable representation of a Job Description.

    Produced by ``src.parsers.jd_parser.JDParser`` and consumed by:
    - ``src.embeddings.engine.EmbeddingEngine``  (text → embedding)
    - ``src.features.engineer.FeatureEngineer``  (requirement matching)
    - ``src.scoring.hybrid_scorer.HybridScorer`` (weight application)
    - ``src.explanation.generator.ExplanationGenerator`` (reasoning)
    """

    # Identity
    title: str
    company: str
    raw_text: str  # Full original text, used for embedding

    # Structured requirements
    must_have_skills: list[SkillRequirement] = Field(default_factory=list)
    nice_to_have_skills: list[SkillRequirement] = Field(default_factory=list)
    disqualifying_patterns: list[str] = Field(default_factory=list)

    experience: ExperienceRequirement
    location: LocationRequirement

    # Domain signals derived from JD text
    key_technologies: list[str] = Field(default_factory=list)
    embedding_text: str = ""  # Optimized text fed to embedding model

    model_config = {"frozen": False}  # Mutable: parser populates fields incrementally

    def build_embedding_text(self) -> str:
        """
        Construct the canonical text representation sent to the embedding model.

        Prioritises must-have skills and key technologies.
        Appends a structured summary to improve semantic alignment.
        """
        parts: list[str] = [
            f"Job Title: {self.title}",
            f"Company: {self.company}",
        ]

        if self.must_have_skills:
            skill_str = ", ".join(s.name for s in self.must_have_skills)
            parts.append(f"Required skills: {skill_str}")

        if self.nice_to_have_skills:
            nh_str = ", ".join(s.name for s in self.nice_to_have_skills)
            parts.append(f"Preferred skills: {nh_str}")

        if self.key_technologies:
            parts.append(f"Key technologies: {', '.join(self.key_technologies)}")

        exp = self.experience
        parts.append(
            f"Experience required: {exp.min_years}-{exp.max_years} years "
            f"({exp.preferred_level.value} level)"
        )

        # Append subset of raw text (first 1500 chars) for dense retrieval context
        parts.append(self.raw_text[:1500])

        return "\n".join(parts)

    @property
    def all_required_skill_names(self) -> list[str]:
        """Lower-cased names of all must-have skills."""
        return [s.name.lower() for s in self.must_have_skills]

    @property
    def all_preferred_skill_names(self) -> list[str]:
        """Lower-cased names of all nice-to-have skills."""
        return [s.name.lower() for s in self.nice_to_have_skills]
