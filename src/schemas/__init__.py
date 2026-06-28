"""
src.schemas — Public API for all data models used across the pipeline.

Import from here rather than from individual modules to maintain
a stable interface that survives internal refactors.
"""

from src.schemas.candidate import (
    CANDIDATE_ID_PATTERN,
    CandidateProfile,
    CareerEntry,
    Certification,
    CompanySize,
    Education,
    EducationTier,
    Language,
    LanguageProficiency,
    Profile,
    RedrobSignals,
    SalaryRange,
    Skill,
    SkillProficiency,
    WorkMode,
)
from src.schemas.jd import (
    ExperienceLevel,
    ExperienceRequirement,
    LocationRequirement,
    RequirementPriority,
    SkillRequirement,
    StructuredJD,
)
from src.schemas.scoring import (
    CrossEncoderScore,
    FeatureVector,
    HybridScore,
    SemanticScore,
    SubmissionResult,
    SubmissionRow,
)

__all__ = [
    # Candidate
    "CANDIDATE_ID_PATTERN",
    "CandidateProfile",
    "CareerEntry",
    "Certification",
    "CompanySize",
    "Education",
    "EducationTier",
    "Language",
    "LanguageProficiency",
    "Profile",
    "RedrobSignals",
    "SalaryRange",
    "Skill",
    "SkillProficiency",
    "WorkMode",
    # JD
    "ExperienceLevel",
    "ExperienceRequirement",
    "LocationRequirement",
    "RequirementPriority",
    "SkillRequirement",
    "StructuredJD",
    # Scoring
    "CrossEncoderScore",
    "FeatureVector",
    "HybridScore",
    "SemanticScore",
    "SubmissionResult",
    "SubmissionRow",
]
