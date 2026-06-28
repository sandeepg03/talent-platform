"""
Unit tests for src.schemas.candidate
"""
from __future__ import annotations

import json
from datetime import date

import pytest

from src.schemas.candidate import (
    CandidateProfile,
    CompanySize,
    EducationTier,
    SkillProficiency,
    WorkMode,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_CANDIDATE_DICT: dict = {
    "candidate_id": "CAND_0000001",
    "profile": {
        "anonymized_name": "Alice Smith",
        "headline": "Senior ML Engineer",
        "summary": "7 years of production ML systems.",
        "location": "Bangalore",
        "country": "India",
        "years_of_experience": 7.0,
        "current_title": "ML Engineer",
        "current_company": "Acme Corp",
        "current_company_size": "1001-5000",
        "current_industry": "AI/ML",
    },
    "career_history": [
        {
            "company": "Acme Corp",
            "title": "ML Engineer",
            "start_date": "2020-01-01",
            "end_date": None,
            "duration_months": 54,
            "is_current": True,
            "industry": "AI/ML",
            "company_size": "1001-5000",
            "description": "Built embedding-based retrieval systems using FAISS.",
        }
    ],
    "education": [
        {
            "institution": "IIT Bombay",
            "degree": "B.Tech",
            "field_of_study": "Computer Science",
            "start_year": 2013,
            "end_year": 2017,
            "grade": "9.0 CGPA",
            "tier": "tier_1",
        }
    ],
    "skills": [
        {"name": "FAISS", "proficiency": "expert", "endorsements": 30, "duration_months": 36},
        {"name": "Python", "proficiency": "expert", "endorsements": 50, "duration_months": 84},
    ],
    "certifications": [
        {"name": "AWS ML Specialty", "issuer": "Amazon", "year": 2022}
    ],
    "languages": [
        {"language": "English", "proficiency": "professional"}
    ],
    "redrob_signals": {
        "profile_completeness_score": 92.0,
        "signup_date": "2023-01-15",
        "last_active_date": "2024-06-01",
        "open_to_work_flag": True,
        "profile_views_received_30d": 45,
        "applications_submitted_30d": 3,
        "recruiter_response_rate": 0.85,
        "avg_response_time_hours": 2.5,
        "skill_assessment_scores": {"Python": 88.0, "Machine Learning": 91.0},
        "connection_count": 320,
        "endorsements_received": 78,
        "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 30.0, "max": 50.0},
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
        "github_activity_score": 72.0,
        "search_appearance_30d": 120,
        "saved_by_recruiters_30d": 8,
        "interview_completion_rate": 0.95,
        "offer_acceptance_rate": 0.80,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    },
}


@pytest.fixture
def valid_candidate() -> CandidateProfile:
    return CandidateProfile.model_validate(VALID_CANDIDATE_DICT)


# ---------------------------------------------------------------------------
# Tests — CandidateProfile validation
# ---------------------------------------------------------------------------


class TestCandidateProfileValidation:
    def test_valid_candidate_parses_successfully(self, valid_candidate: CandidateProfile) -> None:
        assert valid_candidate.candidate_id == "CAND_0000001"

    def test_candidate_id_pattern_enforced(self) -> None:
        bad = {**VALID_CANDIDATE_DICT, "candidate_id": "CAND_123"}
        with pytest.raises(Exception):
            CandidateProfile.model_validate(bad)

    def test_candidate_id_wrong_prefix(self) -> None:
        bad = {**VALID_CANDIDATE_DICT, "candidate_id": "CAND_ABCDEFG"}
        with pytest.raises(Exception):
            CandidateProfile.model_validate(bad)

    def test_years_of_experience_ceiling(self) -> None:
        bad_profile = {**VALID_CANDIDATE_DICT["profile"], "years_of_experience": 55.0}
        bad = {**VALID_CANDIDATE_DICT, "profile": bad_profile}
        with pytest.raises(Exception):
            CandidateProfile.model_validate(bad)

    def test_career_history_minimum_one_entry(self) -> None:
        bad = {**VALID_CANDIDATE_DICT, "career_history": []}
        with pytest.raises(Exception):
            CandidateProfile.model_validate(bad)

    def test_redrob_signal_response_rate_bounds(self) -> None:
        signals = {**VALID_CANDIDATE_DICT["redrob_signals"], "recruiter_response_rate": 1.5}
        bad = {**VALID_CANDIDATE_DICT, "redrob_signals": signals}
        with pytest.raises(Exception):
            CandidateProfile.model_validate(bad)

    def test_multiple_is_current_raises(self) -> None:
        extra_job = {**VALID_CANDIDATE_DICT["career_history"][0], "company": "Other Corp"}
        bad = {**VALID_CANDIDATE_DICT, "career_history": [
            VALID_CANDIDATE_DICT["career_history"][0], extra_job
        ]}
        with pytest.raises(Exception):
            CandidateProfile.model_validate(bad)


# ---------------------------------------------------------------------------
# Tests — Computed properties
# ---------------------------------------------------------------------------


class TestCandidateProperties:
    def test_skill_names_are_lowercased(self, valid_candidate: CandidateProfile) -> None:
        assert "faiss" in valid_candidate.skill_names
        assert "python" in valid_candidate.skill_names

    def test_all_titles(self, valid_candidate: CandidateProfile) -> None:
        assert "ML Engineer" in valid_candidate.all_titles

    def test_total_career_months(self, valid_candidate: CandidateProfile) -> None:
        assert valid_candidate.total_career_months == 54

    def test_has_ai_ml_experience_true(self, valid_candidate: CandidateProfile) -> None:
        assert valid_candidate.has_ai_ml_experience is True

    def test_has_ai_ml_experience_false_for_non_ai_profile(self) -> None:
        career = [
            {
                **VALID_CANDIDATE_DICT["career_history"][0],
                "description": "Managed spreadsheets and filed reports.",
            }
        ]
        no_ai_skills = [
            {"name": "Excel", "proficiency": "expert", "endorsements": 10, "duration_months": 60}
        ]
        modified = {**VALID_CANDIDATE_DICT, "career_history": career, "skills": no_ai_skills}
        candidate = CandidateProfile.model_validate(modified)
        assert candidate.has_ai_ml_experience is False


# ---------------------------------------------------------------------------
# Tests — JSONL round-trip
# ---------------------------------------------------------------------------


class TestJSONLRoundTrip:
    def test_roundtrip_json(self, valid_candidate: CandidateProfile) -> None:
        json_str = valid_candidate.model_dump_json()
        restored = CandidateProfile.model_validate_json(json_str)
        assert restored.candidate_id == valid_candidate.candidate_id

    def test_education_tier_enum(self, valid_candidate: CandidateProfile) -> None:
        assert valid_candidate.education[0].tier == EducationTier.TIER_1

    def test_work_mode_enum(self, valid_candidate: CandidateProfile) -> None:
        assert valid_candidate.redrob_signals.preferred_work_mode == WorkMode.HYBRID
