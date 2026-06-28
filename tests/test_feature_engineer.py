"""
Unit tests for src.features.feature_engineer

All tests use synthetic CandidateProfile fixtures built directly from Pydantic models.
No model downloads, no file I/O — runs in milliseconds.

Covers:
  - FeatureEngineer.extract(): returns well-formed FeatureVector
  - _experience_score(): below/within/above band, no requirement
  - _education_score(): tier weights, degree bonuses, empty education
  - _certification_score(): relevant vs irrelevant certs, cap at 1.0
  - _redrob_score(): all 8 sub-components, github sentinel, empty assessments
  - _is_honeypot(): all 4 heuristic rules
  - Skill matching: must-have and nice-to-have intersection
"""

from __future__ import annotations

import datetime
from typing import Any

from src.features.feature_engineer import FeatureEngineer
from src.schemas.candidate import (
    CandidateProfile,
    CareerEntry,
    Certification,
    CompanySize,
    Education,
    EducationTier,
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
from src.schemas.scoring import FeatureVector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TODAY = datetime.date.today()
YESTERDAY = TODAY - datetime.timedelta(days=1)
LONG_AGO = TODAY - datetime.timedelta(days=200)


def _signals(**overrides: Any) -> RedrobSignals:
    defaults: dict[str, Any] = dict(
        profile_completeness_score=80.0,
        signup_date=datetime.date(2022, 1, 1),
        last_active_date=YESTERDAY,
        open_to_work_flag=True,
        profile_views_received_30d=10,
        applications_submitted_30d=3,
        recruiter_response_rate=0.8,
        avg_response_time_hours=4.0,
        skill_assessment_scores={"python": 75.0, "sql": 60.0},
        connection_count=200,
        endorsements_received=15,
        notice_period_days=30,
        expected_salary_range_inr_lpa=SalaryRange(min=20.0, max=35.0),
        preferred_work_mode=WorkMode.HYBRID,
        willing_to_relocate=True,
        github_activity_score=55.0,
        search_appearance_30d=50,
        saved_by_recruiters_30d=3,
        interview_completion_rate=0.85,
        offer_acceptance_rate=0.9,
        verified_email=True,
        verified_phone=True,
        linkedin_connected=True,
    )
    defaults.update(overrides)
    return RedrobSignals(**defaults)


def _career(**overrides: Any) -> CareerEntry:
    defaults: dict[str, Any] = dict(
        company="Acme AI",
        title="ML Engineer",
        start_date=datetime.date(2019, 1, 1),
        end_date=None,
        duration_months=60,
        is_current=True,
        industry="Technology",
        company_size=CompanySize.SIZE_201_500,
        description="Built FAISS retrieval systems and NLP pipelines.",
    )
    defaults.update(overrides)
    return CareerEntry(**defaults)


def _edu(
    degree: str = "Bachelor of Technology",
    tier: EducationTier = EducationTier.TIER_2,
    institution: str = "IIT Delhi",
) -> Education:
    return Education(
        institution=institution,
        degree=degree,
        field_of_study="Computer Science",
        start_year=2014,
        end_year=2018,
        tier=tier,
    )


def _skill(name: str, proficiency: SkillProficiency = SkillProficiency.ADVANCED) -> Skill:
    return Skill(name=name, proficiency=proficiency, endorsements=5, duration_months=24)


def _cert(name: str, issuer: str = "Coursera", year: int = 2023) -> Certification:
    return Certification(name=name, issuer=issuer, year=year)


def _candidate(
    candidate_id: str = "CAND_0000001",
    years: float = 6.0,
    skills: list[str] | None = None,
    education: list[Education] | None = None,
    certifications: list[Certification] | None = None,
    signals: RedrobSignals | None = None,
    career: list[CareerEntry] | None = None,
) -> CandidateProfile:
    skills_list = [_skill(s) for s in (skills or ["python", "faiss", "pytorch"])]
    edu_list = [_edu()] if education is None else education
    cert_list = [] if certifications is None else certifications
    return CandidateProfile(
        candidate_id=candidate_id,
        profile=Profile(
            anonymized_name="Candidate A",
            headline="ML Engineer",
            summary="Experienced ML practitioner.",
            location="Bangalore",
            country="India",
            years_of_experience=years,
            current_title="Senior ML Engineer",
            current_company="Acme AI",
            current_company_size=CompanySize.SIZE_201_500,
            current_industry="Technology",
        ),
        career_history=career or [_career()],
        education=edu_list,
        skills=skills_list,
        certifications=cert_list,
        redrob_signals=signals or _signals(),
    )


def _jd(
    min_years: float = 4.0,
    max_years: float = 8.0,
    must_have: list[str] | None = None,
    nice_to_have: list[str] | None = None,
    technologies: list[str] | None = None,
) -> StructuredJD:
    must = [
        SkillRequirement(
            name=s,
            priority=RequirementPriority.MUST_HAVE,
            context=f"Required: {s}",
        )
        for s in (must_have or ["python", "faiss", "pytorch"])
    ]
    nice = [
        SkillRequirement(
            name=s,
            priority=RequirementPriority.NICE_TO_HAVE,
            context=f"Preferred: {s}",
        )
        for s in (nice_to_have or ["lora", "kubernetes"])
    ]
    return StructuredJD(
        title="Senior AI Engineer",
        company="Redrob AI",
        raw_text="Senior AI Engineer role.",
        must_have_skills=must,
        nice_to_have_skills=nice,
        disqualifying_patterns=[],
        experience=ExperienceRequirement(
            min_years=min_years,
            max_years=max_years,
            preferred_level=ExperienceLevel.SENIOR,
        ),
        location=LocationRequirement(),
        key_technologies=technologies or ["python", "faiss", "pytorch", "nlp"],
        embedding_text="Senior AI Engineer with FAISS NLP Python.",
    )


# ---------------------------------------------------------------------------
# Tests — extract() top-level
# ---------------------------------------------------------------------------


class TestExtract:
    def test_returns_feature_vector(self) -> None:
        eng = FeatureEngineer()
        fv = eng.extract(_candidate(), _jd())
        assert isinstance(fv, FeatureVector)

    def test_candidate_id_preserved(self) -> None:
        eng = FeatureEngineer()
        fv = eng.extract(_candidate("CAND_0000042"), _jd())
        assert fv.candidate_id == "CAND_0000042"

    def test_all_scores_in_zero_one(self) -> None:
        eng = FeatureEngineer()
        fv = eng.extract(_candidate(), _jd())
        for field_name in (
            "experience_score",
            "education_score",
            "certification_score",
            "redrob_signal_score",
        ):
            score = getattr(fv, field_name)
            assert 0.0 <= score <= 1.0, f"{field_name} out of range: {score}"

    def test_matched_must_have_correct(self) -> None:
        eng = FeatureEngineer()
        # candidate has python + faiss + pytorch; JD must-have = python + faiss
        fv = eng.extract(
            _candidate(skills=["python", "faiss", "tensorflow"]),
            _jd(must_have=["python", "faiss"]),
        )
        assert set(fv.matched_must_have_skills) == {"python", "faiss"}

    def test_matched_nice_to_have_correct(self) -> None:
        eng = FeatureEngineer()
        fv = eng.extract(
            _candidate(skills=["python", "kubernetes"]),
            _jd(must_have=["python"], nice_to_have=["kubernetes", "lora"]),
        )
        assert "kubernetes" in fv.matched_nice_to_have_skills
        assert "lora" not in fv.matched_nice_to_have_skills

    def test_no_skill_overlap_returns_empty_matches(self) -> None:
        eng = FeatureEngineer()
        fv = eng.extract(
            _candidate(skills=["excel", "powerpoint"]),
            _jd(must_have=["python", "faiss"]),
        )
        assert fv.matched_must_have_skills == []

    def test_years_of_experience_stored(self) -> None:
        eng = FeatureEngineer()
        fv = eng.extract(_candidate(years=7.5), _jd())
        assert abs(fv.years_of_experience - 7.5) < 1e-6

    def test_cert_names_stored(self) -> None:
        eng = FeatureEngineer()
        certs = [_cert("AWS Machine Learning Specialty"), _cert("Google Cloud Professional")]
        fv = eng.extract(_candidate(certifications=certs), _jd())
        assert "AWS Machine Learning Specialty" in fv.cert_names


# ---------------------------------------------------------------------------
# Tests — experience score
# ---------------------------------------------------------------------------


class TestExperienceScore:
    def _score(self, years: float, lo: float, hi: float) -> float:
        eng = FeatureEngineer()
        jd = _jd(min_years=lo, max_years=hi)
        cand = _candidate(years=years)
        return eng._experience_score(cand, jd)

    def test_within_band_is_one(self) -> None:
        assert abs(self._score(6.0, 4.0, 8.0) - 1.0) < 1e-6

    def test_at_min_is_one(self) -> None:
        assert abs(self._score(4.0, 4.0, 8.0) - 1.0) < 1e-6

    def test_at_max_is_one(self) -> None:
        assert abs(self._score(8.0, 4.0, 8.0) - 1.0) < 1e-6

    def test_below_min_is_less_than_one(self) -> None:
        score = self._score(2.0, 4.0, 8.0)
        assert score < 1.0
        assert score >= 0.0

    def test_zero_years_zero_score(self) -> None:
        score = self._score(0.0, 4.0, 8.0)
        assert abs(score) < 1e-6

    def test_above_max_decays_but_stays_above_threshold(self) -> None:
        score = self._score(15.0, 4.0, 8.0)
        assert score >= 0.70

    def test_slightly_above_max_close_to_one(self) -> None:
        score = self._score(9.0, 4.0, 8.0)
        assert score > 0.90

    def test_no_requirement_uses_fallback(self) -> None:
        """When min_years=0 and max_years=None, a 5-year candidate should score well."""
        eng = FeatureEngineer()
        jd = _jd(min_years=0.0, max_years=99.0)  # effectively unconstrained
        cand = _candidate(years=5.0)
        score = eng._experience_score(cand, jd)
        # At years=5, within [0, 99] → score = 1.0
        assert abs(score - 1.0) < 1e-6

    def test_10_years_fallback_caps_at_one(self) -> None:
        """A candidate with 10+ years within a wide band still scores 1.0."""
        eng = FeatureEngineer()
        jd = _jd(min_years=0.0, max_years=50.0)
        cand = _candidate(years=12.0)
        assert abs(eng._experience_score(cand, jd) - 1.0) < 1e-6

    def test_output_always_in_zero_one(self) -> None:
        for years in [0.0, 1.0, 4.0, 5.0, 8.0, 12.0, 20.0, 50.0]:
            score = self._score(years, 5.0, 8.0)
            assert 0.0 <= score <= 1.0, f"Score out of range for {years} years: {score}"


# ---------------------------------------------------------------------------
# Tests — education score
# ---------------------------------------------------------------------------


class TestEducationScore:
    def test_tier1_phd_near_one(self) -> None:
        eng = FeatureEngineer()
        cand = _candidate(education=[_edu("PhD Computer Science", EducationTier.TIER_1)])
        score = eng._education_score(cand)
        assert score >= 0.95

    def test_tier1_bachelor_is_one(self) -> None:
        eng = FeatureEngineer()
        cand = _candidate(education=[_edu("Bachelor of Technology", EducationTier.TIER_1)])
        score = eng._education_score(cand)
        assert abs(score - 1.0) < 1e-6

    def test_tier3_bachelor_is_half(self) -> None:
        eng = FeatureEngineer()
        cand = _candidate(education=[_edu("Bachelor", EducationTier.TIER_3)])
        score = eng._education_score(cand)
        assert abs(score - 0.50) < 1e-6

    def test_unknown_tier_small_score(self) -> None:
        eng = FeatureEngineer()
        cand = _candidate(education=[_edu("Bachelor", EducationTier.UNKNOWN)])
        score = eng._education_score(cand)
        assert score < 0.30

    def test_empty_education_baseline(self) -> None:
        """_education_score returns 0.10 when education list is empty."""
        eng = FeatureEngineer()
        # Call _education_score directly with a candidate whose education is empty
        cand = _candidate(education=[])
        score = eng._education_score(cand)
        assert abs(score - 0.10) < 1e-6

    def test_multiple_degrees_takes_best(self) -> None:
        eng = FeatureEngineer()
        cand = _candidate(
            education=[
                _edu("Bachelor", EducationTier.TIER_4),
                _edu("Master of Science", EducationTier.TIER_1),
            ]
        )
        score = eng._education_score(cand)
        # Tier1 master = 1.0 + 0.10 → clamped 1.0
        assert abs(score - 1.0) < 1e-6

    def test_score_in_zero_one(self) -> None:
        eng = FeatureEngineer()
        for tier in EducationTier:
            for degree in ["PhD", "Master", "Bachelor", "Diploma"]:
                cand = _candidate(education=[_edu(degree, tier)])
                score = eng._education_score(cand)
                assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Tests — certification score
# ---------------------------------------------------------------------------


class TestCertificationScore:
    def test_no_certs_is_zero(self) -> None:
        eng = FeatureEngineer()
        fv = eng.extract(_candidate(certifications=[]), _jd())
        assert fv.certification_score == 0.0

    def test_one_relevant_cert_scores_point_three(self) -> None:
        eng = FeatureEngineer()
        certs = [_cert("AWS Machine Learning Specialty")]
        fv = eng.extract(_candidate(certifications=certs), _jd())
        assert abs(fv.certification_score - 0.30) < 1e-6

    def test_one_irrelevant_cert_scores_point_one(self) -> None:
        eng = FeatureEngineer()
        certs = [_cert("IELTS Language Test")]
        fv = eng.extract(_candidate(certifications=certs), _jd())
        assert abs(fv.certification_score - 0.10) < 1e-6

    def test_four_relevant_certs_capped_at_one(self) -> None:
        eng = FeatureEngineer()
        certs = [
            _cert("Deep Learning Specialization"),
            _cert("TensorFlow Developer Certificate"),
            _cert("AWS Machine Learning Specialty"),
            _cert("NLP with Transformers Course"),
        ]
        fv = eng.extract(_candidate(certifications=certs), _jd())
        assert abs(fv.certification_score - 1.0) < 1e-6

    def test_jd_technology_match_relevant(self) -> None:
        eng = FeatureEngineer()
        # "faiss" is in key_technologies; cert name contains faiss → relevant
        certs = [_cert("Advanced FAISS and Vector Search")]
        fv = eng.extract(_candidate(certifications=certs), _jd(technologies=["faiss"]))
        assert fv.certification_score >= 0.30


# ---------------------------------------------------------------------------
# Tests — Redrob signal score
# ---------------------------------------------------------------------------


class TestRedrobScore:
    def test_open_to_work_true_adds_signal(self) -> None:
        eng = FeatureEngineer()
        open_sig = _signals(open_to_work_flag=True)
        closed_sig = _signals(open_to_work_flag=False)
        score_open, *_ = eng._redrob_score(_candidate(signals=open_sig))
        score_closed, *_ = eng._redrob_score(_candidate(signals=closed_sig))
        assert score_open > score_closed

    def test_high_response_rate_higher_score(self) -> None:
        eng = FeatureEngineer()
        high = _signals(recruiter_response_rate=1.0, avg_response_time_hours=1.0)
        low = _signals(recruiter_response_rate=0.0)
        score_high, *_ = eng._redrob_score(_candidate(signals=high))
        score_low, *_ = eng._redrob_score(_candidate(signals=low))
        assert score_high > score_low

    def test_github_sentinel_treated_as_zero(self) -> None:
        eng = FeatureEngineer()
        sig = _signals(github_activity_score=-1.0)
        _, _, _, _, _, _, github, _, _ = eng._redrob_score(_candidate(signals=sig))
        assert github == 0.0

    def test_github_100_maps_to_one(self) -> None:
        eng = FeatureEngineer()
        sig = _signals(
            github_activity_score=100.0,
            interview_completion_rate=0.5,
            profile_completeness_score=80.0,
        )
        _, _, _, _, _, _, github, _, _ = eng._redrob_score(_candidate(signals=sig))
        assert abs(github - 1.0) < 1e-6

    def test_empty_assessments_returns_zero_avg(self) -> None:
        eng = FeatureEngineer()
        sig = _signals(skill_assessment_scores={})
        _, _, _, _, _, _, _, assessment_avg, _ = eng._redrob_score(_candidate(signals=sig))
        assert assessment_avg == 0.0

    def test_saved_by_10_caps_at_one(self) -> None:
        eng = FeatureEngineer()
        sig = _signals(saved_by_recruiters_30d=10)
        _, _, _, _, _, _, _, _, saved = eng._redrob_score(_candidate(signals=sig))
        assert abs(saved - 1.0) < 1e-6

    def test_inactive_200_days_low_recency(self) -> None:
        eng = FeatureEngineer()
        sig = _signals(last_active_date=LONG_AGO)
        _, _, _, _, _, recency, _, _, _ = eng._redrob_score(_candidate(signals=sig))
        assert recency == 0.0  # beyond 90-day window → clipped to 0

    def test_active_yesterday_high_recency(self) -> None:
        eng = FeatureEngineer()
        sig = _signals(last_active_date=YESTERDAY)
        _, _, _, _, _, recency, _, _, _ = eng._redrob_score(_candidate(signals=sig))
        assert recency >= 0.98

    def test_composite_in_zero_one(self) -> None:
        eng = FeatureEngineer()
        for _ in range(5):
            fv = eng.extract(_candidate(), _jd())
            assert 0.0 <= fv.redrob_signal_score <= 1.0


# ---------------------------------------------------------------------------
# Tests — honeypot detection
# ---------------------------------------------------------------------------


class TestHoneypotDetection:
    def test_normal_candidate_not_honeypot(self) -> None:
        eng = FeatureEngineer()
        fv = eng.extract(_candidate(), _jd())
        assert fv.is_honeypot is False

    def test_rule1_instant_universal_responder(self) -> None:
        eng = FeatureEngineer()
        sig = _signals(recruiter_response_rate=1.0, avg_response_time_hours=0.05)
        fv = eng.extract(_candidate(signals=sig), _jd())
        assert fv.is_honeypot is True

    def test_rule2_all_perfect_signals(self) -> None:
        eng = FeatureEngineer()
        sig = _signals(
            github_activity_score=100.0,
            profile_completeness_score=100.0,
            interview_completion_rate=1.0,
        )
        fv = eng.extract(_candidate(signals=sig), _jd())
        assert fv.is_honeypot is True

    def test_rule3_offer_acceptance_sentinel(self) -> None:
        eng = FeatureEngineer()
        sig = _signals(offer_acceptance_rate=-1.0)
        fv = eng.extract(_candidate(signals=sig), _jd())
        assert fv.is_honeypot is True

    def test_rule4_zero_notice_not_open(self) -> None:
        eng = FeatureEngineer()
        sig = _signals(notice_period_days=0, open_to_work_flag=False)
        fv = eng.extract(_candidate(signals=sig), _jd())
        assert fv.is_honeypot is True

    def test_rule1_partial_not_honeypot(self) -> None:
        """High response rate but slow response time — not a honeypot."""
        eng = FeatureEngineer()
        sig = _signals(recruiter_response_rate=1.0, avg_response_time_hours=2.0)
        fv = eng.extract(_candidate(signals=sig), _jd())
        assert fv.is_honeypot is False

    def test_rule2_partial_not_honeypot(self) -> None:
        """Perfect github and profile completeness but imperfect interview rate."""
        eng = FeatureEngineer()
        sig = _signals(
            github_activity_score=100.0,
            profile_completeness_score=100.0,
            interview_completion_rate=0.80,
        )
        fv = eng.extract(_candidate(signals=sig), _jd())
        assert fv.is_honeypot is False

    def test_zero_notice_open_to_work_not_honeypot(self) -> None:
        """Zero notice period is fine if open_to_work=True."""
        eng = FeatureEngineer()
        sig = _signals(notice_period_days=0, open_to_work_flag=True)
        fv = eng.extract(_candidate(signals=sig), _jd())
        assert fv.is_honeypot is False
