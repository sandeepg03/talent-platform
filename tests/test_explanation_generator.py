"""
Unit tests for src.explanation.generator

All tests are fully synthetic — no model downloads.

Covers:
  - ExplanationGenerator.generate(): returns non-empty string
  - Output length cap at 500 chars
  - Contains expected evidence tokens (score, rank, skills, etc.)
  - Handles edge cases: no skills, no certs, zero scores
  - generate_honeypot(): returns meaningful string with candidate_id
"""

from __future__ import annotations


from src.explanation.generator import ExplanationGenerator
from src.schemas.scoring import FeatureVector, HybridScore


def _hs(
    cid: str = "CAND_0000001",
    sem: float = 0.82,
    ce: float = 0.75,
    exp: float = 0.80,
    sig: float = 0.65,
    edu: float = 0.75,
    cert: float = 0.30,
    honeypot: bool = False,
) -> HybridScore:
    return HybridScore.compute(
        candidate_id=cid,
        semantic_similarity=sem,
        cross_encoder_score=ce,
        experience_score=exp,
        redrob_signal_score=sig,
        education_score=edu,
        certification_score=cert,
        is_honeypot=honeypot,
    )


def _fv(
    cid: str = "CAND_0000001",
    must: list[str] | None = None,
    nice: list[str] | None = None,
    certs: list[str] | None = None,
    exp_score: float = 0.80,
    edu_score: float = 0.75,
    sig_score: float = 0.65,
    yoe: float = 6.0,
    degree: str = "Bachelor of Technology",
    open_to_work: float = 1.0,
    response_rate: float = 0.85,
) -> FeatureVector:
    return FeatureVector(
        candidate_id=cid,
        experience_score=exp_score,
        education_score=edu_score,
        certification_score=0.30,
        redrob_signal_score=sig_score,
        signal_open_to_work=open_to_work,
        signal_response_rate=response_rate,
        signal_interview_completion=0.90,
        signal_profile_completeness=0.80,
        signal_recency=0.95,
        signal_github=0.55,
        signal_assessment_avg=0.65,
        signal_saved_by_recruiters=0.30,
        is_honeypot=False,
        years_of_experience=yoe,
        highest_education_degree=degree,
        matched_must_have_skills=must or ["python", "faiss", "pytorch"],
        matched_nice_to_have_skills=nice or ["kubernetes"],
        cert_names=certs or ["AWS Machine Learning Specialty"],
    )


class TestGenerate:
    def test_returns_string(self) -> None:
        gen = ExplanationGenerator()
        result = gen.generate(_hs(), _fv(), rank=1)
        assert isinstance(result, str)

    def test_non_empty(self) -> None:
        gen = ExplanationGenerator()
        result = gen.generate(_hs(), _fv(), rank=1)
        assert len(result.strip()) > 0

    def test_within_500_chars(self) -> None:
        gen = ExplanationGenerator()
        result = gen.generate(_hs(), _fv(), rank=1)
        assert len(result) <= 500

    def test_contains_rank(self) -> None:
        gen = ExplanationGenerator()
        result = gen.generate(_hs(), _fv(), rank=7)
        assert "#7" in result

    def test_contains_score(self) -> None:
        gen = ExplanationGenerator()
        hs = _hs(sem=0.80, ce=0.70, exp=0.80, sig=0.60, edu=0.70, cert=0.20)
        result = gen.generate(hs, _fv(), rank=1)
        # final_score should appear in output
        assert str(round(hs.final_score, 1)) in result

    def test_contains_matched_skills(self) -> None:
        gen = ExplanationGenerator()
        result = gen.generate(_hs(), _fv(must=["python", "faiss"]), rank=1)
        assert "python" in result or "faiss" in result

    def test_contains_years_of_experience(self) -> None:
        gen = ExplanationGenerator()
        result = gen.generate(_hs(), _fv(yoe=8.0), rank=1)
        assert "8" in result

    def test_contains_degree(self) -> None:
        gen = ExplanationGenerator()
        result = gen.generate(_hs(), _fv(degree="Master of Technology"), rank=1)
        assert "Master of Technology" in result

    def test_contains_cert_name(self) -> None:
        gen = ExplanationGenerator()
        result = gen.generate(_hs(), _fv(certs=["Google ML Certificate"]), rank=1)
        assert "Google ML Certificate" in result

    def test_no_skills_still_returns_string(self) -> None:
        gen = ExplanationGenerator()
        result = gen.generate(_hs(), _fv(must=[], nice=[], certs=[]), rank=5)
        assert len(result) > 0

    def test_not_open_to_work_reflected(self) -> None:
        gen = ExplanationGenerator()
        result = gen.generate(_hs(), _fv(open_to_work=0.0), rank=1)
        assert "not currently open" in result

    def test_open_to_work_reflected(self) -> None:
        gen = ExplanationGenerator()
        result = gen.generate(_hs(), _fv(open_to_work=1.0), rank=1)
        assert "open to opportunities" in result

    def test_high_engagement_tier(self) -> None:
        gen = ExplanationGenerator()
        result = gen.generate(_hs(sig=0.85), _fv(sig_score=0.85), rank=1)
        assert "strong" in result

    def test_low_engagement_tier(self) -> None:
        gen = ExplanationGenerator()
        result = gen.generate(_hs(sig=0.30), _fv(sig_score=0.30), rank=1)
        assert "limited" in result

    def test_long_output_truncated_to_500(self) -> None:
        """Generating with many skills and certs should still be ≤ 500 chars."""
        gen = ExplanationGenerator()
        many_skills = [f"skill_{i}" for i in range(20)]
        many_certs = [f"Cert Number {i} Long Name Certification" for i in range(10)]
        fv = _fv(must=many_skills, nice=many_skills, certs=many_certs)
        result = gen.generate(_hs(), fv, rank=1)
        assert len(result) <= 500

    def test_different_ranks_produce_different_output(self) -> None:
        gen = ExplanationGenerator()
        r1 = gen.generate(_hs(), _fv(), rank=1)
        r5 = gen.generate(_hs(), _fv(), rank=5)
        assert r1 != r5

    def test_response_rate_in_output(self) -> None:
        gen = ExplanationGenerator()
        result = gen.generate(_hs(), _fv(response_rate=0.92), rank=1)
        assert "92%" in result


class TestGenerateHoneypot:
    def test_returns_string(self) -> None:
        gen = ExplanationGenerator()
        result = gen.generate_honeypot("CAND_0000001", rank=1)
        assert isinstance(result, str)

    def test_contains_candidate_id(self) -> None:
        gen = ExplanationGenerator()
        result = gen.generate_honeypot("CAND_0000042", rank=1)
        assert "CAND_0000042" in result

    def test_contains_excluded_language(self) -> None:
        gen = ExplanationGenerator()
        result = gen.generate_honeypot("CAND_0000001", rank=1)
        assert "excluded" in result.lower() or "synthetic" in result.lower()
