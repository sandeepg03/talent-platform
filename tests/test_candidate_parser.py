"""
Unit tests for src.parsers.candidate_parser

Covers:
  - CandidateTextBuilder: text construction quality and determinism
  - CandidateParser: iterator correctness, batch yielding, error recovery,
    load_by_id, build_text_corpus, count_records
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.parsers.candidate_parser import CandidateParser, CandidateTextBuilder
from src.schemas.candidate import CandidateProfile


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

BASE_CANDIDATE: dict = {
    "candidate_id": "CAND_0000001",
    "profile": {
        "anonymized_name": "Alice Test",
        "headline": "Senior ML Engineer | FAISS | Transformers",
        "summary": "7 years building production embedding retrieval systems.",
        "location": "Bangalore",
        "country": "India",
        "years_of_experience": 7.0,
        "current_title": "ML Engineer",
        "current_company": "Acme AI",
        "current_company_size": "1001-5000",
        "current_industry": "AI/ML",
    },
    "career_history": [
        {
            "company": "Acme AI",
            "title": "ML Engineer",
            "start_date": "2020-01-01",
            "end_date": None,
            "duration_months": 54,
            "is_current": True,
            "industry": "AI/ML",
            "company_size": "1001-5000",
            "description": "Built embedding-based retrieval using FAISS and sentence-transformers.",
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
        {"name": "Python", "proficiency": "advanced", "endorsements": 50, "duration_months": 84},
        {"name": "SQL", "proficiency": "beginner", "endorsements": 5, "duration_months": 12},
    ],
    "certifications": [
        {"name": "AWS ML Specialty", "issuer": "Amazon", "year": 2022}
    ],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": {
        "profile_completeness_score": 92.0,
        "signup_date": "2023-01-15",
        "last_active_date": "2024-06-01",
        "open_to_work_flag": True,
        "profile_views_received_30d": 45,
        "applications_submitted_30d": 3,
        "recruiter_response_rate": 0.85,
        "avg_response_time_hours": 2.5,
        "skill_assessment_scores": {"Python": 88.0},
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


def _make_candidate(overrides: dict | None = None) -> dict:
    """Deep-copy BASE_CANDIDATE and apply overrides."""
    import copy
    record = copy.deepcopy(BASE_CANDIDATE)
    if overrides:
        record.update(overrides)
    return record


def _write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


@pytest.fixture
def valid_candidate() -> CandidateProfile:
    return CandidateProfile.model_validate(BASE_CANDIDATE)


@pytest.fixture
def single_record_jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "candidates.jsonl"
    _write_jsonl([BASE_CANDIDATE], path)
    return path


@pytest.fixture
def multi_record_jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "candidates.jsonl"
    records = []
    for i in range(1, 11):
        r = _make_candidate({"candidate_id": f"CAND_{i:07d}"})
        records.append(r)
    _write_jsonl(records, path)
    return path


# ---------------------------------------------------------------------------
# Tests — CandidateTextBuilder
# ---------------------------------------------------------------------------


class TestCandidateTextBuilder:
    def test_text_contains_headline(self, valid_candidate: CandidateProfile) -> None:
        text = CandidateTextBuilder.build(valid_candidate)
        assert "Senior ML Engineer" in text

    def test_text_contains_summary(self, valid_candidate: CandidateProfile) -> None:
        text = CandidateTextBuilder.build(valid_candidate)
        assert "7 years building production embedding" in text

    def test_text_contains_career_description(self, valid_candidate: CandidateProfile) -> None:
        text = CandidateTextBuilder.build(valid_candidate)
        assert "FAISS" in text
        assert "sentence-transformers" in text

    def test_expert_skill_repeated_three_times(self, valid_candidate: CandidateProfile) -> None:
        text = CandidateTextBuilder.build(valid_candidate)
        # FAISS is expert → should appear 3 times in skills section
        skills_section = [s for s in text.split("\n\n") if "Skills:" in s][0]
        assert skills_section.count("FAISS") == 3

    def test_advanced_skill_repeated_twice(self, valid_candidate: CandidateProfile) -> None:
        text = CandidateTextBuilder.build(valid_candidate)
        skills_section = [s for s in text.split("\n\n") if "Skills:" in s][0]
        assert skills_section.count("Python") == 2

    def test_beginner_skill_appears_once(self, valid_candidate: CandidateProfile) -> None:
        text = CandidateTextBuilder.build(valid_candidate)
        skills_section = [s for s in text.split("\n\n") if "Skills:" in s][0]
        assert skills_section.count("SQL") == 1

    def test_text_contains_education(self, valid_candidate: CandidateProfile) -> None:
        text = CandidateTextBuilder.build(valid_candidate)
        assert "IIT Bombay" in text
        assert "Computer Science" in text

    def test_text_contains_certification(self, valid_candidate: CandidateProfile) -> None:
        text = CandidateTextBuilder.build(valid_candidate)
        assert "AWS ML Specialty" in text

    def test_github_high_activity_noted(self, valid_candidate: CandidateProfile) -> None:
        text = CandidateTextBuilder.build(valid_candidate)
        assert "high" in text  # github_activity_score=72 → "high"

    def test_github_absent_when_score_is_negative(self) -> None:
        import copy
        rec = copy.deepcopy(BASE_CANDIDATE)
        rec["redrob_signals"]["github_activity_score"] = -1
        candidate = CandidateProfile.model_validate(rec)
        text = CandidateTextBuilder.build(candidate)
        assert "GitHub" not in text

    def test_text_is_deterministic(self, valid_candidate: CandidateProfile) -> None:
        text1 = CandidateTextBuilder.build(valid_candidate)
        text2 = CandidateTextBuilder.build(valid_candidate)
        assert text1 == text2

    def test_text_is_nonempty_string(self, valid_candidate: CandidateProfile) -> None:
        text = CandidateTextBuilder.build(valid_candidate)
        assert isinstance(text, str)
        assert len(text) > 100


# ---------------------------------------------------------------------------
# Tests — CandidateParser initialisation
# ---------------------------------------------------------------------------


class TestCandidateParserInit:
    def test_raises_if_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            CandidateParser(tmp_path / "missing.jsonl")

    def test_raises_if_wrong_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "candidates.json"
        p.write_text("{}")
        with pytest.raises(ValueError, match=".json"):
            CandidateParser(p)

    def test_accepts_path_and_string(self, single_record_jsonl: Path) -> None:
        p1 = CandidateParser(single_record_jsonl)
        p2 = CandidateParser(str(single_record_jsonl))
        assert p1._path == p2._path


# ---------------------------------------------------------------------------
# Tests — iter_candidates
# ---------------------------------------------------------------------------


class TestIterCandidates:
    def test_yields_candidate_profile_objects(self, single_record_jsonl: Path) -> None:
        parser = CandidateParser(single_record_jsonl)
        results = list(parser.iter_candidates())
        assert len(results) == 1
        assert isinstance(results[0], CandidateProfile)

    def test_yields_all_valid_records(self, multi_record_jsonl: Path) -> None:
        parser = CandidateParser(multi_record_jsonl)
        results = list(parser.iter_candidates())
        assert len(results) == 10

    def test_skips_malformed_json(self, tmp_path: Path) -> None:
        path = tmp_path / "candidates.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(BASE_CANDIDATE) + "\n")
            f.write("NOT_VALID_JSON\n")
            r2 = _make_candidate({"candidate_id": "CAND_0000002"})
            f.write(json.dumps(r2) + "\n")
        parser = CandidateParser(path)
        results = list(parser.iter_candidates())
        assert len(results) == 2

    def test_skips_validation_errors(self, tmp_path: Path) -> None:
        path = tmp_path / "candidates.jsonl"
        bad = _make_candidate({"candidate_id": "BAD_ID"})
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(BASE_CANDIDATE) + "\n")
            f.write(json.dumps(bad) + "\n")
        parser = CandidateParser(path)
        results = list(parser.iter_candidates())
        assert len(results) == 1
        assert results[0].candidate_id == "CAND_0000001"

    def test_empty_lines_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "candidates.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n")
            f.write(json.dumps(BASE_CANDIDATE) + "\n")
            f.write("   \n")
        parser = CandidateParser(path)
        results = list(parser.iter_candidates())
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Tests — iter_batches
# ---------------------------------------------------------------------------


class TestIterBatches:
    def test_batch_size_respected(self, multi_record_jsonl: Path) -> None:
        parser = CandidateParser(multi_record_jsonl)
        batches = list(parser.iter_batches(batch_size=3))
        # 10 records: batches of 3, 3, 3, 1
        assert len(batches) == 4
        assert len(batches[0]) == 3
        assert len(batches[-1]) == 1

    def test_total_records_preserved(self, multi_record_jsonl: Path) -> None:
        parser = CandidateParser(multi_record_jsonl)
        total = sum(len(b) for b in parser.iter_batches(batch_size=4))
        assert total == 10

    def test_batch_size_one(self, multi_record_jsonl: Path) -> None:
        parser = CandidateParser(multi_record_jsonl)
        batches = list(parser.iter_batches(batch_size=1))
        assert len(batches) == 10
        assert all(len(b) == 1 for b in batches)

    def test_invalid_batch_size_raises(self, single_record_jsonl: Path) -> None:
        parser = CandidateParser(single_record_jsonl)
        with pytest.raises(ValueError, match="batch_size"):
            list(parser.iter_batches(batch_size=0))

    def test_each_batch_item_is_candidate_profile(self, multi_record_jsonl: Path) -> None:
        parser = CandidateParser(multi_record_jsonl)
        for batch in parser.iter_batches(batch_size=5):
            for item in batch:
                assert isinstance(item, CandidateProfile)


# ---------------------------------------------------------------------------
# Tests — load_by_id
# ---------------------------------------------------------------------------


class TestLoadById:
    def test_returns_correct_candidate(self, multi_record_jsonl: Path) -> None:
        parser = CandidateParser(multi_record_jsonl)
        result = parser.load_by_id("CAND_0000005")
        assert result is not None
        assert result.candidate_id == "CAND_0000005"

    def test_returns_none_when_not_found(self, multi_record_jsonl: Path) -> None:
        parser = CandidateParser(multi_record_jsonl)
        result = parser.load_by_id("CAND_9999999")
        assert result is None


# ---------------------------------------------------------------------------
# Tests — build_text_corpus
# ---------------------------------------------------------------------------


class TestBuildTextCorpus:
    def test_returns_parallel_lists(self, multi_record_jsonl: Path) -> None:
        parser = CandidateParser(multi_record_jsonl)
        ids, texts = parser.build_text_corpus()
        assert len(ids) == len(texts) == 10

    def test_ids_are_strings(self, multi_record_jsonl: Path) -> None:
        parser = CandidateParser(multi_record_jsonl)
        ids, _ = parser.build_text_corpus()
        assert all(isinstance(i, str) for i in ids)

    def test_texts_are_nonempty(self, multi_record_jsonl: Path) -> None:
        parser = CandidateParser(multi_record_jsonl)
        _, texts = parser.build_text_corpus()
        assert all(len(t) > 50 for t in texts)

    def test_ids_match_candidate_ids(self, multi_record_jsonl: Path) -> None:
        parser = CandidateParser(multi_record_jsonl)
        ids, _ = parser.build_text_corpus()
        expected = {f"CAND_{i:07d}" for i in range(1, 11)}
        assert set(ids) == expected


# ---------------------------------------------------------------------------
# Tests — count_records
# ---------------------------------------------------------------------------


class TestCountRecords:
    def test_counts_all_lines(self, multi_record_jsonl: Path) -> None:
        parser = CandidateParser(multi_record_jsonl)
        assert parser.count_records() == 10

    def test_ignores_empty_lines_in_count(self, tmp_path: Path) -> None:
        path = tmp_path / "candidates.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(BASE_CANDIDATE) + "\n")
            f.write("\n")
            f.write(json.dumps(_make_candidate({"candidate_id": "CAND_0000002"})) + "\n")
        parser = CandidateParser(path)
        # count_records counts non-empty lines (including malformed)
        assert parser.count_records() == 2
