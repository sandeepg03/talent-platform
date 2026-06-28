"""
Unit tests for src.parsers.jd_parser

Covers:
  - JDParser.from_canonical() — structure & field integrity
  - JDParser.from_raw_text() — custom text accepted
  - JDParser.save() / JDParser.load() — round-trip serialisation
  - StructuredJD.build_embedding_text() — content checks
  - Validation: empty text raises, bad file extension raises, missing file raises
  - Skill lists: must-have ⊄ nice-to-have, all priorities correct
  - Experience band: 5-9 years, SENIOR level
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.parsers.jd_parser import (
    JDParser,
    _DISQUALIFYING_PATTERNS,
    _KEY_TECHNOLOGIES,
    _MUST_HAVE_SKILLS,
    _NICE_TO_HAVE_SKILLS,
)
from src.schemas.jd import (
    ExperienceLevel,
    RequirementPriority,
    StructuredJD,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def canonical_jd() -> StructuredJD:
    """Parse from the embedded canonical text — no file I/O."""
    return JDParser.from_canonical()


# ---------------------------------------------------------------------------
# Tests — from_canonical() structure
# ---------------------------------------------------------------------------


class TestFromCanonical:
    def test_returns_structured_jd(self, canonical_jd: StructuredJD) -> None:
        assert isinstance(canonical_jd, StructuredJD)

    def test_title_correct(self, canonical_jd: StructuredJD) -> None:
        assert canonical_jd.title == "Senior AI Engineer"

    def test_company_correct(self, canonical_jd: StructuredJD) -> None:
        assert canonical_jd.company == "Redrob AI"

    def test_must_have_skills_populated(self, canonical_jd: StructuredJD) -> None:
        assert len(canonical_jd.must_have_skills) == len(_MUST_HAVE_SKILLS)

    def test_nice_to_have_skills_populated(self, canonical_jd: StructuredJD) -> None:
        assert len(canonical_jd.nice_to_have_skills) == len(_NICE_TO_HAVE_SKILLS)

    def test_disqualifying_patterns_populated(self, canonical_jd: StructuredJD) -> None:
        assert len(canonical_jd.disqualifying_patterns) == len(_DISQUALIFYING_PATTERNS)

    def test_key_technologies_populated(self, canonical_jd: StructuredJD) -> None:
        assert len(canonical_jd.key_technologies) == len(_KEY_TECHNOLOGIES)

    def test_experience_min_years(self, canonical_jd: StructuredJD) -> None:
        assert canonical_jd.experience.min_years == 5.0

    def test_experience_max_years(self, canonical_jd: StructuredJD) -> None:
        assert canonical_jd.experience.max_years == 9.0

    def test_experience_level_is_senior(self, canonical_jd: StructuredJD) -> None:
        assert canonical_jd.experience.preferred_level == ExperienceLevel.SENIOR

    def test_location_includes_india_cities(self, canonical_jd: StructuredJD) -> None:
        cities = canonical_jd.location.cities
        assert "Pune" in cities
        assert "Noida" in cities

    def test_relocation_open(self, canonical_jd: StructuredJD) -> None:
        assert canonical_jd.location.relocation_open is True

    def test_raw_text_nonempty(self, canonical_jd: StructuredJD) -> None:
        assert len(canonical_jd.raw_text) > 200

    def test_embedding_text_populated(self, canonical_jd: StructuredJD) -> None:
        assert len(canonical_jd.embedding_text) > 100


# ---------------------------------------------------------------------------
# Tests — Skill priority correctness
# ---------------------------------------------------------------------------


class TestSkillPriorities:
    def test_all_must_have_have_correct_priority(
        self, canonical_jd: StructuredJD
    ) -> None:
        for skill in canonical_jd.must_have_skills:
            assert skill.priority == RequirementPriority.MUST_HAVE, (
                f"{skill.name!r} has wrong priority: {skill.priority}"
            )

    def test_all_nice_to_have_have_correct_priority(
        self, canonical_jd: StructuredJD
    ) -> None:
        for skill in canonical_jd.nice_to_have_skills:
            assert skill.priority == RequirementPriority.NICE_TO_HAVE, (
                f"{skill.name!r} has wrong priority: {skill.priority}"
            )

    def test_python_is_must_have(self, canonical_jd: StructuredJD) -> None:
        must_names = canonical_jd.all_required_skill_names
        assert "python" in must_names

    def test_faiss_is_must_have(self, canonical_jd: StructuredJD) -> None:
        must_names = canonical_jd.all_required_skill_names
        assert "faiss" in must_names

    def test_lora_is_nice_to_have(self, canonical_jd: StructuredJD) -> None:
        pref_names = canonical_jd.all_preferred_skill_names
        assert "lora" in pref_names

    def test_no_overlap_between_must_and_nice(
        self, canonical_jd: StructuredJD
    ) -> None:
        must = set(canonical_jd.all_required_skill_names)
        nice = set(canonical_jd.all_preferred_skill_names)
        overlap = must & nice
        assert not overlap, f"Skills in both lists: {overlap}"

    def test_all_must_have_have_nonempty_context(
        self, canonical_jd: StructuredJD
    ) -> None:
        for skill in canonical_jd.must_have_skills:
            assert skill.context.strip(), (
                f"Must-have skill {skill.name!r} has empty context"
            )


# ---------------------------------------------------------------------------
# Tests — build_embedding_text content
# ---------------------------------------------------------------------------


class TestEmbeddingText:
    def test_embedding_text_contains_title(
        self, canonical_jd: StructuredJD
    ) -> None:
        assert "Senior AI Engineer" in canonical_jd.embedding_text

    def test_embedding_text_contains_required_skills(
        self, canonical_jd: StructuredJD
    ) -> None:
        assert "Python" in canonical_jd.embedding_text
        assert "FAISS" in canonical_jd.embedding_text

    def test_embedding_text_contains_preferred_skills(
        self, canonical_jd: StructuredJD
    ) -> None:
        assert "LoRA" in canonical_jd.embedding_text

    def test_embedding_text_contains_experience_range(
        self, canonical_jd: StructuredJD
    ) -> None:
        assert "5.0" in canonical_jd.embedding_text or "5-9" in canonical_jd.embedding_text

    def test_build_embedding_text_is_deterministic(
        self, canonical_jd: StructuredJD
    ) -> None:
        t1 = canonical_jd.build_embedding_text()
        t2 = canonical_jd.build_embedding_text()
        assert t1 == t2


# ---------------------------------------------------------------------------
# Tests — from_raw_text()
# ---------------------------------------------------------------------------


class TestFromRawText:
    def test_accepts_custom_text(self) -> None:
        jd = JDParser.from_raw_text("Senior Engineer at Acme. Needs Python and FAISS.")
        assert isinstance(jd, StructuredJD)

    def test_raw_text_stored(self) -> None:
        custom = "Looking for ML Engineer with 5+ years experience in NLP."
        jd = JDParser.from_raw_text(custom)
        assert jd.raw_text == custom

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            JDParser.from_raw_text("   ")

    def test_must_have_skills_from_canonical_included(self) -> None:
        jd = JDParser.from_raw_text("Any raw text describing the position.")
        # Expert-structured requirements are ALWAYS included regardless of text
        assert len(jd.must_have_skills) == len(_MUST_HAVE_SKILLS)


# ---------------------------------------------------------------------------
# Tests — save() / load() round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_creates_json_file(
        self, canonical_jd: StructuredJD, tmp_path: Path
    ) -> None:
        output = tmp_path / "jd_cache.json"
        JDParser.save(canonical_jd, output)
        assert output.exists()

    def test_saved_file_is_valid_json(
        self, canonical_jd: StructuredJD, tmp_path: Path
    ) -> None:
        output = tmp_path / "jd_cache.json"
        JDParser.save(canonical_jd, output)
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert "title" in payload
        assert "must_have_skills" in payload

    def test_load_returns_structured_jd(
        self, canonical_jd: StructuredJD, tmp_path: Path
    ) -> None:
        output = tmp_path / "jd_cache.json"
        JDParser.save(canonical_jd, output)
        restored = JDParser.load(output)
        assert isinstance(restored, StructuredJD)

    def test_load_preserves_title(
        self, canonical_jd: StructuredJD, tmp_path: Path
    ) -> None:
        output = tmp_path / "jd_cache.json"
        JDParser.save(canonical_jd, output)
        restored = JDParser.load(output)
        assert restored.title == canonical_jd.title

    def test_load_preserves_must_have_count(
        self, canonical_jd: StructuredJD, tmp_path: Path
    ) -> None:
        output = tmp_path / "jd_cache.json"
        JDParser.save(canonical_jd, output)
        restored = JDParser.load(output)
        assert len(restored.must_have_skills) == len(canonical_jd.must_have_skills)

    def test_load_preserves_experience_band(
        self, canonical_jd: StructuredJD, tmp_path: Path
    ) -> None:
        output = tmp_path / "jd_cache.json"
        JDParser.save(canonical_jd, output)
        restored = JDParser.load(output)
        assert restored.experience.min_years == canonical_jd.experience.min_years
        assert restored.experience.max_years == canonical_jd.experience.max_years

    def test_load_preserves_embedding_text(
        self, canonical_jd: StructuredJD, tmp_path: Path
    ) -> None:
        output = tmp_path / "jd_cache.json"
        JDParser.save(canonical_jd, output)
        restored = JDParser.load(output)
        assert restored.embedding_text == canonical_jd.embedding_text

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            JDParser.load(tmp_path / "nonexistent.json")

    def test_save_creates_parent_dirs(
        self, canonical_jd: StructuredJD, tmp_path: Path
    ) -> None:
        output = tmp_path / "subdir" / "deep" / "jd.json"
        JDParser.save(canonical_jd, output)
        assert output.exists()


# ---------------------------------------------------------------------------
# Tests — from_docx() error handling
# ---------------------------------------------------------------------------


class TestFromDocxErrors:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            JDParser.from_docx(tmp_path / "missing.docx")

    def test_wrong_extension_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "jd.txt"
        p.write_text("Some JD text")
        with pytest.raises(ValueError, match=".txt"):
            JDParser.from_docx(p)
