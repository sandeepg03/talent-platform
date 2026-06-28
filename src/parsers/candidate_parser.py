"""
Candidate Parser — loads and validates candidates from candidates.jsonl.

Responsibilities:
  - Stream-parse the 100K JSONL file without loading it all into RAM
  - Validate every record against CandidateProfile (Pydantic v2)
  - Build a rich plain-text representation of each candidate for embedding
  - Log and skip malformed records rather than crashing the pipeline
  - Expose an iterator, a batch loader, and a single-record loader

Design constraints:
  - Memory-efficient: processes one record at a time when iterating
  - No O(N²) operations
  - Fully typed — no bare dict[str, Any] in public interfaces
  - Honeypot-safe: does NOT filter honeypots here (that is Feature Engineer's job)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator, Iterator

from loguru import logger
from pydantic import ValidationError

from src.schemas.candidate import CandidateProfile, SkillProficiency

# ---------------------------------------------------------------------------
# Text builder
# ---------------------------------------------------------------------------


class CandidateTextBuilder:
    """
    Constructs a semantically rich plain-text representation of a candidate.

    This text is the sole input to the embedding engine — it is deliberately
    designed to surface the signals an expert recruiter would look for:
      1. Current role and headline (most recent, highest weight)
      2. Professional summary
      3. Career descriptions (ordered newest-first)
      4. Skills with proficiency level (expert/advanced up-weighted via repetition)
      5. Education fields
      6. Certification names
      7. GitHub activity note (only if score > 0)

    The format is intentionally verbose for dense retrieval: sentence
    transformers compress the text into a fixed-size vector, so adding
    context rarely hurts and often helps.
    """

    # Proficiency-level repetition multipliers: expert skills are mentioned
    # multiple times so the embedding has higher weight on them.
    _PROFICIENCY_REPEAT: dict[SkillProficiency, int] = {
        SkillProficiency.EXPERT: 3,
        SkillProficiency.ADVANCED: 2,
        SkillProficiency.INTERMEDIATE: 1,
        SkillProficiency.BEGINNER: 1,
    }

    @classmethod
    def build(cls, candidate: CandidateProfile) -> str:
        """
        Return the canonical embedding text for a single candidate.

        The text is deterministic for a given candidate — calling this
        method twice on the same object produces identical output.
        """
        sections: list[str] = []

        # --- Section 1: Identity & headline ---
        profile = candidate.profile
        sections.append(
            f"Professional Profile: {profile.headline}. "
            f"Current role: {profile.current_title} at {profile.current_company} "
            f"({profile.current_industry}). "
            f"Total experience: {profile.years_of_experience} years."
        )

        # --- Section 2: Summary ---
        if profile.summary.strip():
            sections.append(f"Summary: {profile.summary.strip()}")

        # --- Section 3: Career history (newest-first, up to 5 entries) ---
        career_sorted = sorted(
            candidate.career_history,
            key=lambda e: (e.is_current, e.start_date),
            reverse=True,
        )
        career_parts: list[str] = []
        for entry in career_sorted[:5]:
            duration_years = round(entry.duration_months / 12, 1)
            career_parts.append(
                f"[{entry.title} at {entry.company}, {duration_years} yrs] {entry.description}"
            )
        if career_parts:
            sections.append("Work experience: " + " | ".join(career_parts))

        # --- Section 4: Skills (expert/advanced repeated for embedding weight) ---
        skill_tokens: list[str] = []
        for skill in candidate.skills:
            repeat = cls._PROFICIENCY_REPEAT.get(skill.proficiency, 1)
            skill_tokens.extend([skill.name] * repeat)
        if skill_tokens:
            sections.append(f"Skills: {', '.join(skill_tokens)}")

        # --- Section 5: Education ---
        edu_parts: list[str] = []
        for edu in candidate.education:
            edu_parts.append(f"{edu.degree} in {edu.field_of_study} from {edu.institution}")
        if edu_parts:
            sections.append(f"Education: {'; '.join(edu_parts)}")

        # --- Section 6: Certifications ---
        if candidate.certifications:
            cert_names = [c.name for c in candidate.certifications]
            sections.append(f"Certifications: {', '.join(cert_names)}")

        # --- Section 7: GitHub activity note ---
        gh_score = candidate.redrob_signals.github_activity_score
        if gh_score > 0:
            level = "high" if gh_score >= 60 else ("moderate" if gh_score >= 30 else "low")
            sections.append(f"Open source / GitHub activity: {level} ({gh_score:.0f}/100)")

        return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class CandidateParser:
    """
    Streams and validates candidate records from a JSONL file.

    Usage patterns:

        # 1. Iterate one by one (memory-efficient)
        parser = CandidateParser(path)
        for candidate in parser.iter_candidates():
            process(candidate)

        # 2. Load all into memory (for small datasets / tests)
        candidates = parser.load_all()

        # 3. Load in batches (for embedding engine)
        for batch in parser.iter_batches(batch_size=512):
            embed(batch)

        # 4. Build the id→text mapping (for FAISS index building)
        texts = parser.build_text_corpus()

        # 5. Load a single candidate by ID
        candidate = parser.load_by_id("CAND_0000001")
    """

    def __init__(self, jsonl_path: Path | str) -> None:
        self._path = Path(jsonl_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Candidates file not found: {self._path}")
        if self._path.suffix.lower() != ".jsonl":
            raise ValueError(f"Expected a .jsonl file, got: {self._path.suffix!r}")
        self._text_builder = CandidateTextBuilder()
        logger.info(f"CandidateParser initialized with: {self._path}")

    # ------------------------------------------------------------------
    # Public iterators
    # ------------------------------------------------------------------

    def iter_candidates(self) -> Iterator[CandidateProfile]:
        """
        Yield validated CandidateProfile objects one at a time.

        Malformed or invalid records are logged at WARNING level and skipped.
        The iterator does NOT load the entire file into memory.
        """
        valid_count = 0
        error_count = 0

        with open(self._path, "r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw: dict = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(f"Line {line_number}: JSON parse error — {exc}. Skipping.")
                    error_count += 1
                    continue
                try:
                    candidate = CandidateProfile.model_validate(raw)
                    valid_count += 1
                    yield candidate
                except ValidationError as exc:
                    cid = raw.get("candidate_id", "<unknown>")
                    logger.warning(
                        f"Line {line_number} (id={cid!r}): "
                        f"Validation error — {exc.error_count()} issue(s). Skipping."
                    )
                    error_count += 1

        logger.info(f"Parsing complete: {valid_count} valid, {error_count} skipped.")

    def iter_batches(self, batch_size: int = 512) -> Generator[list[CandidateProfile], None, None]:
        """
        Yield lists of up to ``batch_size`` validated CandidateProfile objects.

        Designed for the embedding engine's batch inference loop — batching
        avoids the per-sample overhead of calling the model N times.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        batch: list[CandidateProfile] = []
        for candidate in self.iter_candidates():
            batch.append(candidate)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    # ------------------------------------------------------------------
    # Convenience loaders
    # ------------------------------------------------------------------

    def load_all(self) -> list[CandidateProfile]:
        """
        Load the entire JSONL file into memory as a list.

        Use only for small datasets or tests — at 100K candidates this
        consumes ~2-4 GB depending on record size.
        """
        logger.warning(
            "load_all() called — this loads all candidates into memory. "
            "Use iter_batches() for production workloads."
        )
        return list(self.iter_candidates())

    def load_by_id(self, candidate_id: str) -> CandidateProfile | None:
        """
        Scan the file and return the first record matching ``candidate_id``.

        Linear scan O(N) — only for debugging / single-record lookups.
        For bulk lookups, build an in-memory id→candidate dict from iter_candidates().
        """
        for candidate in self.iter_candidates():
            if candidate.candidate_id == candidate_id:
                return candidate
        logger.warning(f"Candidate {candidate_id!r} not found in {self._path}")
        return None

    # ------------------------------------------------------------------
    # Text corpus builder
    # ------------------------------------------------------------------

    def build_text_corpus(
        self,
        *,
        batch_size: int = 512,
        progress: bool = False,
    ) -> tuple[list[str], list[str]]:
        """
        Stream through the JSONL and build two parallel lists:
          - ``candidate_ids``: the CAND_XXXXXXX id strings
          - ``texts``: the corresponding embedding texts

        Both lists share the same index so that
        ``candidate_ids[i]`` is the owner of ``texts[i]``.

        Args:
            batch_size: Number of records to process per log interval.
            progress:   If True, log progress every ``batch_size`` records.

        Returns:
            Tuple of (candidate_ids, texts) — equal-length lists.
        """
        candidate_ids: list[str] = []
        texts: list[str] = []
        processed = 0

        for candidate in self.iter_candidates():
            text = CandidateTextBuilder.build(candidate)
            candidate_ids.append(candidate.candidate_id)
            texts.append(text)
            processed += 1
            if progress and processed % batch_size == 0:
                logger.info(f"  Built text corpus: {processed:,} records processed")

        logger.info(
            f"Text corpus complete: {len(candidate_ids):,} candidates, "
            f"avg text length {sum(len(t) for t in texts) // max(len(texts), 1)} chars"
        )
        return candidate_ids, texts

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def count_records(self) -> int:
        """Count total lines (valid + invalid) in the JSONL file."""
        count = 0
        with open(self._path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    count += 1
        return count
