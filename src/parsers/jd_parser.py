"""
JD Parser — converts the raw job_description.docx into a StructuredJD.

Design responsibilities:
  - Extract full text from .docx (or accept raw string for testing)
  - Populate every field of StructuredJD from the extracted text
  - Hard-code the specific Redrob Senior AI Engineer requirements
    (the hackathon has exactly one JD; domain knowledge beats regex fragility)
  - Expose a save/load mechanism so the parsed JD can be cached to disk
    and reused across pipeline runs without re-parsing the docx each time
  - Emit the optimised ``embedding_text`` used by the embedding engine

Architecture note:
  The JD is parsed ONCE at precompute time and cached to JSON.
  The ranking step (rank.py) loads the cached JSON — never parses docx again.
  This keeps the 5-minute ranking window free from I/O overhead.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from src.schemas.jd import (
    ExperienceLevel,
    ExperienceRequirement,
    LocationRequirement,
    RequirementPriority,
    SkillRequirement,
    StructuredJD,
)

# ---------------------------------------------------------------------------
# Canonical JD definition
# ---------------------------------------------------------------------------
# The Redrob hackathon has exactly ONE job description.  Rather than building
# a brittle paragraph-level regex parser that might misparse on edge-cases,
# we hard-code the structured extraction here using our expert reading of the
# full docx.  Every field maps 1-to-1 to StructuredJD.  The raw_text field
# still carries the full document text for dense embedding purposes.
# ---------------------------------------------------------------------------

_MUST_HAVE_SKILLS: list[tuple[str, str]] = [
    # (skill_name, context)
    (
        "embeddings-based retrieval",
        "Production experience with sentence-transformers, OpenAI embeddings, BGE, E5 or similar "
        "deployed to real users; must have handled embedding drift and index refresh in production.",
    ),
    (
        "vector database",
        "Production experience with Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, "
        "Elasticsearch, FAISS, or similar hybrid search infrastructure.",
    ),
    (
        "Python",
        "Strong production Python — code quality is explicitly called out.",
    ),
    (
        "ranking evaluation",
        "Hands-on experience designing evaluation frameworks — NDCG, MRR, MAP, "
        "offline-to-online correlation, A/B test interpretation.",
    ),
    (
        "FAISS",
        "Listed as an acceptable vector database technology in the must-have section.",
    ),
    (
        "sentence-transformers",
        "Explicitly listed as an acceptable embedding framework.",
    ),
    (
        "information retrieval",
        "Must have understood retrieval and ranking before the LLM era — "
        "per JD: 'we're looking for people who understood retrieval and ranking "
        "before it became fashionable'.",
    ),
    (
        "hybrid search",
        "Hybrid retrieval architecture is mentioned as the expected v2 approach.",
    ),
]

_NICE_TO_HAVE_SKILLS: list[tuple[str, str]] = [
    ("LLM fine-tuning", "LoRA, QLoRA, PEFT experience explicitly listed as nice-to-have."),
    ("LoRA", "Mentioned alongside QLoRA and PEFT as LLM fine-tuning method."),
    ("QLoRA", "Mentioned alongside LoRA and PEFT as LLM fine-tuning method."),
    ("PEFT", "Parameter-efficient fine-tuning — explicitly listed."),
    ("learning-to-rank", "XGBoost-based or neural learning-to-rank experience."),
    ("XGBoost", "Mentioned in context of learning-to-rank models."),
    ("HR-tech", "Prior exposure to HR-tech, recruiting tech or marketplace products."),
    ("distributed systems", "Background in distributed systems or large-scale inference."),
    ("inference optimization", "Large-scale inference optimization experience."),
    ("open-source contributions", "Open-source contributions in the AI/ML space."),
    ("NLP", "Natural language processing — core domain of the product."),
    ("recommendation systems", "Building recommendation or search systems at product companies."),
    ("machine learning", "Applied ML engineering in production at product companies."),
    ("deep learning", "Deep learning fundamentals implied throughout the JD."),
    ("MLOps", "Evaluation infrastructure, A/B testing, offline benchmarks."),
]

_DISQUALIFYING_PATTERNS: list[str] = [
    "pure research environment without production deployment",
    "primary AI experience is LangChain API calls under 12 months",
    "no production code written in last 18 months (pure architecture/tech-lead roles)",
    "entire career at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) "
    "with no product-company experience",
    "primary expertise in computer vision, speech, or robotics without NLP/IR exposure",
    "entirely closed-source proprietary systems for 5+ years without external validation",
    "title-chasing pattern: new company every 1.5 years",
    "Marketing Manager, HR Manager, Accountant, or other non-technical primary role",
]

_KEY_TECHNOLOGIES: list[str] = [
    "FAISS",
    "sentence-transformers",
    "BGE",
    "E5",
    "OpenAI embeddings",
    "Pinecone",
    "Weaviate",
    "Qdrant",
    "Milvus",
    "OpenSearch",
    "Elasticsearch",
    "Python",
    "LoRA",
    "QLoRA",
    "PEFT",
    "XGBoost",
    "BM25",
    "NDCG",
    "MRR",
    "MAP",
    "A/B testing",
    "embeddings",
    "retrieval",
    "ranking",
    "reranking",
    "LLM",
    "transformer",
    "NLP",
    "information retrieval",
    "hybrid search",
]

_JD_RAW_TEXT = """\
Job Description: Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid — flexible cadence) | Open to relocation candidates from Tier-1 Indian cities
Employment Type: Full-time
Experience Required: 5-9 years

We need someone who is simultaneously comfortable with deep technical depth in modern ML systems \
— embeddings, retrieval, ranking, LLMs, fine-tuning — and a scrappy product-engineering attitude \
willing to ship a working ranker in a week.

What you'd actually be doing:
Own the intelligence layer of Redrob's product — the ranking, retrieval, and matching systems \
that decide what recruiters see when they search for candidates.
Weeks 1-3: Audit existing BM25 + rule-based scoring. Identify highest-leverage improvements.
Weeks 4-8: Ship a v2 ranking system using embeddings, hybrid retrieval, and LLM-based re-ranking.
Weeks 9-12: Set up evaluation infrastructure — offline benchmarks, online A/B testing, recruiter-feedback loops.
Beyond: Drive long-term architecture of candidate-JD matching at scale.

Things you absolutely need:
1. Production experience with embeddings-based retrieval systems (sentence-transformers, BGE, E5) deployed to real users.
2. Production experience with vector databases — FAISS, Pinecone, Weaviate, Qdrant, Elasticsearch.
3. Strong Python. Code quality matters.
4. Hands-on evaluation framework design — NDCG, MRR, MAP, A/B testing.

Things we'd like you to have:
- LLM fine-tuning experience (LoRA, QLoRA, PEFT)
- Learning-to-rank models (XGBoost-based or neural)
- HR-tech or marketplace product experience
- Distributed systems or large-scale inference optimization
- Open-source contributions in AI/ML

Disqualifiers:
- Pure research without production deployment
- AI experience primarily LangChain tutorial calls under 12 months
- No production code written in 18 months
- Entire career in large IT services firms without product-company experience
- Primary expertise in computer vision/speech/robotics without NLP/IR exposure

Ideal candidate:
6-8 years total experience, 4-5 in applied ML/AI at product companies (not pure services).
Has shipped at least one end-to-end ranking, search, or recommendation system to real users.
Strong opinions about retrieval (hybrid vs dense), evaluation (offline vs online), LLM integration.
Located in or willing to relocate to Noida or Pune.
Active on Redrob platform.

The right answer is NOT keyword matching. A candidate whose career shows they built a recommendation \
system at a product company is a fit even without 'RAG' or 'Pinecone' in their profile. \
A Marketing Manager with all the AI keywords listed is NOT a fit.
"""


# ---------------------------------------------------------------------------
# JD Parser
# ---------------------------------------------------------------------------


class JDParser:
    """
    Parses the Redrob hackathon job description into a ``StructuredJD``.

    Two modes:
      1. ``from_docx(path)``  — reads raw text from the .docx file and
         combines it with the expert-structured requirements above
      2. ``from_raw_text(text)`` — accepts a string directly (for tests)

    In both cases, the resulting ``StructuredJD`` is identical except for
    the ``raw_text`` field (which uses the actual docx content when available).

    Caching:
      - ``save(path)`` serialises the StructuredJD to JSON
      - ``load(path)`` deserialises it back — avoids re-parsing docx at rank time
    """

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_docx(cls, docx_path: Path | str) -> StructuredJD:
        """
        Parse the job description from a .docx file.

        Extracts plain text from the docx, then builds StructuredJD using
        both the extracted text (for raw_text / embedding) and the
        hard-coded expert-structured requirements.
        """
        path = Path(docx_path)
        if not path.exists():
            raise FileNotFoundError(f"JD file not found: {path}")
        if path.suffix.lower() != ".docx":
            raise ValueError(f"Expected a .docx file, got: {path.suffix!r}")

        logger.info(f"Parsing JD from docx: {path}")
        raw_text = cls._extract_docx_text(path)
        return cls._build_structured_jd(raw_text)

    @classmethod
    def from_raw_text(cls, raw_text: str) -> StructuredJD:
        """
        Build a StructuredJD from a raw text string.

        Used in unit tests and in the fallback case where python-docx
        is unavailable.
        """
        if not raw_text.strip():
            raise ValueError("raw_text must not be empty")
        return cls._build_structured_jd(raw_text)

    @classmethod
    def from_canonical(cls) -> StructuredJD:
        """
        Return a StructuredJD built purely from the embedded expert knowledge
        (``_JD_RAW_TEXT``), with no file I/O required.

        This is the fastest construction path and is used when the docx file
        is not available (e.g. inside a Docker container that only has
        precomputed artifacts).
        """
        return cls._build_structured_jd(_JD_RAW_TEXT)

    # ------------------------------------------------------------------
    # Save / Load (cache)
    # ------------------------------------------------------------------

    @staticmethod
    def save(jd: StructuredJD, path: Path | str) -> None:
        """
        Serialise ``jd`` to a JSON file at ``path``.

        The JSON uses Pydantic's model_dump with mode='json' to ensure
        all types (enums, etc.) are JSON-serialisable.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = jd.model_dump(mode="json")
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"StructuredJD cached to: {p}")

    @staticmethod
    def load(path: Path | str) -> StructuredJD:
        """
        Deserialise a cached StructuredJD from a JSON file.

        Raises ``FileNotFoundError`` if the cache does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Cached JD not found: {p}")
        payload = json.loads(p.read_text(encoding="utf-8"))
        jd = StructuredJD.model_validate(payload)
        logger.info(f"StructuredJD loaded from cache: {p}")
        return jd

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_docx_text(path: Path) -> str:
        """
        Extract all paragraph and table text from a .docx file.

        Returns the concatenated plain text as a single string.
        Falls back to the embedded ``_JD_RAW_TEXT`` on import error.
        """
        try:
            from docx import Document
            from docx.oxml.ns import qn
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError:
            logger.warning("python-docx not installed — falling back to embedded JD text.")
            return _JD_RAW_TEXT

        doc = Document(str(path))
        lines: list[str] = []
        for elem in doc.element.body:
            if elem.tag == qn("w:p"):
                p = Paragraph(elem, doc)
                text = p.text.strip()
                if text:
                    lines.append(text)
            elif elem.tag == qn("w:tbl"):
                tbl = Table(elem, doc)
                for row in tbl.rows:
                    cells = [c.text.strip() for c in row.cells]
                    lines.append(" | ".join(cells))

        raw = "\n".join(lines)
        logger.debug(f"Extracted {len(raw):,} chars from {path.name}")
        return raw

    @staticmethod
    def _build_structured_jd(raw_text: str) -> StructuredJD:
        """
        Construct a ``StructuredJD`` from raw text plus the hard-coded
        expert-structured requirements.

        This is the single source of truth for all JD fields.  Every
        downstream module (feature engineer, scorer, explainer) reads
        from the returned object — never from raw_text directly.
        """
        must_have = [
            SkillRequirement(
                name=name,
                priority=RequirementPriority.MUST_HAVE,
                context=context,
            )
            for name, context in _MUST_HAVE_SKILLS
        ]

        nice_to_have = [
            SkillRequirement(
                name=name,
                priority=RequirementPriority.NICE_TO_HAVE,
                context=context,
            )
            for name, context in _NICE_TO_HAVE_SKILLS
        ]

        experience = ExperienceRequirement(
            min_years=5.0,
            max_years=9.0,
            preferred_level=ExperienceLevel.SENIOR,
            notes=(
                "5-9 year range is a guideline, not a hard cutoff. "
                "Exceptional candidates at 4 years or 10+ years are in scope. "
                "Disqualifiers apply regardless of YoE."
            ),
        )

        location = LocationRequirement(
            cities=["Pune", "Noida", "Hyderabad", "Mumbai", "Delhi NCR", "Bangalore"],
            country="India",
            work_modes=["hybrid", "onsite", "flexible"],
            relocation_open=True,
        )

        jd = StructuredJD(
            title="Senior AI Engineer",
            company="Redrob AI",
            raw_text=raw_text,
            must_have_skills=must_have,
            nice_to_have_skills=nice_to_have,
            disqualifying_patterns=_DISQUALIFYING_PATTERNS,
            experience=experience,
            location=location,
            key_technologies=_KEY_TECHNOLOGIES,
        )

        # Build and attach the optimised embedding text
        jd.embedding_text = jd.build_embedding_text()

        logger.info(
            f"StructuredJD built: {len(must_have)} must-have skills, "
            f"{len(nice_to_have)} nice-to-have skills, "
            f"{len(_DISQUALIFYING_PATTERNS)} disqualifiers, "
            f"embedding_text length={len(jd.embedding_text):,} chars"
        )
        return jd
