"""
Unit tests for src.retrieval.retriever

All tests use:
  - Synthetic L2-normalised numpy vectors (no model download)
  - A mock EmbeddingEngine that returns a fixed query vector
  - A real VectorStore built from the synthetic vectors

Covers:
  - RetrievalResult: construction, validation, score_for(), top_n_ids(), as_score_dict()
  - Retriever: construction, retrieve(), retrieve_by_text(), num_candidates
  - Retriever.from_artifacts(): missing artifact detection
  - Edge cases: empty query, top_k capping, score ordering
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

from src.retrieval.retriever import RetrievalResult, Retriever
from src.retrieval.vector_store import VectorStore
from src.schemas.jd import (
    ExperienceLevel,
    ExperienceRequirement,
    LocationRequirement,
    StructuredJD,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 384


def _unit_vecs(n: int, dim: int = DIM, seed: int = 42) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / norms).astype(np.float32)


def _make_ids(n: int, start: int = 1) -> list[str]:
    return [f"CAND_{i:07d}" for i in range(start, start + n)]


def _make_structured_jd(raw_text: str = "Senior ML Engineer, FAISS, NLP") -> StructuredJD:
    return StructuredJD(
        title="Senior AI Engineer",
        company="Redrob AI",
        raw_text=raw_text,
        must_have_skills=[],
        nice_to_have_skills=[],
        disqualifying_patterns=[],
        experience=ExperienceRequirement(
            min_years=5.0,
            max_years=9.0,
            preferred_level=ExperienceLevel.SENIOR,
        ),
        location=LocationRequirement(),
        key_technologies=[],
        embedding_text="Senior AI Engineer skilled in FAISS, NLP, embeddings.",
    )


class _MockEngine:
    """Minimal EmbeddingEngine stub that returns a fixed query vector."""

    def __init__(self, query_vec: NDArray[np.float32]) -> None:
        self._query_vec = query_vec

    def encode_query(self, text: str) -> NDArray[np.float32]:
        return self._query_vec.reshape(1, -1)


def _make_retriever(
    n: int = 100,
    query_idx: int = 0,
) -> tuple[Retriever, list[str], NDArray[np.float32]]:
    """
    Build a Retriever backed by a synthetic VectorStore.

    The engine's query vector is set to corpus[query_idx] so that
    candidate at query_idx should always be the top result.
    """
    ids = _make_ids(n)
    vecs = _unit_vecs(n)
    store = VectorStore.build(ids, vecs)
    engine = _MockEngine(vecs[query_idx])
    retriever = Retriever(engine=engine, store=store, default_top_k=50)  # type: ignore[arg-type]
    return retriever, ids, vecs


# ---------------------------------------------------------------------------
# Tests — RetrievalResult
# ---------------------------------------------------------------------------


class TestRetrievalResult:
    def test_constructs_successfully(self) -> None:
        ids = _make_ids(5)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
        result = RetrievalResult(
            candidate_ids=ids,
            semantic_scores=scores,
            query_text="test query",
            top_k=5,
            retrieval_time_ms=2.5,
        )
        assert result.num_retrieved == 5

    def test_mismatched_lengths_raises(self) -> None:
        ids = _make_ids(5)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        with pytest.raises(ValueError, match="equal length"):
            RetrievalResult(
                candidate_ids=ids,
                semantic_scores=scores,
                query_text="q",
                top_k=5,
                retrieval_time_ms=1.0,
            )

    def test_score_for_known_id(self) -> None:
        ids = _make_ids(3)
        scores = np.array([0.9, 0.75, 0.6], dtype=np.float32)
        result = RetrievalResult(
            candidate_ids=ids, semantic_scores=scores,
            query_text="q", top_k=3, retrieval_time_ms=1.0,
        )
        assert abs(result.score_for(ids[1]) - 0.75) < 1e-5

    def test_score_for_missing_id_returns_minus_one(self) -> None:
        ids = _make_ids(3)
        scores = np.array([0.9, 0.75, 0.6], dtype=np.float32)
        result = RetrievalResult(
            candidate_ids=ids, semantic_scores=scores,
            query_text="q", top_k=3, retrieval_time_ms=1.0,
        )
        assert result.score_for("CAND_9999999") == -1.0

    def test_top_n_ids_returns_correct_slice(self) -> None:
        ids = _make_ids(10)
        scores = np.linspace(1.0, 0.1, 10, dtype=np.float32)
        result = RetrievalResult(
            candidate_ids=ids, semantic_scores=scores,
            query_text="q", top_k=10, retrieval_time_ms=1.0,
        )
        assert result.top_n_ids(3) == ids[:3]

    def test_as_score_dict_keys_match_ids(self) -> None:
        ids = _make_ids(5)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
        result = RetrievalResult(
            candidate_ids=ids, semantic_scores=scores,
            query_text="q", top_k=5, retrieval_time_ms=1.0,
        )
        d = result.as_score_dict()
        assert set(d.keys()) == set(ids)
        assert abs(d[ids[0]] - 0.9) < 1e-5

    def test_as_score_dict_values_are_floats(self) -> None:
        ids = _make_ids(3)
        scores = np.array([0.9, 0.7, 0.5], dtype=np.float32)
        result = RetrievalResult(
            candidate_ids=ids, semantic_scores=scores,
            query_text="q", top_k=3, retrieval_time_ms=1.0,
        )
        d = result.as_score_dict()
        for v in d.values():
            assert isinstance(v, float)

    def test_num_retrieved_property(self) -> None:
        ids = _make_ids(7)
        scores = np.zeros(7, dtype=np.float32)
        result = RetrievalResult(
            candidate_ids=ids, semantic_scores=scores,
            query_text="q", top_k=10, retrieval_time_ms=1.0,
        )
        assert result.num_retrieved == 7

    def test_result_is_frozen(self) -> None:
        ids = _make_ids(2)
        scores = np.array([0.9, 0.5], dtype=np.float32)
        result = RetrievalResult(
            candidate_ids=ids, semantic_scores=scores,
            query_text="q", top_k=2, retrieval_time_ms=1.0,
        )
        with pytest.raises((TypeError, AttributeError)):
            result.top_k = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests — Retriever construction
# ---------------------------------------------------------------------------


class TestRetrieverConstruction:
    def test_builds_successfully(self) -> None:
        retriever, _, _ = _make_retriever()
        assert isinstance(retriever, Retriever)

    def test_num_candidates_property(self) -> None:
        retriever, ids, _ = _make_retriever(75)
        assert retriever.num_candidates == 75

    def test_invalid_default_top_k_raises(self) -> None:
        ids = _make_ids(10)
        vecs = _unit_vecs(10)
        store = VectorStore.build(ids, vecs)
        engine = _MockEngine(vecs[0])
        with pytest.raises(ValueError, match="default_top_k"):
            Retriever(engine=engine, store=store, default_top_k=0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests — Retriever.retrieve()
# ---------------------------------------------------------------------------


class TestRetrieve:
    def test_returns_retrieval_result(self) -> None:
        retriever, _, _ = _make_retriever()
        jd = _make_structured_jd()
        result = retriever.retrieve(jd)
        assert isinstance(result, RetrievalResult)

    def test_top_k_respected(self) -> None:
        retriever, _, _ = _make_retriever(100)
        jd = _make_structured_jd()
        result = retriever.retrieve(jd, top_k=20)
        assert result.num_retrieved == 20

    def test_default_top_k_used_when_not_specified(self) -> None:
        retriever, _, _ = _make_retriever(100)
        jd = _make_structured_jd()
        result = retriever.retrieve(jd)
        assert result.top_k == 50  # default set in _make_retriever

    def test_scores_descending(self) -> None:
        retriever, _, _ = _make_retriever(100)
        jd = _make_structured_jd()
        result = retriever.retrieve(jd)
        diffs = np.diff(result.semantic_scores)
        assert np.all(diffs <= 1e-6), "Scores must be non-increasing"

    def test_top_result_is_query_vector_itself(self) -> None:
        """When query == corpus[5], corpus[5] must rank first."""
        ids = _make_ids(100)
        vecs = _unit_vecs(100)
        store = VectorStore.build(ids, vecs)
        engine = _MockEngine(vecs[5])  # query is exactly corpus[5]
        retriever = Retriever(engine=engine, store=store, default_top_k=100)  # type: ignore[arg-type]
        jd = _make_structured_jd()
        result = retriever.retrieve(jd)
        assert result.candidate_ids[0] == ids[5]
        assert abs(float(result.semantic_scores[0]) - 1.0) < 1e-4

    def test_scores_in_zero_one_range(self) -> None:
        retriever, _, _ = _make_retriever(100)
        jd = _make_structured_jd()
        result = retriever.retrieve(jd, top_k=100)
        assert np.all(result.semantic_scores >= 0.0)
        assert np.all(result.semantic_scores <= 1.0)

    def test_result_ids_are_subset_of_corpus(self) -> None:
        retriever, ids, _ = _make_retriever(100)
        jd = _make_structured_jd()
        result = retriever.retrieve(jd, top_k=30)
        corpus_set = set(ids)
        for cid in result.candidate_ids:
            assert cid in corpus_set

    def test_query_text_stored_in_result(self) -> None:
        retriever, _, _ = _make_retriever()
        jd = _make_structured_jd()
        result = retriever.retrieve(jd)
        assert jd.embedding_text in result.query_text

    def test_retrieval_time_ms_positive(self) -> None:
        retriever, _, _ = _make_retriever()
        jd = _make_structured_jd()
        result = retriever.retrieve(jd)
        assert result.retrieval_time_ms >= 0.0

    def test_top_k_capped_at_corpus_size(self) -> None:
        retriever, ids, _ = _make_retriever(20)
        jd = _make_structured_jd()
        result = retriever.retrieve(jd, top_k=9999)
        assert result.num_retrieved == 20

    def test_jd_without_embedding_text_uses_build(self) -> None:
        """StructuredJD with empty embedding_text should still work via build_embedding_text()."""
        retriever, _, _ = _make_retriever()
        jd = _make_structured_jd()
        # Force empty embedding_text to exercise the fallback branch
        object.__setattr__(jd, "embedding_text", "")
        result = retriever.retrieve(jd)
        assert isinstance(result, RetrievalResult)


# ---------------------------------------------------------------------------
# Tests — Retriever.retrieve_by_text()
# ---------------------------------------------------------------------------


class TestRetrieveByText:
    def test_returns_retrieval_result(self) -> None:
        retriever, _, _ = _make_retriever()
        result = retriever.retrieve_by_text("ML engineer with FAISS experience")
        assert isinstance(result, RetrievalResult)

    def test_empty_query_raises(self) -> None:
        retriever, _, _ = _make_retriever()
        with pytest.raises(ValueError, match="empty"):
            retriever.retrieve_by_text("   ")

    def test_custom_top_k(self) -> None:
        retriever, _, _ = _make_retriever(100)
        result = retriever.retrieve_by_text("any text", top_k=15)
        assert result.num_retrieved == 15

    def test_scores_descending(self) -> None:
        retriever, _, _ = _make_retriever(100)
        result = retriever.retrieve_by_text("ML ranking retrieval NLP", top_k=50)
        diffs = np.diff(result.semantic_scores)
        assert np.all(diffs <= 1e-6)

    def test_query_text_preserved_in_result(self) -> None:
        retriever, _, _ = _make_retriever()
        query = "Searching for a senior NLP engineer"
        result = retriever.retrieve_by_text(query)
        assert result.query_text == query


# ---------------------------------------------------------------------------
# Tests — Retriever.from_artifacts() — error paths only (no real model)
# ---------------------------------------------------------------------------


class TestFromArtifactsErrors:
    def test_raises_if_embedding_artifacts_missing(self, tmp_path: Path) -> None:
        """No artifacts at all → should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="precompute"):
            Retriever.from_artifacts(tmp_path)

    def test_raises_if_faiss_index_missing(self, tmp_path: Path) -> None:
        """Only embedding artifacts, no FAISS index → should raise FileNotFoundError."""
        # Create the embedding artifact files
        ids = _make_ids(5)
        vecs = _unit_vecs(5)
        from src.embeddings.engine import EmbeddingEngine
        EmbeddingEngine.save(ids, vecs, tmp_path)
        # FAISS index NOT saved — from_artifacts must detect this
        with pytest.raises(FileNotFoundError, match="precompute"):
            Retriever.from_artifacts(tmp_path)
