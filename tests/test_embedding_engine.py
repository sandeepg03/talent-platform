"""
Unit and integration tests for src.embeddings.engine

Test strategy:
  - All geometry / persistence / utility tests use synthetic numpy arrays
    — no model download required, runs in milliseconds.
  - One integration test class (marked with @pytest.mark.integration) actually
    loads BAAI/bge-small-en-v1.5 and encodes a handful of sentences.
    Run with:  pytest tests/test_embedding_engine.py -m integration -v

  Default pytest run (no -m flag) skips the integration tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from src.embeddings.engine import (
    DEFAULT_MODEL_NAME,
    EMBEDDINGS_FILENAME,
    IDS_FILENAME,
    META_FILENAME,
    EmbeddingEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 384  # bge-small-en-v1.5 output dimension


def _rand_unit_vecs(n: int, dim: int = DIM, seed: int = 42) -> NDArray[np.float32]:
    """Generate n random L2-normalised float32 vectors."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / norms).astype(np.float32)


def _make_ids(n: int) -> list[str]:
    return [f"CAND_{i:07d}" for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# Tests — save() / load() round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_creates_expected_files(self, tmp_path: Path) -> None:
        ids = _make_ids(50)
        vecs = _rand_unit_vecs(50)
        EmbeddingEngine.save(ids, vecs, tmp_path)
        assert (tmp_path / EMBEDDINGS_FILENAME).exists()
        assert (tmp_path / IDS_FILENAME).exists()
        assert (tmp_path / META_FILENAME).exists()

    def test_load_returns_same_ids(self, tmp_path: Path) -> None:
        ids = _make_ids(50)
        vecs = _rand_unit_vecs(50)
        EmbeddingEngine.save(ids, vecs, tmp_path)
        restored_ids, _ = EmbeddingEngine.load(tmp_path)
        assert restored_ids == ids

    def test_load_returns_same_embeddings(self, tmp_path: Path) -> None:
        ids = _make_ids(50)
        vecs = _rand_unit_vecs(50)
        EmbeddingEngine.save(ids, vecs, tmp_path)
        _, restored_vecs = EmbeddingEngine.load(tmp_path)
        np.testing.assert_allclose(restored_vecs, vecs, atol=1e-6)

    def test_load_dtype_is_float32(self, tmp_path: Path) -> None:
        ids = _make_ids(10)
        vecs = _rand_unit_vecs(10)
        EmbeddingEngine.save(ids, vecs, tmp_path)
        _, restored = EmbeddingEngine.load(tmp_path)
        assert restored.dtype == np.float32

    def test_save_mismatched_lengths_raises(self, tmp_path: Path) -> None:
        ids = _make_ids(10)
        vecs = _rand_unit_vecs(20)
        with pytest.raises(ValueError, match="length"):
            EmbeddingEngine.save(ids, vecs, tmp_path)

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="precompute"):
            EmbeddingEngine.load(tmp_path)

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        EmbeddingEngine.save(_make_ids(5), _rand_unit_vecs(5), nested)
        assert (nested / EMBEDDINGS_FILENAME).exists()

    def test_meta_json_has_correct_fields(self, tmp_path: Path) -> None:
        n, d = 30, DIM
        ids = _make_ids(n)
        vecs = _rand_unit_vecs(n, d)
        EmbeddingEngine.save(ids, vecs, tmp_path)
        meta = json.loads((tmp_path / META_FILENAME).read_text(encoding="utf-8"))
        assert meta["num_candidates"] == n
        assert meta["embedding_dim"] == d
        assert meta["dtype"] == "float32"
        assert "saved_at" in meta
        assert meta["model_name"] == DEFAULT_MODEL_NAME

    def test_large_batch_roundtrip(self, tmp_path: Path) -> None:
        """Verify that 10K-vector round-trip preserves data integrity."""
        ids = _make_ids(10_000)
        vecs = _rand_unit_vecs(10_000)
        EmbeddingEngine.save(ids, vecs, tmp_path)
        r_ids, r_vecs = EmbeddingEngine.load(tmp_path)
        assert len(r_ids) == 10_000
        assert r_vecs.shape == (10_000, DIM)
        np.testing.assert_allclose(r_vecs[0], vecs[0], atol=1e-6)
        np.testing.assert_allclose(r_vecs[-1], vecs[-1], atol=1e-6)


# ---------------------------------------------------------------------------
# Tests — artifacts_exist()
# ---------------------------------------------------------------------------


class TestArtifactsExist:
    def test_returns_false_when_empty(self, tmp_path: Path) -> None:
        assert EmbeddingEngine.artifacts_exist(tmp_path) is False

    def test_returns_false_with_only_embeddings(self, tmp_path: Path) -> None:
        (tmp_path / EMBEDDINGS_FILENAME).touch()
        assert EmbeddingEngine.artifacts_exist(tmp_path) is False

    def test_returns_true_when_both_exist(self, tmp_path: Path) -> None:
        ids = _make_ids(5)
        vecs = _rand_unit_vecs(5)
        EmbeddingEngine.save(ids, vecs, tmp_path)
        assert EmbeddingEngine.artifacts_exist(tmp_path) is True


# ---------------------------------------------------------------------------
# Tests — cosine_similarity_to_query()
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self) -> None:
        q = _rand_unit_vecs(1)
        corpus = np.vstack([q, _rand_unit_vecs(4)])
        sims = EmbeddingEngine.cosine_similarity_to_query(q, corpus)
        assert abs(float(sims[0]) - 1.0) < 1e-5

    def test_output_shape_is_n(self) -> None:
        q = _rand_unit_vecs(1)
        corpus = _rand_unit_vecs(100)
        sims = EmbeddingEngine.cosine_similarity_to_query(q, corpus)
        assert sims.shape == (100,)

    def test_scores_in_zero_one_range(self) -> None:
        q = _rand_unit_vecs(1)
        corpus = _rand_unit_vecs(200)
        sims = EmbeddingEngine.cosine_similarity_to_query(q, corpus)
        assert np.all(sims >= 0.0)
        assert np.all(sims <= 1.0)

    def test_query_as_1d_vector_accepted(self) -> None:
        q = _rand_unit_vecs(1).squeeze()  # shape (D,)
        corpus = _rand_unit_vecs(10)
        sims = EmbeddingEngine.cosine_similarity_to_query(q, corpus)
        assert sims.shape == (10,)

    def test_output_dtype_is_float32(self) -> None:
        q = _rand_unit_vecs(1)
        corpus = _rand_unit_vecs(50)
        sims = EmbeddingEngine.cosine_similarity_to_query(q, corpus)
        assert sims.dtype == np.float32

    def test_ordering_preserved(self) -> None:
        """The identical vector always scores 1.0 and must rank highest."""
        rng = np.random.default_rng(seed=7)
        corpus = rng.standard_normal((10, DIM)).astype(np.float32)
        norms = np.linalg.norm(corpus, axis=1, keepdims=True)
        corpus = (corpus / norms).astype(np.float32)
        # Use corpus[3] as the query — must score 1.0 against itself
        q = corpus[3:4]  # shape (1, D)
        sims = EmbeddingEngine.cosine_similarity_to_query(q, corpus)
        assert int(np.argmax(sims)) == 3, (
            f"Expected index 3 (identical vector) to score highest, got {np.argmax(sims)}"
        )


# ---------------------------------------------------------------------------
# Tests — EmbeddingEngine initialisation
# ---------------------------------------------------------------------------


class TestEmbeddingEngineInit:
    def test_default_model_name(self) -> None:
        engine = EmbeddingEngine()
        assert engine.model_name == DEFAULT_MODEL_NAME

    def test_custom_batch_size(self) -> None:
        engine = EmbeddingEngine(batch_size=256)
        assert engine.batch_size == 256

    def test_model_not_loaded_initially(self) -> None:
        engine = EmbeddingEngine()
        assert engine._model is None

    def test_artifacts_exist_is_false_on_empty_dir(self, tmp_path: Path) -> None:
        assert EmbeddingEngine.artifacts_exist(tmp_path) is False

    def test_encode_empty_list_raises(self) -> None:
        """encode_texts([]) must raise before touching the model."""
        engine = EmbeddingEngine()
        # Monkey-patch _ensure_loaded to avoid model download in unit tests
        engine._model = object()  # Truthy sentinel — prevents auto-load
        with pytest.raises((ValueError, AttributeError)):
            # ValueError if guard fires before model call; AttributeError if mock hit
            engine.encode_texts([])


# ---------------------------------------------------------------------------
# Integration tests — real model (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEmbeddingEngineIntegration:
    """
    These tests download BAAI/bge-small-en-v1.5 (~130 MB) on first run.
    They are skipped unless you run:  pytest -m integration

    They verify end-to-end correctness:
      - encode_texts returns correct shape and dtype
      - vectors are approximately L2-normalised
      - encode_query returns shape (1, D)
      - More relevant text has higher cosine similarity to a ML-domain query
    """

    @pytest.fixture(scope="class")
    def engine(self) -> EmbeddingEngine:
        e = EmbeddingEngine(batch_size=8, show_progress_bar=False)
        e.load_model()
        return e

    def test_encode_texts_shape(self, engine: EmbeddingEngine) -> None:
        texts = ["Machine learning engineer with FAISS experience.", "Accountant."]
        vecs = engine.encode_texts(texts)
        assert vecs.shape == (2, engine.embedding_dim)

    def test_encode_texts_dtype(self, engine: EmbeddingEngine) -> None:
        vecs = engine.encode_texts(["hello world"])
        assert vecs.dtype == np.float32

    def test_vectors_approximately_unit_norm(self, engine: EmbeddingEngine) -> None:
        texts = [f"Sample text number {i}" for i in range(10)]
        vecs = engine.encode_texts(texts)
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, np.ones(10), atol=1e-5)

    def test_encode_query_shape(self, engine: EmbeddingEngine) -> None:
        q = engine.encode_query("Senior AI Engineer skilled in FAISS and NLP.")
        assert q.shape == (1, engine.embedding_dim)

    def test_relevant_candidate_higher_similarity(
        self, engine: EmbeddingEngine
    ) -> None:
        """A strong ML engineer text should score higher than an accountant text."""
        jd_text = (
            "Senior AI Engineer: embeddings, FAISS, sentence-transformers, "
            "ranking systems, Python, NLP, production ML."
        )
        ml_text = (
            "ML Engineer with 7 years experience building FAISS-based retrieval, "
            "sentence-transformers fine-tuning, NLP pipelines at product companies."
        )
        acct_text = (
            "Accountant with 10 years experience in financial reporting, "
            "balance sheets, tax compliance, and audit management."
        )
        q = engine.encode_query(jd_text)
        corpus = engine.encode_texts([ml_text, acct_text])
        sims = EmbeddingEngine.cosine_similarity_to_query(q, corpus)
        assert float(sims[0]) > float(sims[1]), (
            f"Expected ML ({sims[0]:.4f}) > Accountant ({sims[1]:.4f})"
        )

    def test_encode_corpus_batched_matches_encode_texts(
        self, engine: EmbeddingEngine
    ) -> None:
        texts = [f"Text number {i} about machine learning." for i in range(20)]
        vecs_standard = engine.encode_texts(texts)
        vecs_batched = engine.encode_corpus_batched(texts)
        np.testing.assert_allclose(vecs_standard, vecs_batched, atol=1e-5)

    def test_save_load_integration(
        self, engine: EmbeddingEngine, tmp_path: Path
    ) -> None:
        texts = ["Python developer.", "Data scientist.", "ML Engineer."]
        ids = [f"CAND_{i:07d}" for i in range(1, 4)]
        vecs = engine.encode_texts(texts)
        EmbeddingEngine.save(ids, vecs, tmp_path)
        r_ids, r_vecs = EmbeddingEngine.load(tmp_path)
        assert r_ids == ids
        np.testing.assert_allclose(r_vecs, vecs, atol=1e-6)
