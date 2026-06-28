"""
Unit tests for src.retrieval.vector_store

All tests use synthetic numpy vectors — no model download, runs in milliseconds.

Covers:
  - VectorStore.build(): construction, validation, properties
  - VectorStore.save() / load(): binary round-trip, file existence checks
  - VectorStore.search(): top-K correctness, score ordering, score bounds,
    sentinel filtering, invalid top_k
  - VectorStore.search_all(): completeness
  - VectorStore.get_score_for_id(): exact lookup, missing ID
  - VectorStore.artifacts_exist(): presence/absence detection
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pytest
from numpy.typing import NDArray

from src.retrieval.vector_store import INDEX_FILENAME, VectorStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 384


def _unit_vecs(n: int, dim: int = DIM, seed: int = 0) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / norms).astype(np.float32)


def _make_ids(n: int, start: int = 1) -> list[str]:
    return [f"CAND_{i:07d}" for i in range(start, start + n)]


def _make_store(n: int = 100, dim: int = DIM) -> tuple[VectorStore, list[str], NDArray]:
    ids = _make_ids(n)
    vecs = _unit_vecs(n, dim)
    store = VectorStore.build(ids, vecs)
    return store, ids, vecs


# ---------------------------------------------------------------------------
# Tests — build()
# ---------------------------------------------------------------------------


class TestBuild:
    def test_returns_vector_store(self) -> None:
        store, _, _ = _make_store(50)
        assert isinstance(store, VectorStore)

    def test_num_candidates_correct(self) -> None:
        store, _, _ = _make_store(200)
        assert store.num_candidates == 200

    def test_candidate_ids_preserved(self) -> None:
        ids = _make_ids(10)
        vecs = _unit_vecs(10)
        store = VectorStore.build(ids, vecs)
        assert store.candidate_ids == ids

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            VectorStore.build(_make_ids(10), _unit_vecs(20))

    def test_float64_input_converted(self) -> None:
        ids = _make_ids(5)
        vecs = _unit_vecs(5).astype(np.float64)
        store = VectorStore.build(ids, vecs)
        assert store.num_candidates == 5

    def test_large_corpus(self) -> None:
        """10K candidates must build without error."""
        store, _, _ = _make_store(10_000)
        assert store.num_candidates == 10_000


# ---------------------------------------------------------------------------
# Tests — save() / load()
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_creates_index_file(self, tmp_path: Path) -> None:
        store, ids, vecs = _make_store(50)
        # Save IDs via numpy (as EmbeddingEngine would do)
        np.save(tmp_path / "candidate_ids.npy", np.array(ids, dtype=object))
        store.save(tmp_path)
        assert (tmp_path / INDEX_FILENAME).exists()

    def test_load_returns_vector_store(self, tmp_path: Path) -> None:
        store, ids, vecs = _make_store(50)
        np.save(tmp_path / "candidate_ids.npy", np.array(ids, dtype=object))
        store.save(tmp_path)
        loaded = VectorStore.load(tmp_path)
        assert isinstance(loaded, VectorStore)

    def test_load_preserves_num_candidates(self, tmp_path: Path) -> None:
        store, ids, _ = _make_store(75)
        np.save(tmp_path / "candidate_ids.npy", np.array(ids, dtype=object))
        store.save(tmp_path)
        loaded = VectorStore.load(tmp_path)
        assert loaded.num_candidates == 75

    def test_load_preserves_candidate_ids(self, tmp_path: Path) -> None:
        store, ids, _ = _make_store(30)
        np.save(tmp_path / "candidate_ids.npy", np.array(ids, dtype=object))
        store.save(tmp_path)
        loaded = VectorStore.load(tmp_path)
        assert loaded.candidate_ids == ids

    def test_load_search_results_match_original(self, tmp_path: Path) -> None:
        """Search results must be identical before and after save/load."""
        store, ids, vecs = _make_store(50)
        np.save(tmp_path / "candidate_ids.npy", np.array(ids, dtype=object))
        store.save(tmp_path)
        loaded = VectorStore.load(tmp_path)

        query = vecs[0:1]
        r_ids1, scores1 = store.search(query, top_k=10)
        r_ids2, scores2 = loaded.search(query, top_k=10)
        assert r_ids1 == r_ids2
        np.testing.assert_allclose(scores1, scores2, atol=1e-5)

    def test_load_missing_index_raises(self, tmp_path: Path) -> None:
        np.save(tmp_path / "candidate_ids.npy", np.array(_make_ids(5), dtype=object))
        with pytest.raises(FileNotFoundError, match="precompute"):
            VectorStore.load(tmp_path)

    def test_load_missing_ids_raises(self, tmp_path: Path) -> None:
        store, _, _ = _make_store(5)
        store.save(tmp_path)  # only writes faiss.index
        with pytest.raises(FileNotFoundError, match="precompute"):
            VectorStore.load(tmp_path)

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "x" / "y"
        store, ids, _ = _make_store(5)
        np.save(tmp_path / "candidate_ids.npy", np.array(ids, dtype=object))
        store.save(nested)
        assert (nested / INDEX_FILENAME).exists()


# ---------------------------------------------------------------------------
# Tests — search()
# ---------------------------------------------------------------------------


class TestSearch:
    def test_returns_tuple_of_ids_and_scores(self) -> None:
        store, _, vecs = _make_store(100)
        result_ids, scores = store.search(vecs[0], top_k=10)
        assert isinstance(result_ids, list)
        assert isinstance(scores, np.ndarray)

    def test_top_k_respected(self) -> None:
        store, _, vecs = _make_store(100)
        result_ids, scores = store.search(vecs[0], top_k=10)
        assert len(result_ids) == 10
        assert len(scores) == 10

    def test_scores_descending(self) -> None:
        store, _, vecs = _make_store(100)
        _, scores = store.search(vecs[0], top_k=50)
        diffs = np.diff(scores)
        assert np.all(diffs <= 1e-6), "Scores must be non-increasing"

    def test_self_similarity_first(self) -> None:
        """The query vector itself must be the top result."""
        store, ids, vecs = _make_store(100)
        result_ids, scores = store.search(vecs[5], top_k=5)
        assert result_ids[0] == ids[5]
        assert abs(float(scores[0]) - 1.0) < 1e-4

    def test_scores_in_zero_one_range(self) -> None:
        store, _, vecs = _make_store(100)
        _, scores = store.search(vecs[0], top_k=100)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_scores_dtype_float32(self) -> None:
        store, _, vecs = _make_store(50)
        _, scores = store.search(vecs[0], top_k=10)
        assert scores.dtype == np.float32

    def test_query_as_1d_accepted(self) -> None:
        store, _, vecs = _make_store(50)
        q = vecs[0].squeeze()  # (D,) instead of (1,D)
        result_ids, scores = store.search(q, top_k=5)
        assert len(result_ids) == 5

    def test_top_k_capped_at_n(self) -> None:
        store, _, vecs = _make_store(20)
        result_ids, scores = store.search(vecs[0], top_k=1000)
        assert len(result_ids) == 20

    def test_invalid_top_k_raises(self) -> None:
        store, _, vecs = _make_store(10)
        with pytest.raises(ValueError, match="top_k"):
            store.search(vecs[0], top_k=0)

    def test_result_ids_are_valid_candidate_ids(self) -> None:
        store, ids, vecs = _make_store(50)
        result_ids, _ = store.search(vecs[0], top_k=20)
        id_set = set(ids)
        for cid in result_ids:
            assert cid in id_set, f"Unexpected id: {cid}"


# ---------------------------------------------------------------------------
# Tests — search_all()
# ---------------------------------------------------------------------------


class TestSearchAll:
    def test_returns_all_candidates(self) -> None:
        store, ids, vecs = _make_store(50)
        result_ids, scores = store.search_all(vecs[0])
        assert len(result_ids) == 50
        assert len(scores) == 50

    def test_all_ids_present(self) -> None:
        store, ids, vecs = _make_store(50)
        result_ids, _ = store.search_all(vecs[0])
        assert set(result_ids) == set(ids)

    def test_scores_descending(self) -> None:
        store, _, vecs = _make_store(50)
        _, scores = store.search_all(vecs[0])
        diffs = np.diff(scores)
        assert np.all(diffs <= 1e-6)


# ---------------------------------------------------------------------------
# Tests — get_score_for_id()
# ---------------------------------------------------------------------------


class TestGetScoreForId:
    def test_self_score_is_one(self) -> None:
        store, ids, vecs = _make_store(50)
        score = store.get_score_for_id(ids[7], vecs[7])
        assert abs(score - 1.0) < 1e-4

    def test_missing_id_returns_minus_one(self) -> None:
        store, _, vecs = _make_store(20)
        score = store.get_score_for_id("CAND_9999999", vecs[0])
        assert score == -1.0

    def test_score_in_zero_one_range(self) -> None:
        store, ids, vecs = _make_store(50)
        score = store.get_score_for_id(ids[3], vecs[0])
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Tests — artifacts_exist()
# ---------------------------------------------------------------------------


class TestArtifactsExist:
    def test_false_when_empty_dir(self, tmp_path: Path) -> None:
        assert VectorStore.artifacts_exist(tmp_path) is False

    def test_true_after_save(self, tmp_path: Path) -> None:
        store, _, _ = _make_store(5)
        store.save(tmp_path)
        assert VectorStore.artifacts_exist(tmp_path) is True

    def test_false_when_only_ids_present(self, tmp_path: Path) -> None:
        ids = _make_ids(5)
        np.save(tmp_path / "candidate_ids.npy", np.array(ids, dtype=object))
        assert VectorStore.artifacts_exist(tmp_path) is False


# ---------------------------------------------------------------------------
# Tests — constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_mismatched_index_and_ids_raises(self) -> None:
        index = faiss.IndexFlatIP(DIM)
        vecs = _unit_vecs(10)
        index.add(np.ascontiguousarray(vecs, dtype=np.float32))
        with pytest.raises(ValueError, match="candidate_ids"):
            VectorStore(index, _make_ids(5))  # 10 in index, 5 in ids
