"""
Unit and integration tests for src.reranker.cross_encoder

Strategy:
  Unit tests  — use a mock CrossEncoder that returns deterministic raw scores.
                No model download, no network access, runs in milliseconds.
  Integration — marked @pytest.mark.integration; loads the real
                cross-encoder/ms-marco-MiniLM-L-6-v2 model.
                Run with:  pytest tests/test_cross_encoder.py -m integration -v

Covers:
  - RerankResult: construction, validation, helpers
  - CrossEncoderReranker._normalise(): sigmoid + min-max correctness
  - CrossEncoderReranker.sigmoid(): numeric correctness
  - CrossEncoderReranker.rerank(): ordering, top_k, missing ID handling,
    score bounds, parallel list lengths
  - Integration: real model produces valid scores, ML > accountant
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.reranker.cross_encoder import (
    DEFAULT_CE_MODEL,
    CrossEncoderReranker,
    RerankResult,
)
from src.retrieval.retriever import RetrievalResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_retrieval_result(
    n: int,
    query_text: str = "Senior AI Engineer with FAISS and NLP experience",
    seed: int = 0,
) -> RetrievalResult:
    """Create a RetrievalResult with n candidates and random decreasing scores."""
    rng = np.random.default_rng(seed)
    ids = [f"CAND_{i:07d}" for i in range(1, n + 1)]
    scores = np.sort(rng.uniform(0.3, 0.9, n).astype(np.float32))[::-1].copy()
    return RetrievalResult(
        candidate_ids=ids,
        semantic_scores=scores,
        query_text=query_text,
        top_k=n,
        retrieval_time_ms=5.0,
    )


def _make_candidate_texts(ids: list[str], prefix: str = "ML Engineer with ") -> dict[str, str]:
    return {
        cid: f"{prefix}FAISS and Python experience. Candidate {cid}."
        for cid in ids
    }


def _make_reranker_with_mock(
    raw_scores: list[float],
) -> CrossEncoderReranker:
    """
    Return a CrossEncoderReranker whose _model.predict() returns ``raw_scores``.
    """
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model_name = DEFAULT_CE_MODEL
    reranker.batch_size = 32
    reranker.device = "cpu"
    reranker.max_length = 512

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array(raw_scores, dtype=np.float32)
    reranker._model = mock_model
    return reranker


# ---------------------------------------------------------------------------
# Tests — RerankResult
# ---------------------------------------------------------------------------


class TestRerankResult:
    def _make(self, n: int = 5) -> RerankResult:
        ids = [f"CAND_{i:07d}" for i in range(1, n + 1)]
        return RerankResult(
            candidate_ids=ids,
            semantic_scores=np.linspace(0.9, 0.5, n, dtype=np.float32),
            ce_scores=np.linspace(0.95, 0.55, n, dtype=np.float32),
            ce_raw_scores=np.linspace(5.0, 1.0, n, dtype=np.float32),
            query_text="test query",
            rerank_time_ms=50.0,
        )

    def test_constructs_successfully(self) -> None:
        r = self._make()
        assert r.num_candidates == 5

    def test_mismatched_semantic_scores_raises(self) -> None:
        ids = [f"CAND_{i:07d}" for i in range(1, 4)]
        with pytest.raises(ValueError, match="semantic_scores"):
            RerankResult(
                candidate_ids=ids,
                semantic_scores=np.array([0.9, 0.8], dtype=np.float32),  # wrong len
                ce_scores=np.array([0.9, 0.8, 0.7], dtype=np.float32),
                ce_raw_scores=np.array([5.0, 4.0, 3.0], dtype=np.float32),
                query_text="q",
                rerank_time_ms=1.0,
            )

    def test_mismatched_ce_scores_raises(self) -> None:
        ids = [f"CAND_{i:07d}" for i in range(1, 4)]
        with pytest.raises(ValueError, match="ce_scores"):
            RerankResult(
                candidate_ids=ids,
                semantic_scores=np.array([0.9, 0.8, 0.7], dtype=np.float32),
                ce_scores=np.array([0.9], dtype=np.float32),  # wrong len
                ce_raw_scores=np.array([5.0, 4.0, 3.0], dtype=np.float32),
                query_text="q",
                rerank_time_ms=1.0,
            )

    def test_top_n_ids_correct(self) -> None:
        r = self._make(10)
        assert r.top_n_ids(3) == r.candidate_ids[:3]

    def test_as_ce_score_dict_keys(self) -> None:
        r = self._make(5)
        d = r.as_ce_score_dict()
        assert set(d.keys()) == set(r.candidate_ids)

    def test_as_ce_score_dict_values_are_floats(self) -> None:
        r = self._make(5)
        d = r.as_ce_score_dict()
        assert all(isinstance(v, float) for v in d.values())

    def test_as_semantic_score_dict(self) -> None:
        r = self._make(5)
        d = r.as_semantic_score_dict()
        assert set(d.keys()) == set(r.candidate_ids)
        assert abs(d[r.candidate_ids[0]] - float(r.semantic_scores[0])) < 1e-5

    def test_is_frozen(self) -> None:
        r = self._make(3)
        with pytest.raises((TypeError, AttributeError)):
            r.rerank_time_ms = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests — _normalise() and sigmoid()
# ---------------------------------------------------------------------------


class TestNormalise:
    def test_output_in_zero_one_range(self) -> None:
        raw = np.array([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=np.float32)
        norm = CrossEncoderReranker._normalise(raw)
        assert np.all(norm >= 0.0)
        assert np.all(norm <= 1.0)

    def test_order_preserved(self) -> None:
        raw = np.array([1.0, 3.0, 2.0, -1.0, 0.5], dtype=np.float32)
        norm = CrossEncoderReranker._normalise(raw)
        # Normalisation is monotone — order must be preserved
        for i in range(len(raw)):
            for j in range(len(raw)):
                if raw[i] > raw[j]:
                    assert norm[i] >= norm[j]

    def test_degenerate_all_equal_returns_half(self) -> None:
        raw = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        norm = CrossEncoderReranker._normalise(raw)
        np.testing.assert_allclose(norm, np.array([0.5, 0.5, 0.5]), atol=1e-5)

    def test_extremes_map_to_zero_and_one(self) -> None:
        raw = np.array([-10.0, 10.0], dtype=np.float32)
        norm = CrossEncoderReranker._normalise(raw)
        assert abs(float(norm[0])) < 1e-4   # min → 0
        assert abs(float(norm[1]) - 1.0) < 1e-4  # max → 1

    def test_output_dtype_float32(self) -> None:
        raw = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        norm = CrossEncoderReranker._normalise(raw)
        assert norm.dtype == np.float32

    def test_single_element_returns_half(self) -> None:
        raw = np.array([3.5], dtype=np.float32)
        norm = CrossEncoderReranker._normalise(raw)
        np.testing.assert_allclose(norm, [0.5], atol=1e-5)


class TestSigmoid:
    def test_zero_maps_to_half(self) -> None:
        x = np.array([0.0], dtype=np.float32)
        result = CrossEncoderReranker.sigmoid(x)
        np.testing.assert_allclose(result, [0.5], atol=1e-5)

    def test_large_positive_near_one(self) -> None:
        x = np.array([20.0], dtype=np.float32)
        result = CrossEncoderReranker.sigmoid(x)
        assert float(result[0]) > 0.99

    def test_large_negative_near_zero(self) -> None:
        x = np.array([-20.0], dtype=np.float32)
        result = CrossEncoderReranker.sigmoid(x)
        assert float(result[0]) < 0.01

    def test_output_dtype_float32(self) -> None:
        x = np.array([1.0, 2.0], dtype=np.float32)
        assert CrossEncoderReranker.sigmoid(x).dtype == np.float32


# ---------------------------------------------------------------------------
# Tests — rerank() with mock model
# ---------------------------------------------------------------------------


class TestRerank:
    def test_returns_rerank_result(self) -> None:
        n = 10
        rr = _make_retrieval_result(n)
        texts = _make_candidate_texts(rr.candidate_ids)
        raw = list(np.linspace(3.0, 0.5, n))
        reranker = _make_reranker_with_mock(raw)
        result = reranker.rerank(rr, texts, top_k=5)
        assert isinstance(result, RerankResult)

    def test_top_k_respected(self) -> None:
        n = 20
        rr = _make_retrieval_result(n)
        texts = _make_candidate_texts(rr.candidate_ids)
        raw = list(np.random.default_rng(0).uniform(-2, 5, n))
        reranker = _make_reranker_with_mock(raw)
        result = reranker.rerank(rr, texts, top_k=7)
        assert result.num_candidates == 7

    def test_sorted_by_ce_score_descending(self) -> None:
        n = 15
        rr = _make_retrieval_result(n)
        texts = _make_candidate_texts(rr.candidate_ids)
        # Raw scores in intentionally shuffled order
        rng = np.random.default_rng(1)
        raw = list(rng.uniform(-3, 3, n))
        reranker = _make_reranker_with_mock(raw)
        result = reranker.rerank(rr, texts, top_k=n)
        diffs = np.diff(result.ce_scores)
        assert np.all(diffs <= 1e-6), "ce_scores must be non-increasing"

    def test_ce_scores_in_zero_one_range(self) -> None:
        n = 10
        rr = _make_retrieval_result(n)
        texts = _make_candidate_texts(rr.candidate_ids)
        raw = list(np.linspace(-5, 5, n))
        reranker = _make_reranker_with_mock(raw)
        result = reranker.rerank(rr, texts, top_k=n)
        assert np.all(result.ce_scores >= 0.0)
        assert np.all(result.ce_scores <= 1.0)

    def test_parallel_lists_equal_length(self) -> None:
        n = 12
        rr = _make_retrieval_result(n)
        texts = _make_candidate_texts(rr.candidate_ids)
        raw = list(np.zeros(n))
        reranker = _make_reranker_with_mock(raw)
        result = reranker.rerank(rr, texts, top_k=n)
        assert len(result.candidate_ids) == len(result.ce_scores)
        assert len(result.candidate_ids) == len(result.semantic_scores)
        assert len(result.candidate_ids) == len(result.ce_raw_scores)

    def test_missing_candidate_texts_skipped(self) -> None:
        n = 10
        rr = _make_retrieval_result(n)
        # Provide texts for only the first 7
        texts = _make_candidate_texts(rr.candidate_ids[:7])
        raw = list(np.ones(7))  # mock returns 7 scores (only 7 pairs built)
        reranker = _make_reranker_with_mock(raw)
        result = reranker.rerank(rr, texts, top_k=7)
        assert result.num_candidates == 7
        # All returned IDs must be from the 7 provided
        provided = set(rr.candidate_ids[:7])
        assert all(cid in provided for cid in result.candidate_ids)

    def test_all_texts_missing_raises(self) -> None:
        rr = _make_retrieval_result(5)
        reranker = _make_reranker_with_mock([])
        with pytest.raises(ValueError, match="No valid"):
            reranker.rerank(rr, {}, top_k=5)

    def test_top_k_capped_at_valid_count(self) -> None:
        n = 8
        rr = _make_retrieval_result(n)
        texts = _make_candidate_texts(rr.candidate_ids)
        raw = list(np.ones(n))
        reranker = _make_reranker_with_mock(raw)
        result = reranker.rerank(rr, texts, top_k=9999)
        assert result.num_candidates == n

    def test_query_text_preserved(self) -> None:
        rr = _make_retrieval_result(5)
        texts = _make_candidate_texts(rr.candidate_ids)
        raw = [1.0, 2.0, 3.0, 4.0, 5.0]
        reranker = _make_reranker_with_mock(raw)
        result = reranker.rerank(rr, texts, top_k=5)
        assert result.query_text == rr.query_text

    def test_best_raw_score_becomes_top_candidate(self) -> None:
        n = 5
        rr = _make_retrieval_result(n)
        texts = _make_candidate_texts(rr.candidate_ids)
        # Give CAND_0000004 (index 3) the highest raw score
        raw = [1.0, 0.5, 0.2, 8.0, -1.0]
        reranker = _make_reranker_with_mock(raw)
        result = reranker.rerank(rr, texts, top_k=5)
        assert result.candidate_ids[0] == "CAND_0000004"

    def test_rerank_time_ms_positive(self) -> None:
        rr = _make_retrieval_result(5)
        texts = _make_candidate_texts(rr.candidate_ids)
        reranker = _make_reranker_with_mock([1.0] * 5)
        result = reranker.rerank(rr, texts, top_k=5)
        assert result.rerank_time_ms >= 0.0


# ---------------------------------------------------------------------------
# Tests — CrossEncoderReranker construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_model_not_loaded_initially(self) -> None:
        reranker = CrossEncoderReranker()
        assert reranker._model is None

    def test_default_model_name(self) -> None:
        reranker = CrossEncoderReranker()
        assert reranker.model_name == DEFAULT_CE_MODEL

    def test_custom_batch_size(self) -> None:
        reranker = CrossEncoderReranker(batch_size=16)
        assert reranker.batch_size == 16


# ---------------------------------------------------------------------------
# Integration tests — real model (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCrossEncoderIntegration:
    """
    Downloads cross-encoder/ms-marco-MiniLM-L-6-v2 (~90 MB) on first run.
    Run with:  pytest tests/test_cross_encoder.py -m integration -v
    """

    @pytest.fixture(scope="class")
    def reranker(self) -> CrossEncoderReranker:
        r = CrossEncoderReranker(batch_size=8)
        r.load_model()
        return r

    def test_model_loaded(self, reranker: CrossEncoderReranker) -> None:
        assert reranker._model is not None

    def test_rerank_produces_valid_result(
        self, reranker: CrossEncoderReranker
    ) -> None:
        rr = _make_retrieval_result(10)
        texts = _make_candidate_texts(rr.candidate_ids)
        result = reranker.rerank(rr, texts, top_k=5)
        assert isinstance(result, RerankResult)
        assert result.num_candidates == 5

    def test_ce_scores_in_range(self, reranker: CrossEncoderReranker) -> None:
        rr = _make_retrieval_result(10)
        texts = _make_candidate_texts(rr.candidate_ids)
        result = reranker.rerank(rr, texts, top_k=10)
        assert np.all(result.ce_scores >= 0.0)
        assert np.all(result.ce_scores <= 1.0)

    def test_ml_candidate_scores_higher_than_accountant(
        self, reranker: CrossEncoderReranker
    ) -> None:
        """
        A relevant ML engineer candidate must score higher than an accountant
        when the query is the JD embedding text.
        """
        query_text = (
            "Senior AI Engineer: production embeddings FAISS sentence-transformers "
            "NLP ranking systems hybrid search Python evaluation NDCG."
        )
        ml_text = (
            "ML Engineer 7 years. Built FAISS-based candidate retrieval at product company. "
            "Sentence-transformers fine-tuning, NLP pipelines, A/B testing ranking systems."
        )
        acct_text = (
            "Senior Accountant 10 years. Financial reporting, balance sheets, "
            "tax compliance, audit management, budget forecasting."
        )
        rr = RetrievalResult(
            candidate_ids=["CAND_0000001", "CAND_0000002"],
            semantic_scores=np.array([0.8, 0.3], dtype=np.float32),
            query_text=query_text,
            top_k=2,
            retrieval_time_ms=1.0,
        )
        texts = {"CAND_0000001": ml_text, "CAND_0000002": acct_text}
        result = reranker.rerank(rr, texts, top_k=2)
        ce_dict = result.as_ce_score_dict()
        assert ce_dict["CAND_0000001"] > ce_dict["CAND_0000002"], (
            f"ML ({ce_dict['CAND_0000001']:.4f}) should > Accountant "
            f"({ce_dict['CAND_0000002']:.4f})"
        )

    def test_scores_descending_after_rerank(
        self, reranker: CrossEncoderReranker
    ) -> None:
        rr = _make_retrieval_result(20)
        texts = _make_candidate_texts(rr.candidate_ids)
        result = reranker.rerank(rr, texts, top_k=20)
        diffs = np.diff(result.ce_scores)
        assert np.all(diffs <= 1e-5), "ce_scores must be non-increasing"
