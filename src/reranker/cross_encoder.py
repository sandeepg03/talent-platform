"""
Cross Encoder Reranker — second-pass scoring for top-K candidate pool.

Architecture position:
  Retriever → RetrievalResult (top-500)
                    ↓
             CrossEncoderReranker
                    ↓
           RerankResult (top-K, cross-encoder scored)
                    ↓
           Feature Engineer

Why a cross encoder:
  The bi-encoder (FAISS retrieval) compares JD and candidate independently —
  fast, but it misses fine-grained token-level interactions.
  The cross encoder receives (query, candidate) as a SINGLE sequence, giving
  the model full attention over both sides simultaneously.  This captures:
    - Semantic nuances (e.g. "managed FAISS" vs "read FAISS tutorial")
    - Negation ("no production deployment" → disqualifier)
    - Experience-length context
  At 500 candidates this is ~2s on CPU — well within the 5-minute budget.

Model choice: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22M parameters, fast CPU inference
  - Strong BEIR/MS-MARCO reranking performance for sub-100M models
  - Returns logits (not probabilities) — we sigmoid-normalise to [0, 1]
  - Can be swapped at construction time for any HuggingFace cross-encoder

Score normalisation:
  Raw cross-encoder logits are unbounded (typically -10 to +10 for this model).
  We apply sigmoid to map to (0, 1), then min-max normalise the batch
  to spread scores across [0, 1] for the hybrid scoring formula.
  Min-max is computed over the retrieved pool only (not all 100K candidates).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from src.retrieval.retriever import RetrievalResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CE_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CE_BATCH_SIZE: int = 64  # cross-encoders are heavier than bi-encoders


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RerankResult:
    """
    Output of the cross-encoder reranking stage.

    Parallel lists: candidate_ids[i] ↔ semantic_scores[i] ↔ ce_scores[i].
    All lists are sorted descending by ce_score (best match first).
    """

    candidate_ids: list[str]
    semantic_scores: NDArray[np.float32]  # from FAISS bi-encoder
    ce_scores: NDArray[np.float32]  # normalised cross-encoder scores [0,1]
    ce_raw_scores: NDArray[np.float32]  # raw logits before normalisation
    query_text: str
    rerank_time_ms: float

    def __post_init__(self) -> None:
        n = len(self.candidate_ids)
        for arr_name, arr in (
            ("semantic_scores", self.semantic_scores),
            ("ce_scores", self.ce_scores),
            ("ce_raw_scores", self.ce_raw_scores),
        ):
            if len(arr) != n:
                raise ValueError(
                    f"{arr_name} length ({len(arr)}) must equal candidate_ids length ({n})."
                )

    @property
    def num_candidates(self) -> int:
        return len(self.candidate_ids)

    def top_n_ids(self, n: int) -> list[str]:
        """Return the top-N candidate IDs ranked by ce_score."""
        return self.candidate_ids[:n]

    def as_ce_score_dict(self) -> dict[str, float]:
        """Return dict mapping candidate_id → ce_score for O(1) lookups."""
        return {cid: float(s) for cid, s in zip(self.candidate_ids, self.ce_scores)}

    def as_semantic_score_dict(self) -> dict[str, float]:
        """Return dict mapping candidate_id → semantic_score for O(1) lookups."""
        return {cid: float(s) for cid, s in zip(self.candidate_ids, self.semantic_scores)}


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class CrossEncoderReranker:
    """
    Cross-encoder reranker wrapping a HuggingFace CrossEncoder model.

    Usage:
        reranker = CrossEncoderReranker()          # model not loaded yet
        reranker.load_model()                      # download / load weights
        result = reranker.rerank(retrieval_result, candidate_texts, top_k=200)

    The reranker is stateless after model loading — safe for repeated calls.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CE_MODEL,
        batch_size: int = DEFAULT_CE_BATCH_SIZE,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        """
        Args:
            model_name:  HuggingFace model identifier for the cross-encoder.
            batch_size:  Pairs per inference batch. Lower on OOM-prone hardware.
            device:      'cuda', 'cpu', or None (auto-detect).
            max_length:  Max token length for the (query, candidate) pair.
                         512 is the BERT limit and sufficient for this use-case.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self._model: object | None = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load the CrossEncoder model into memory.

        Idempotent — safe to call multiple times.
        Auto-detects GPU; falls back to CPU.
        """
        if self._model is not None:
            return

        from sentence_transformers import CrossEncoder

        effective_device = self._resolve_device()
        logger.info(f"Loading CrossEncoder: {self.model_name!r} on device={effective_device!r}")
        t0 = time.perf_counter()
        self._model = CrossEncoder(
            self.model_name,
            max_length=self.max_length,
            device=effective_device,
        )
        elapsed = time.perf_counter() - t0
        logger.info(f"CrossEncoder loaded in {elapsed:.1f}s")

    def _resolve_device(self) -> str:
        if self.device is not None:
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load_model()

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def rerank(
        self,
        retrieval_result: RetrievalResult,
        candidate_texts: dict[str, str],
        *,
        top_k: int = 200,
    ) -> RerankResult:
        """
        Score every candidate in ``retrieval_result`` with the cross-encoder
        and return the top-K sorted by cross-encoder score descending.

        Args:
            retrieval_result:  Output of Retriever.retrieve() — top-500 candidates.
            candidate_texts:   Mapping of candidate_id → embedding text (from
                               CandidateTextBuilder). Only IDs in retrieval_result
                               are accessed; missing IDs are skipped with a warning.
            top_k:             How many candidates to keep after reranking.
                               The Feature Engineer receives this truncated list.

        Returns:
            RerankResult with candidates sorted by ce_score descending.
        """
        self._ensure_loaded()

        query = retrieval_result.query_text
        candidate_ids = retrieval_result.candidate_ids
        semantic_scores = retrieval_result.semantic_scores

        # Build (query, candidate_text) pairs in retrieval order.
        # Candidates missing from candidate_texts are dropped gracefully.
        valid_ids: list[str] = []
        valid_sem: list[float] = []
        pairs: list[tuple[str, str]] = []

        sem_dict = dict(zip(candidate_ids, semantic_scores.tolist()))
        for cid in candidate_ids:
            text = candidate_texts.get(cid)
            if text is None:
                logger.warning(f"candidate_id {cid!r} not found in candidate_texts — skipping.")
                continue
            # Truncate candidate text to keep within max_length budget
            # Cross-encoder tokenises both sides together; trim to ~1000 chars
            truncated = text[:1000]
            pairs.append((query[:512], truncated))
            valid_ids.append(cid)
            valid_sem.append(sem_dict[cid])

        if not pairs:
            raise ValueError(
                "No valid (query, candidate_text) pairs could be constructed. "
                "Ensure candidate_texts contains the retrieved candidate IDs."
            )

        logger.info(
            f"CrossEncoder scoring {len(pairs)} pairs (batch_size={self.batch_size}, top_k={top_k})"
        )
        t0 = time.perf_counter()

        raw_scores: NDArray[np.float32] = self._model.predict(  # type: ignore[union-attr]
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=len(pairs) > 100,
            convert_to_numpy=True,
        ).astype(np.float32)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            f"CrossEncoder scoring complete: {len(pairs)} pairs in {elapsed_ms:.0f}ms "
            f"({len(pairs) / (elapsed_ms / 1000):.0f} pairs/sec)"
        )

        # Normalise raw logits → [0, 1]
        norm_scores = self._normalise(raw_scores)

        # Sort by ce_score descending and truncate to top_k
        order = np.argsort(-norm_scores)
        k = min(top_k, len(valid_ids))
        top_order = order[:k]

        sem_array = np.array(valid_sem, dtype=np.float32)

        return RerankResult(
            candidate_ids=[valid_ids[i] for i in top_order],
            semantic_scores=sem_array[top_order],
            ce_scores=norm_scores[top_order],
            ce_raw_scores=raw_scores[top_order],
            query_text=query,
            rerank_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Score utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(raw: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Normalise an array of raw cross-encoder logits to [0, 1].

        Strategy (two-step):
          1. Sigmoid: maps unbounded logits to (0, 1), preserving ordinality.
          2. Min-max rescale: spreads the sigmoid outputs across the full [0, 1]
             range within this batch so the downstream scorer is not compressed
             into a small interval of the [0, 1] weight range.

        If all values are equal (degenerate batch), returns 0.5 for all.
        """
        # Step 1: sigmoid
        sigmoid = 1.0 / (1.0 + np.exp(-raw.astype(np.float64))).astype(np.float32)

        # Step 2: min-max
        lo, hi = float(sigmoid.min()), float(sigmoid.max())
        if hi - lo < 1e-8:
            return np.full_like(sigmoid, 0.5, dtype=np.float32)
        rescaled = (sigmoid - lo) / (hi - lo)
        return rescaled.astype(np.float32)

    @staticmethod
    def sigmoid(x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Public sigmoid utility — exposed for testing and downstream use."""
        return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)
