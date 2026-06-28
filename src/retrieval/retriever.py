"""
Retriever — orchestrates semantic retrieval of candidate profiles for a JD.

Responsibilities:
  - Accept a StructuredJD and return the top-K most semantically relevant
    candidates with their semantic similarity scores
  - Bridge between EmbeddingEngine (query encoding) and VectorStore (ANN search)
  - Expose a RetrievalResult dataclass consumed by the cross-encoder reranker
  - Optionally load both artifacts from disk (precomputed path) or accept
    injected instances (for testing and the FastAPI server)

Architecture position:
  JD Parser → StructuredJD
                 ↓
              Retriever  ←── EmbeddingEngine  (encodes query)
                 ↓       ←── VectorStore      (exact search)
           RetrievalResult (top-K ids + semantic scores)
                 ↓
          CrossEncoderReranker

Performance:
  - Query encoding:  ~5ms on CPU (one forward pass, 384 dim)
  - FAISS search:    <1ms on CPU for 100K candidates
  - Total latency:   <10ms for top-500 retrieval
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from src.embeddings.engine import EmbeddingEngine
from src.retrieval.vector_store import VectorStore
from src.schemas.jd import StructuredJD

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalResult:
    """
    Output of a single retrieval run.

    All parallel lists share the same index:
      candidate_ids[i], semantic_scores[i] describe the same candidate.

    Lists are ordered descending by semantic_score (best match first).
    """

    candidate_ids: list[str]
    semantic_scores: NDArray[np.float32]
    query_text: str  # The embedding text used for retrieval (for logging/debug)
    top_k: int
    retrieval_time_ms: float

    def __post_init__(self) -> None:
        if len(self.candidate_ids) != len(self.semantic_scores):
            raise ValueError(
                f"candidate_ids ({len(self.candidate_ids)}) and "
                f"semantic_scores ({len(self.semantic_scores)}) must have equal length."
            )

    @property
    def num_retrieved(self) -> int:
        """Actual number of candidates retrieved (may be < top_k if corpus is small)."""
        return len(self.candidate_ids)

    def score_for(self, candidate_id: str) -> float:
        """
        Return the semantic score for a candidate by ID.

        Returns -1.0 if the candidate was not in the retrieved set.
        """
        try:
            idx = self.candidate_ids.index(candidate_id)
            return float(self.semantic_scores[idx])
        except ValueError:
            return -1.0

    def top_n_ids(self, n: int) -> list[str]:
        """Return the top-N candidate_ids from this result."""
        return self.candidate_ids[:n]

    def as_score_dict(self) -> dict[str, float]:
        """Return a dict mapping candidate_id → semantic_score for O(1) lookups."""
        return {cid: float(score) for cid, score in zip(self.candidate_ids, self.semantic_scores)}


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Retriever:
    """
    Semantic retriever: encodes a JD query and searches the FAISS index.

    Two construction patterns:

    Pattern A — inject pre-built instances (testing, FastAPI server):
        engine = EmbeddingEngine()
        engine.load_model()
        store = VectorStore.load(artifacts_dir)
        retriever = Retriever(engine=engine, store=store)

    Pattern B — load from artifacts directory (rank.py CLI):
        retriever = Retriever.from_artifacts(artifacts_dir)

    After construction, call retrieve() once per JD.
    """

    def __init__(
        self,
        engine: EmbeddingEngine,
        store: VectorStore,
        default_top_k: int = 500,
    ) -> None:
        """
        Args:
            engine:       Loaded EmbeddingEngine (model must already be loaded).
            store:        Populated VectorStore (index must already be built/loaded).
            default_top_k: Default number of candidates to retrieve per query.
                           The cross-encoder reranker will trim this to 100.
        """
        if default_top_k < 1:
            raise ValueError(f"default_top_k must be >= 1, got {default_top_k}")
        self._engine = engine
        self._store = store
        self.default_top_k = default_top_k

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: Path | str,
        *,
        default_top_k: int = 500,
        model_name: str | None = None,
        batch_size: int = 512,
    ) -> "Retriever":
        """
        Build a Retriever by loading pre-computed artifacts from disk.

        This is the path used by rank.py — no re-encoding of 100K candidates.

        Args:
            artifacts_dir: Directory containing embeddings.npy, candidate_ids.npy,
                           and faiss.index (all written by precompute.py).
            default_top_k: Default retrieval depth.
            model_name:    Override the embedding model (defaults to bge-small-en-v1.5).
            batch_size:    Model inference batch size.

        Raises:
            FileNotFoundError: If any required artifact is missing.
        """
        from src.embeddings.engine import DEFAULT_MODEL_NAME

        d = Path(artifacts_dir)
        logger.info(f"Building Retriever from artifacts: {d}")

        # Validate artifacts before loading model (fast fail)
        if not EmbeddingEngine.artifacts_exist(d):
            raise FileNotFoundError(
                f"Embedding artifacts not found in {d}. Run precompute.py first."
            )
        if not VectorStore.artifacts_exist(d):
            raise FileNotFoundError(f"FAISS index not found in {d}. Run precompute.py first.")

        engine = EmbeddingEngine(
            model_name=model_name or DEFAULT_MODEL_NAME,
            batch_size=batch_size,
        )
        engine.load_model()

        store = VectorStore.load(d)

        logger.info(
            f"Retriever ready: {store.num_candidates:,} candidates indexed, "
            f"default_top_k={default_top_k}"
        )
        return cls(engine=engine, store=store, default_top_k=default_top_k)

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        jd: StructuredJD,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """
        Encode the JD and retrieve the top-K most semantically similar candidates.

        Args:
            jd:    The parsed StructuredJD (produced by JDParser).
            top_k: Number of candidates to return. Defaults to self.default_top_k.

        Returns:
            RetrievalResult with candidate_ids and semantic_scores, sorted
            descending by score.
        """
        k = top_k if top_k is not None else self.default_top_k
        query_text = jd.embedding_text or jd.build_embedding_text()

        logger.info(f"Retrieval: encoding JD query ({len(query_text)} chars), top_k={k}")
        t0 = time.perf_counter()

        query_vec: NDArray[np.float32] = self._engine.encode_query(query_text)
        result_ids, scores = self._store.search(query_vec, top_k=k)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            f"Retrieval complete: {len(result_ids)} candidates in {elapsed_ms:.1f}ms "
            f"— top score={scores[0]:.4f}, bottom score={scores[-1]:.4f}"
        )

        return RetrievalResult(
            candidate_ids=result_ids,
            semantic_scores=scores,
            query_text=query_text,
            top_k=k,
            retrieval_time_ms=elapsed_ms,
        )

    def retrieve_by_text(
        self,
        query_text: str,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """
        Retrieve candidates using an arbitrary text query string.

        Useful for debugging and the Streamlit UI's free-text search mode.

        Args:
            query_text: Any natural language search query.
            top_k:      Number of candidates to return.

        Returns:
            RetrievalResult sorted descending by semantic_score.
        """
        if not query_text.strip():
            raise ValueError("query_text must not be empty.")
        k = top_k if top_k is not None else self.default_top_k

        logger.info(f"Free-text retrieval: {query_text[:80]!r}, top_k={k}")
        t0 = time.perf_counter()

        query_vec: NDArray[np.float32] = self._engine.encode_query(query_text)
        result_ids, scores = self._store.search(query_vec, top_k=k)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return RetrievalResult(
            candidate_ids=result_ids,
            semantic_scores=scores,
            query_text=query_text,
            top_k=k,
            retrieval_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_candidates(self) -> int:
        """Total number of candidates in the vector index."""
        return self._store.num_candidates
