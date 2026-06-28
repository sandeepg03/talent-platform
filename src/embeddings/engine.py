"""
Embedding Engine — converts text into dense vector representations.

Responsibilities:
  - Load and cache the SentenceTransformer model (BAAI/bge-small-en-v1.5)
  - Encode candidate texts in batches using the model's encode() method
  - Encode the JD query text to a single query vector
  - L2-normalise all vectors so cosine similarity == dot product (required by FAISS IndexFlatIP)
  - Persist the full candidate embedding matrix and corresponding ID list to disk
  - Load a persisted embedding matrix back into memory
  - Support GPU when available, fall back to CPU transparently

Performance constraints:
  - 100K candidates × 384 dims × float32 = ~146 MB in RAM (well within 16 GB)
  - Batch inference: model.encode() on 512-512 texts per call avoids per-sample overhead
  - GPU is auto-detected at initialisation — precompute on GPU, rank on CPU
  - Embeddings are persisted ONCE (precompute.py) and loaded at rank time (rank.py)
    so the 5-minute CPU ranking window is never spent re-encoding 100K texts

Model choice: BAAI/bge-small-en-v1.5
  - 33M parameters, 384-dim output
  - Top MTEB score-per-second ratio among sub-100M models
  - Produces unit-norm vectors — dot product equals cosine similarity
  - Runs at ~6K texts/sec on CPU (100K texts in ~17 seconds)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
DEFAULT_BATCH_SIZE: int = 512
EMBEDDINGS_FILENAME: str = "embeddings.npy"
IDS_FILENAME: str = "candidate_ids.npy"
META_FILENAME: str = "embedding_meta.json"

# BGE models require this instruction prefix for retrieval tasks
_BGE_QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------------------
# Embedding Engine
# ---------------------------------------------------------------------------


class EmbeddingEngine:
    """
    Wraps a SentenceTransformer model for batch text encoding.

    Lifecycle:
        engine = EmbeddingEngine()             # lazy — model not loaded yet
        engine.load_model()                    # explicit load (or auto-loads on first encode call)
        vecs = engine.encode_texts(texts)      # (N, 384) float32, L2-normalised
        engine.save(ids, vecs, artifacts_dir)  # persist to disk
        ids2, vecs2 = EmbeddingEngine.load(artifacts_dir)  # restore

    The engine is stateless after loading the model — it holds no mutable
    per-candidate state, making it safe to use from multiple threads.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,
        show_progress_bar: bool = True,
    ) -> None:
        """
        Args:
            model_name:        HuggingFace model identifier.
            batch_size:        Number of texts per model.encode() call.
            device:            'cuda', 'cpu', or None (auto-detect).
            show_progress_bar: Forward to model.encode() for long runs.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device  # None → auto-detected on load
        self.show_progress_bar = show_progress_bar
        self._model: Any = None  # SentenceTransformer, loaded lazily

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load the SentenceTransformer model into memory.

        Auto-detects GPU availability. Logs the effective device.
        Idempotent — safe to call multiple times.
        """
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        effective_device = self._resolve_device()
        logger.info(f"Loading embedding model: {self.model_name!r} on device={effective_device!r}")
        t0 = time.perf_counter()
        self._model = SentenceTransformer(self.model_name, device=effective_device)
        elapsed = time.perf_counter() - t0
        logger.info(
            f"Model loaded in {elapsed:.1f}s — "
            f"embedding dim={self.embedding_dim}, device={effective_device!r}"
        )

    def _resolve_device(self) -> str:
        """Return the effective torch device string."""
        if self.device is not None:
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                logger.info("GPU detected — using CUDA for embedding.")
                return "cuda"
        except ImportError:
            pass
        logger.info("No GPU detected — using CPU for embedding.")
        return "cpu"

    def _ensure_loaded(self) -> None:
        """Auto-load the model if not yet loaded."""
        if self._model is None:
            self.load_model()

    @property
    def embedding_dim(self) -> int:
        """Dimension of the output vectors (384 for bge-small-en-v1.5)."""
        self._ensure_loaded()
        return self._model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_texts(
        self,
        texts: list[str],
        *,
        is_query: bool = False,
        normalize: bool = True,
    ) -> NDArray[np.float32]:
        """
        Encode a list of texts into a 2-D float32 numpy array.

        Args:
            texts:     List of strings to encode.
            is_query:  If True, prepend the BGE retrieval instruction prefix.
                       Use True for the JD query; False for candidate corpus.
            normalize: If True, L2-normalise each vector so dot product == cosine sim.

        Returns:
            NDArray of shape (len(texts), embedding_dim), dtype float32.

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("encode_texts() received an empty list.")
        self._ensure_loaded()

        # BGE query instruction prefix
        if is_query:
            texts = [f"{_BGE_QUERY_PREFIX}{t}" for t in texts]

        logger.info(
            f"Encoding {len(texts):,} texts (is_query={is_query}, "
            f"batch_size={self.batch_size}, normalize={normalize})"
        )
        t0 = time.perf_counter()

        embeddings: NDArray[np.float32] = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

        elapsed = time.perf_counter() - t0
        throughput = len(texts) / elapsed if elapsed > 0 else 0.0
        logger.info(
            f"Encoding complete: {len(texts):,} texts in {elapsed:.1f}s "
            f"({throughput:,.0f} texts/sec) — shape={embeddings.shape}"
        )
        return embeddings.astype(np.float32)

    def encode_query(self, text: str) -> NDArray[np.float32]:
        """
        Encode a single query string (the JD).

        Returns:
            NDArray of shape (1, embedding_dim), dtype float32, L2-normalised.
        """
        return self.encode_texts([text], is_query=True, normalize=True)

    def encode_corpus_batched(
        self,
        texts: list[str],
        *,
        log_every: int = 5000,
    ) -> NDArray[np.float32]:
        """
        Encode a large corpus in streaming batches with periodic progress logs.

        Identical to encode_texts() but logs progress every ``log_every`` texts.
        Use this for the 100K candidate encode during precompute.

        Args:
            texts:     All corpus texts.
            log_every: Log a progress line after every this many texts.

        Returns:
            NDArray of shape (len(texts), embedding_dim), dtype float32.
        """
        if not texts:
            raise ValueError("encode_corpus_batched() received an empty list.")
        self._ensure_loaded()

        all_embeddings: list[NDArray[np.float32]] = []
        total = len(texts)
        processed = 0
        t0 = time.perf_counter()

        for start in range(0, total, self.batch_size):
            chunk = texts[start : start + self.batch_size]
            batch_emb = self._model.encode(
                chunk,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            all_embeddings.append(batch_emb.astype(np.float32))
            processed += len(chunk)
            if processed % log_every < self.batch_size or processed >= total:
                elapsed = time.perf_counter() - t0
                pct = 100.0 * processed / total
                throughput = processed / elapsed if elapsed > 0 else 0.0
                logger.info(
                    f"  Corpus encoding: {processed:,}/{total:,} ({pct:.1f}%) "
                    f"— {throughput:,.0f} texts/sec"
                )

        matrix = np.vstack(all_embeddings)
        total_elapsed = time.perf_counter() - t0
        logger.info(
            f"Corpus encoding complete: {total:,} texts in {total_elapsed:.1f}s "
            f"— matrix shape={matrix.shape}, size={matrix.nbytes / 1e6:.1f} MB"
        )
        return matrix

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save(
        candidate_ids: list[str],
        embeddings: NDArray[np.float32],
        artifacts_dir: Path | str,
    ) -> None:
        """
        Persist the embedding matrix and ID list to ``artifacts_dir``.

        Three files are written:
          - ``embeddings.npy``       — (N, D) float32 matrix
          - ``candidate_ids.npy``    — (N,) array of unicode strings
          - ``embedding_meta.json``  — metadata (model, dim, count, timestamp)

        Args:
            candidate_ids: Parallel list of CAND_XXXXXXX strings.
            embeddings:    Float32 matrix of shape (N, D).
            artifacts_dir: Output directory (created if absent).

        Raises:
            ValueError: If candidate_ids and embeddings lengths differ.
        """
        if len(candidate_ids) != len(embeddings):
            raise ValueError(
                f"candidate_ids length ({len(candidate_ids)}) != "
                f"embeddings length ({len(embeddings)})"
            )

        out = Path(artifacts_dir)
        out.mkdir(parents=True, exist_ok=True)

        emb_path = out / EMBEDDINGS_FILENAME
        ids_path = out / IDS_FILENAME
        meta_path = out / META_FILENAME

        np.save(emb_path, embeddings)
        np.save(ids_path, np.array(candidate_ids, dtype=object))

        import datetime as dt
        import json

        meta = {
            "model_name": DEFAULT_MODEL_NAME,
            "embedding_dim": int(embeddings.shape[1]),
            "num_candidates": int(embeddings.shape[0]),
            "saved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "dtype": str(embeddings.dtype),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        size_mb = embeddings.nbytes / 1e6
        logger.info(
            f"Embeddings saved: {len(candidate_ids):,} vectors, "
            f"dim={embeddings.shape[1]}, {size_mb:.1f} MB → {out}"
        )

    @staticmethod
    def load(
        artifacts_dir: Path | str,
    ) -> tuple[list[str], NDArray[np.float32]]:
        """
        Load the persisted embedding matrix and ID list from ``artifacts_dir``.

        Returns:
            Tuple of (candidate_ids, embeddings):
              - candidate_ids: list of CAND_XXXXXXX strings, length N
              - embeddings:    NDArray float32 of shape (N, D), L2-normalised

        Raises:
            FileNotFoundError: If the expected files do not exist in artifacts_dir.
        """
        d = Path(artifacts_dir)
        emb_path = d / EMBEDDINGS_FILENAME
        ids_path = d / IDS_FILENAME

        for p in (emb_path, ids_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Embedding artifact not found: {p}. "
                    "Run precompute.py first to generate embeddings."
                )

        t0 = time.perf_counter()
        embeddings: NDArray[np.float32] = np.load(emb_path).astype(np.float32)
        raw_ids: NDArray = np.load(ids_path, allow_pickle=True)
        candidate_ids: list[str] = [str(x) for x in raw_ids.tolist()]
        elapsed = time.perf_counter() - t0

        logger.info(
            f"Embeddings loaded: {len(candidate_ids):,} vectors, "
            f"shape={embeddings.shape}, {elapsed:.2f}s ← {d}"
        )
        return candidate_ids, embeddings

    @staticmethod
    def artifacts_exist(artifacts_dir: Path | str) -> bool:
        """Return True only if both required artifact files exist."""
        d = Path(artifacts_dir)
        return (d / EMBEDDINGS_FILENAME).exists() and (d / IDS_FILENAME).exists()

    # ------------------------------------------------------------------
    # Similarity utilities
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity_to_query(
        query_vec: NDArray[np.float32],
        corpus_vecs: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Compute cosine similarity between one query and a corpus.

        Because both vectors are L2-normalised, this is a simple dot product.

        Args:
            query_vec:   Shape (1, D) or (D,) — the JD query vector.
            corpus_vecs: Shape (N, D) — the candidate corpus embeddings.

        Returns:
            NDArray of shape (N,) with similarity scores in [0, 1].
        """
        q = query_vec.reshape(1, -1)
        # dot product of unit vectors = cosine similarity
        sims: NDArray[np.float32] = (corpus_vecs @ q.T).squeeze(axis=1)
        # Clamp to [0, 1] to handle floating-point edge cases
        return np.clip(sims, 0.0, 1.0).astype(np.float32)
