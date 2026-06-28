"""
src.embeddings — Public API for the embedding engine.
"""

from src.embeddings.engine import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_NAME,
    EMBEDDINGS_FILENAME,
    IDS_FILENAME,
    META_FILENAME,
    EmbeddingEngine,
)

__all__ = [
    "EmbeddingEngine",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_BATCH_SIZE",
    "EMBEDDINGS_FILENAME",
    "IDS_FILENAME",
    "META_FILENAME",
]
