"""FAISS vector index."""

import logging
from pathlib import Path

import faiss
import numpy as np

from zotwatch.core.exceptions import ConfigurationError, ValidationError

logger = logging.getLogger(__name__)


class FaissIndex:
    """FAISS vector index for similarity search."""

    def __init__(self, dim: int, index: faiss.Index | None = None) -> None:
        if faiss is None:
            raise ConfigurationError("faiss is required; install faiss-cpu or adjust configuration.")
        self.dim = dim
        self.index = index or faiss.IndexFlatIP(dim)

    @classmethod
    def from_vectors(cls, vectors: np.ndarray) -> tuple["FaissIndex", np.ndarray]:
        """Create index from vector array.

        Vectors are L2-normalized before adding to the index so that
        IndexFlatIP (inner product) produces cosine similarity scores.
        """
        if vectors.ndim != 2:
            raise ValidationError(f"Vectors must be a 2D array, got {vectors.ndim}D")
        dim = vectors.shape[1]
        instance = cls(dim)
        # L2-normalize so inner product equals cosine similarity
        vectors = vectors.astype("float32")
        faiss.normalize_L2(vectors)
        instance.index.add(vectors)
        return instance, np.arange(vectors.shape[0])

    def save(self, path: Path | str) -> None:
        """Save index to disk."""
        logger.info("Saving FAISS index to %s", path)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path | str) -> "FaissIndex":
        """Load index from disk."""
        if faiss is None:
            raise ConfigurationError("faiss is required; install faiss-cpu or adjust configuration.")
        index = faiss.read_index(str(path))
        if index.ntotal == 0:
            raise ValidationError(f"Loaded FAISS index from {path} is empty")
        return cls(index.d, index)

    @property
    def ntotal(self) -> int:
        """Expose total vector count for compatibility with faiss.Index."""
        return int(getattr(self.index, "ntotal", 0))

    def search(self, vectors: np.ndarray, top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors.

        Returns empty arrays if the index is empty.
        """
        if self.ntotal == 0:
            n = 1 if vectors.ndim == 1 else vectors.shape[0]
            return np.zeros((n, top_k), dtype=np.float32), -np.ones((n, top_k), dtype=np.int64)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        vectors = vectors.astype("float32")
        faiss.normalize_L2(vectors)
        return self.index.search(vectors, top_k)


__all__ = ["FaissIndex"]
