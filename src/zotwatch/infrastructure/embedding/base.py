"""Base classes for embedding and reranking providers."""

import logging
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Embedding dimensionality."""
        ...

    @abstractmethod
    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        ...

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text."""
        return self.encode([text])[0]

    def encode_query(self, texts: Iterable[str]) -> np.ndarray:
        """Encode query texts for retrieval.

        Override this method to use query-specific encoding (e.g., Voyage input_type="query").
        Default implementation calls encode() for backward compatibility.

        Args:
            texts: Iterable of query text strings to encode.

        Returns:
            numpy array of shape (n_texts, dimensions).
        """
        return self.encode(texts)


class BaseReranker(ABC):
    """Abstract base class for reranking providers.

    Subclasses must set the max_documents class attribute to the API limit.
    """

    max_documents: int  # Must be set in subclass (Voyage: 1000, DashScope: 500)

    @abstractmethod
    def _rerank_batch(
        self,
        query: str,
        documents: list[str],
        top_k: int,
    ) -> list[tuple[int, float]]:
        """Provider-specific reranking implementation.

        Args:
            query: Search query.
            documents: List of document texts (guaranteed <= max_documents).
            top_k: Number of top results to return.

        Returns:
            List of (index, relevance_score) tuples, sorted by score descending.
        """
        ...

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Rerank documents by relevance to query.

        Args:
            query: Search query.
            documents: List of document texts (must not exceed max_documents).
            top_k: Number of top results to return.

        Returns:
            List of (original_index, relevance_score) tuples, sorted by score descending.

        Raises:
            ValueError: If documents exceed max_documents limit.
        """
        if not documents:
            return []

        total = len(documents)

        # Validate document count against API limit
        if total > self.max_documents:
            raise ValueError(
                f"Document count ({total}) exceeds reranker limit ({self.max_documents}). "
                f"Use embedding similarity to pre-filter candidates before reranking."
            )

        top_k = min(top_k, total)
        logger.info("Reranking %d documents with query (top_k=%d)", total, top_k)

        try:
            results = self._rerank_batch(query, documents, top_k)
            logger.info(
                "Reranking complete: %d results, top score=%.4f",
                len(results),
                results[0][1] if results else 0.0,
            )
            return results
        except Exception as e:
            logger.error("Reranking failed: %s", e)
            raise


__all__ = ["BaseEmbeddingProvider", "BaseReranker"]
