"""Voyage AI embedding and reranking providers."""

import logging
from typing import Iterable

import numpy as np
import voyageai

from zotwatch.core.constants import VOYAGE_EMBEDDING_DIM
from zotwatch.core.exceptions import ConfigurationError

from .base import BaseEmbeddingProvider, BaseReranker

logger = logging.getLogger(__name__)


class VoyageEmbedding(BaseEmbeddingProvider):
    """Voyage AI text embedding provider."""

    def __init__(
        self,
        model_name: str = "voyage-3.5",
        api_key: str = "",
        batch_size: int = 128,
    ):
        self._model_name = model_name
        self._api_key = api_key
        self.batch_size = batch_size
        self._client = None
        self._dimensions = VOYAGE_EMBEDDING_DIM

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _get_client(self) -> voyageai.Client:
        """Get or create Voyage AI client."""
        if self._client is None:
            if not self._api_key:
                raise ConfigurationError("Voyage API key is required. Set VOYAGE_API_KEY environment variable.")
            self._client = voyageai.Client(api_key=self._api_key)
        return self._client

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        client = self._get_client()
        # Replace empty strings with placeholder (Voyage API rejects empty input)
        texts = [t.strip() if t and t.strip() else "[untitled]" for t in texts]
        total = len(texts)
        num_batches = (total + self.batch_size - 1) // self.batch_size
        logger.info("Encoding %d texts with %s (%d batches)", total, self._model_name, num_batches)

        all_embeddings = []
        for batch_idx, i in enumerate(range(0, total, self.batch_size)):
            batch = texts[i : i + self.batch_size]
            logger.info("  Batch %d/%d: encoding %d texts...", batch_idx + 1, num_batches, len(batch))
            result = client.embed(
                batch,
                model=self._model_name,
                input_type="document",
            )
            all_embeddings.extend(result.embeddings)

        embeddings = np.asarray(all_embeddings, dtype=np.float32)
        # L2 normalization for FAISS IndexFlatIP (inner product = cosine similarity)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        return embeddings / norms

    def encode_query(self, texts: Iterable[str]) -> np.ndarray:
        """Encode query texts using Voyage's query-specific encoding.

        Uses input_type="query" for better retrieval performance.
        See: https://docs.voyageai.com/docs/embeddings

        Args:
            texts: Iterable of query text strings to encode.

        Returns:
            numpy array of shape (n_texts, dimensions) with L2-normalized embeddings.
        """
        client = self._get_client()
        texts = [t.strip() if t and t.strip() else "[untitled]" for t in texts]
        total = len(texts)
        num_batches = (total + self.batch_size - 1) // self.batch_size
        logger.info("Encoding %d query texts with %s (%d batches)", total, self._model_name, num_batches)

        all_embeddings = []
        for batch_idx, i in enumerate(range(0, total, self.batch_size)):
            batch = texts[i : i + self.batch_size]
            logger.info("  Batch %d/%d: encoding %d query texts...", batch_idx + 1, num_batches, len(batch))
            result = client.embed(
                batch,
                model=self._model_name,
                input_type="query",
            )
            all_embeddings.extend(result.embeddings)

        embeddings = np.asarray(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        return embeddings / norms


class VoyageReranker(BaseReranker):
    """Voyage AI Reranker service for semantic re-ranking of documents."""

    max_documents = 1000  # Voyage API limit

    def __init__(self, api_key: str, model: str = "rerank-2"):
        """Initialize Voyage Reranker.

        Args:
            api_key: Voyage AI API key.
            model: Rerank model name (default: rerank-2).
        """
        if not api_key:
            raise ConfigurationError("Voyage API key is required. Set VOYAGE_API_KEY environment variable.")
        self._client = voyageai.Client(api_key=api_key)
        self.model = model

    def _rerank_batch(
        self,
        query: str,
        documents: list[str],
        top_k: int,
    ) -> list[tuple[int, float]]:
        """Voyage-specific single-batch reranking."""
        result = self._client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_k=top_k,
        )
        return [(r.index, r.relevance_score) for r in result.results]


__all__ = ["VoyageEmbedding", "VoyageReranker"]
