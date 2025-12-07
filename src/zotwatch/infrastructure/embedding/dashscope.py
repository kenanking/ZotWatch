"""DashScope (Alibaba Cloud) embedding and reranking providers."""

import logging
from http import HTTPStatus
from typing import Iterable

import numpy as np
from dashscope import TextEmbedding, TextReRank

from zotwatch.core.constants import DASHSCOPE_EMBEDDING_DIM
from zotwatch.core.exceptions import ConfigurationError

from .base import BaseEmbeddingProvider, BaseReranker

logger = logging.getLogger(__name__)


class DashScopeEmbedding(BaseEmbeddingProvider):
    """Alibaba Cloud DashScope text embedding provider.

    Uses the text-embedding-v4 model by default with configurable dimension.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-v4",
        api_key: str = "",
        dimension: int = DASHSCOPE_EMBEDDING_DIM,
        batch_size: int = 25,  # DashScope recommends smaller batches
    ):
        """Initialize DashScope embedding provider.

        Args:
            model_name: Model identifier (default: text-embedding-v4).
            api_key: DashScope API key (DASHSCOPE_API_KEY).
            dimension: Embedding dimension (default: 1024).
            batch_size: Number of texts per batch (default: 25).
        """
        self._model_name = model_name
        self._api_key = api_key
        self._dimension = dimension
        self.batch_size = batch_size

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimension

    def _ensure_api_key(self) -> str:
        """Ensure API key is configured."""
        if not self._api_key:
            raise ConfigurationError("DashScope API key is required. Set DASHSCOPE_API_KEY environment variable.")
        return self._api_key

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: Iterable of text strings to encode.

        Returns:
            numpy array of shape (n_texts, dimension) with L2-normalized embeddings.
        """
        api_key = self._ensure_api_key()
        # Convert to list and handle empty strings
        texts = [t.strip() if t and t.strip() else "[untitled]" for t in texts]
        total = len(texts)
        num_batches = (total + self.batch_size - 1) // self.batch_size
        logger.info("Encoding %d texts with %s (%d batches)", total, self._model_name, num_batches)

        all_embeddings = []
        for batch_idx, i in enumerate(range(0, total, self.batch_size)):
            batch = texts[i : i + self.batch_size]
            logger.info("  Batch %d/%d: encoding %d texts...", batch_idx + 1, num_batches, len(batch))

            resp = TextEmbedding.call(
                model=self._model_name,
                input=batch,
                dimension=self._dimension,
                api_key=api_key,
            )

            if resp.status_code != HTTPStatus.OK:
                raise RuntimeError(f"DashScope embedding failed: {resp.code} - {resp.message}")

            # Extract embeddings from response
            for item in resp.output["embeddings"]:
                all_embeddings.append(item["embedding"])

        embeddings = np.asarray(all_embeddings, dtype=np.float32)
        # L2 normalization for FAISS IndexFlatIP (inner product = cosine similarity)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        return embeddings / norms


class DashScopeReranker(BaseReranker):
    """DashScope Reranker service for semantic re-ranking of documents.

    Uses qwen3-rerank model by default.
    """

    max_documents = 500  # DashScope recommended limit

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-rerank",
        instruct: str | None = None,
    ):
        """Initialize DashScope Reranker.

        Args:
            api_key: DashScope API key.
            model: Rerank model name (default: qwen3-rerank).
            instruct: Optional task instruction for qwen3-rerank model.
        """
        self._api_key = api_key
        self.model = model
        self.instruct = instruct

    def _ensure_api_key(self) -> str:
        """Ensure API key is configured."""
        if not self._api_key:
            raise ConfigurationError("DashScope API key is required. Set DASHSCOPE_API_KEY environment variable.")
        return self._api_key

    def _rerank_batch(
        self,
        query: str,
        documents: list[str],
        top_k: int,
    ) -> list[tuple[int, float]]:
        """DashScope-specific single-batch reranking."""
        api_key = self._ensure_api_key()

        resp = TextReRank.call(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_k,
            return_documents=False,
            api_key=api_key,
        )

        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(f"DashScope rerank failed: {resp.code} - {resp.message}")

        results = resp.output["results"]
        return [(r["index"], r["relevance_score"]) for r in results]


__all__ = ["DashScopeEmbedding", "DashScopeReranker"]
