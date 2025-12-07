"""Caching embedding provider with unified SQLite backend."""

import logging
from typing import Iterable

import numpy as np

from zotwatch.core.exceptions import ValidationError
from zotwatch.infrastructure.embedding.base import BaseEmbeddingProvider
from zotwatch.infrastructure.embedding.cache import EmbeddingCache
from zotwatch.utils.hashing import hash_content

logger = logging.getLogger(__name__)


class CachingEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider with caching support.

    Wraps any BaseEmbeddingProvider and adds SQLite-backed caching.
    Uses (content_hash, model) as composite key for automatic
    invalidation when switching embedding models.
    """

    def __init__(
        self,
        provider: BaseEmbeddingProvider,
        cache: EmbeddingCache,
        source_type: str = "generic",
        ttl_days: int | None = None,
    ):
        """Initialize caching embedding provider.

        Args:
            provider: Base embedding provider (e.g., VoyageEmbedding).
            cache: Unified embedding cache storage.
            source_type: Type identifier for cached embeddings ("profile" or "candidate").
            ttl_days: Time-to-live in days. None for permanent storage.
        """
        self.provider = provider
        self.cache = cache
        self.source_type = source_type
        self.ttl_days = ttl_days
        self._stats = {"hits": 0, "misses": 0}

    @property
    def model_name(self) -> str:
        """Model identifier from underlying provider."""
        return self.provider.model_name

    @property
    def dimensions(self) -> int:
        """Embedding dimensionality from underlying provider."""
        return self.provider.dimensions

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """Encode texts with caching.

        Args:
            texts: Iterable of text strings to encode.

        Returns:
            numpy array of shape (n_texts, dimensions).
        """
        texts_list = list(texts)
        if not texts_list:
            return np.array([], dtype=np.float32).reshape(0, self.dimensions)

        # Compute content hashes
        hashes = [hash_content(t) for t in texts_list]

        # Batch query cache
        cached = self.cache.get_batch(hashes, self.model_name)

        # Separate hits and misses
        results: list[np.ndarray | None] = [None] * len(texts_list)
        to_encode_idx: list[int] = []
        to_encode_texts: list[str] = []

        for i, h in enumerate(hashes):
            if h in cached:
                results[i] = np.frombuffer(cached[h], dtype=np.float32).copy()
                self._stats["hits"] += 1
            else:
                to_encode_idx.append(i)
                to_encode_texts.append(texts_list[i])
                self._stats["misses"] += 1

        # Encode cache misses
        if to_encode_texts:
            logger.info(
                "Encoding %d new texts (cache hits: %d)",
                len(to_encode_texts),
                len(texts_list) - len(to_encode_texts),
            )
            new_vectors = self.provider.encode(to_encode_texts)

            # Prepare cache entries
            new_cache_items: list[tuple[str, bytes]] = []
            for idx, vec in zip(to_encode_idx, new_vectors):
                results[idx] = vec
                new_cache_items.append((hashes[idx], vec.tobytes()))

            # Batch save to cache
            self.cache.put_batch(
                new_cache_items,
                model=self.model_name,
                source_type=self.source_type,
                ttl_days=self.ttl_days,
            )
        else:
            logger.info("All %d texts found in cache", len(texts_list))

        # Log cache statistics
        self._log_stats()

        return np.stack(results)

    def encode_query(self, texts: Iterable[str]) -> np.ndarray:
        """Encode query texts (bypasses cache, uses query-specific encoding).

        Query embeddings are not cached because:
        1. Typically only one query per run
        2. LLM-refined queries may vary between runs

        Args:
            texts: Iterable of query text strings to encode.

        Returns:
            numpy array of shape (n_texts, dimensions).
        """
        return self.provider.encode_query(texts)

    def encode_with_ids(
        self,
        texts: Iterable[str],
        source_ids: list[str] | None = None,
    ) -> np.ndarray:
        """Encode texts with optional source ID tracking.

        Args:
            texts: Iterable of text strings to encode.
            source_ids: Optional list of source identifiers for each text.

        Returns:
            numpy array of shape (n_texts, dimensions).
        """
        texts_list = list(texts)
        if not texts_list:
            return np.array([], dtype=np.float32).reshape(0, self.dimensions)

        if source_ids is not None and len(source_ids) != len(texts_list):
            raise ValidationError(f"source_ids length ({len(source_ids)}) must match texts length ({len(texts_list)})")

        # Compute content hashes
        hashes = [hash_content(t) for t in texts_list]

        # Batch query cache
        cached = self.cache.get_batch(hashes, self.model_name)

        # Separate hits and misses
        results: list[np.ndarray | None] = [None] * len(texts_list)
        to_encode_idx: list[int] = []
        to_encode_texts: list[str] = []

        for i, h in enumerate(hashes):
            if h in cached:
                results[i] = np.frombuffer(cached[h], dtype=np.float32).copy()
                self._stats["hits"] += 1
            else:
                to_encode_idx.append(i)
                to_encode_texts.append(texts_list[i])
                self._stats["misses"] += 1

        # Encode cache misses
        if to_encode_texts:
            logger.info(
                "Encoding %d new texts (cache hits: %d)",
                len(to_encode_texts),
                len(texts_list) - len(to_encode_texts),
            )
            new_vectors = self.provider.encode(to_encode_texts)

            # Prepare cache entries with source IDs
            new_cache_items: list[tuple[str, bytes]] = []
            new_source_ids: list[str] | None = None
            if source_ids is not None:
                new_source_ids = []

            for idx, vec in zip(to_encode_idx, new_vectors):
                results[idx] = vec
                new_cache_items.append((hashes[idx], vec.tobytes()))
                if new_source_ids is not None and source_ids is not None:
                    new_source_ids.append(source_ids[idx])

            # Batch save to cache
            self.cache.put_batch(
                new_cache_items,
                model=self.model_name,
                source_type=self.source_type,
                source_ids=new_source_ids,
                ttl_days=self.ttl_days,
            )
        else:
            logger.info("All %d texts found in cache", len(texts_list))

        # Log cache statistics
        self._log_stats()

        return np.stack(results)

    def _log_stats(self) -> None:
        """Log cache hit/miss statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        if total > 0:
            hit_rate = 100 * self._stats["hits"] / total
            logger.info(
                "Embedding cache stats: %d hits, %d misses (%.1f%% hit rate)",
                self._stats["hits"],
                self._stats["misses"],
                hit_rate,
            )

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._stats = {"hits": 0, "misses": 0}

    @property
    def stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dict with 'hits' and 'misses' counts.
        """
        return self._stats.copy()


__all__ = ["CachingEmbeddingProvider"]
