"""Profile building pipeline."""

import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from zotwatch.core.models import ProfileArtifacts, ZoteroItem
from zotwatch.infrastructure.embedding import (
    CachingEmbeddingProvider,
    EmbeddingCache,
    FaissIndex,
    VoyageEmbedding,
)
from zotwatch.infrastructure.storage import ProfileStorage
from zotwatch.utils.datetime import utc_now
from zotwatch.utils.text import json_dumps

if TYPE_CHECKING:
    from zotwatch.config.settings import Settings
    from zotwatch.infrastructure.embedding.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class ProfileBuilder:
    """Builds user research profile from library."""

    def __init__(
        self,
        base_dir: Path | str,
        storage: ProfileStorage,
        settings: Settings,
        vectorizer: Optional[BaseEmbeddingProvider] = None,
        embedding_cache: Optional[EmbeddingCache] = None,
    ):
        """Initialize profile builder.

        Args:
            base_dir: Base directory for data files.
            storage: Profile storage for items.
            settings: Application settings.
            vectorizer: Optional base embedding provider (defaults to VoyageEmbedding).
            embedding_cache: Optional embedding cache. If provided, wraps vectorizer
                            with CachingEmbeddingProvider for profile source type.
        """
        self.base_dir = Path(base_dir)
        self.storage = storage
        self.settings = settings

        # Create base vectorizer
        base_vectorizer = vectorizer or VoyageEmbedding(
            model_name=settings.embedding.model,
            api_key=settings.embedding.api_key,
            input_type=settings.embedding.input_type,
            batch_size=settings.embedding.batch_size,
        )

        # Wrap with caching if cache is provided
        if embedding_cache is not None:
            self.vectorizer: BaseEmbeddingProvider = CachingEmbeddingProvider(
                provider=base_vectorizer,
                cache=embedding_cache,
                source_type="profile",
                ttl_days=None,  # Profile embeddings never expire
            )
            self._cache = embedding_cache
        else:
            self.vectorizer = base_vectorizer
            self._cache = None

        self.artifacts = ProfileArtifacts(
            sqlite_path=str(self.base_dir / "data" / "profile.sqlite"),
            faiss_path=str(self.base_dir / "data" / "faiss.index"),
            profile_json_path=str(self.base_dir / "data" / "profile.json"),
        )

    def run(self, *, full: bool = False) -> ProfileArtifacts:
        """Build profile from library items.

        Args:
            full: If True, invalidate all profile embeddings and recompute.
                  If False (default), use cached embeddings where available.
        """
        items = list(self.storage.iter_items())
        if not items:
            raise RuntimeError("No items found in storage; run ingest before building profile.")

        logger.info("Building profile from %d library items", len(items))

        # If full rebuild requested and cache is available, invalidate profile embeddings
        if full and self._cache is not None:
            invalidated = self._cache.invalidate_source("profile")
            if invalidated > 0:
                logger.info("Invalidated %d cached profile embeddings for full rebuild", invalidated)

        # Encode all items (caching handled automatically by CachingEmbeddingProvider)
        texts = [item.content_for_embedding() for item in items]
        source_ids = [item.key for item in items]

        # Use encode_with_ids if available (for source tracking)
        if isinstance(self.vectorizer, CachingEmbeddingProvider):
            vectors = self.vectorizer.encode_with_ids(texts, source_ids=source_ids)
        else:
            vectors = self.vectorizer.encode(texts)

        logger.info("Computed embeddings for %d items", len(items))

        # Build FAISS index
        logger.info("Building FAISS index")
        index, _ = FaissIndex.from_vectors(vectors)
        index.save(self.artifacts.faiss_path)

        # Generate profile summary
        profile_summary = self._summarize(items, vectors)
        json_path = Path(self.artifacts.profile_json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json_dumps(profile_summary, indent=2), encoding="utf-8")
        logger.info("Wrote profile summary to %s", json_path)

        return self.artifacts

    def _summarize(self, items: List[ZoteroItem], vectors: np.ndarray) -> dict:
        """Generate profile summary."""
        authors = Counter()
        venues = Counter()
        for item in items:
            authors.update(item.creators)
            venue = item.raw.get("data", {}).get("publicationTitle")
            if venue:
                venues.update([venue])

        # Compute centroid
        centroid = np.mean(vectors, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        top_authors = [{"author": k, "count": v} for k, v in authors.most_common(20)]
        top_venues = [{"venue": k, "count": v} for k, v in venues.most_common(20)]

        return {
            "generated_at": utc_now().isoformat(),
            "item_count": len(items),
            "model": self.vectorizer.model_name,
            "centroid": centroid.tolist(),
            "top_authors": top_authors,
            "top_venues": top_venues,
        }


__all__ = ["ProfileBuilder"]
