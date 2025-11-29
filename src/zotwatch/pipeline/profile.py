"""Profile building pipeline."""

import logging
from pathlib import Path


from zotwatch.config.settings import Settings
from zotwatch.core.models import ProfileArtifacts
from zotwatch.infrastructure.embedding import (
    CachingEmbeddingProvider,
    EmbeddingCache,
    FaissIndex,
    VoyageEmbedding,
)
from zotwatch.infrastructure.embedding.base import BaseEmbeddingProvider
from zotwatch.infrastructure.storage import ProfileStorage

logger = logging.getLogger(__name__)


class ProfileBuilder:
    """Builds user research profile from library."""

    def __init__(
        self,
        base_dir: Path | str,
        storage: ProfileStorage,
        settings: Settings,
        vectorizer: BaseEmbeddingProvider | None = None,
        embedding_cache: EmbeddingCache | None = None,
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
        )

    def run(self, *, full: bool = False) -> ProfileArtifacts:
        """Build profile from library items.

        Args:
            full: If True, invalidate all profile embeddings and recompute.
                  If False (default), use cached embeddings where available.
        """
        all_items = list(self.storage.iter_items())
        if not all_items:
            raise RuntimeError("No items found in storage; run ingest before building profile.")

        # Filter items: only include those with abstracts
        items_with_abstract = [item for item in all_items if item.abstract]
        items_without_abstract = len(all_items) - len(items_with_abstract)

        logger.info(
            "Library statistics: %d items with abstract, %d items without abstract",
            len(items_with_abstract),
            items_without_abstract,
        )

        if not items_with_abstract:
            raise RuntimeError(
                "No items with abstracts found in storage; profile building requires items with abstracts."
            )

        items = items_with_abstract
        logger.info("Building profile from %d items (with abstracts)", len(items))

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

        return self.artifacts


__all__ = ["ProfileBuilder"]
