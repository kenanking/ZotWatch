"""Profile building pipeline."""

import logging
from pathlib import Path

import numpy as np

from zotwatch.config.settings import Settings
from zotwatch.core.exceptions import ProfileBuildError
from zotwatch.core.models import ClusteredProfile, ProfileArtifacts, ZoteroItem
from zotwatch.infrastructure.embedding import (
    CachingEmbeddingProvider,
    EmbeddingCache,
    FaissIndex,
    create_embedding_provider,
)
from zotwatch.infrastructure.embedding.base import BaseEmbeddingProvider
from zotwatch.infrastructure.storage import ProfileStorage
from zotwatch.utils.temporal import compute_batch_weights

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
        base_vectorizer = vectorizer or create_embedding_provider(settings.embedding)

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
        total_items = self.storage.count_items()
        items = self.storage.get_items_with_abstract()

        if total_items == 0:
            raise ProfileBuildError("No items found in storage; run ingest before building profile.")

        items_without_abstract = total_items - len(items)

        logger.info(
            "Library statistics: %d items with abstract, %d items without abstract",
            len(items),
            items_without_abstract,
        )

        if not items:
            raise ProfileBuildError(
                "No items with abstracts found in storage; profile building requires items with abstracts."
            )

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

        # Persist embedding signature to detect provider/model changes across runs
        signature = self.settings.embedding.signature
        self.storage.set_metadata("embedding_signature", signature)

        # Run clustering if enabled
        if self.settings.profile.clustering.enabled:
            self._run_clustering(vectors, items, signature)

        return self.artifacts

    def _run_clustering(
        self,
        vectors: np.ndarray,
        items: list[ZoteroItem],
        embedding_signature: str,
    ) -> ClusteredProfile | None:
        """Run k-means clustering on profile embeddings.

        Args:
            vectors: Embedding matrix (N x dim).
            items: Corresponding ZoteroItem list.
            embedding_signature: Embedding provider and model signature.

        Returns:
            ClusteredProfile if clustering was successful, None otherwise.
        """
        from zotwatch.pipeline.profile_clusterer import ProfileClusterer

        clusterer = ProfileClusterer(
            config=self.settings.profile.clustering,
            embedding_signature=embedding_signature,
        )

        # Compute temporal weights if enabled
        temporal_weights = None
        temporal_config = self.settings.profile.clustering.temporal
        if temporal_config.enabled:
            weights_list = compute_batch_weights(
                items,
                halflife_days=temporal_config.halflife_days,
                min_weight=temporal_config.min_weight,
            )
            temporal_weights = np.array(weights_list, dtype=np.float32)
            logger.info(
                "Computed temporal weights for %d items (halflife=%.0f days, mean_weight=%.3f)",
                len(items),
                temporal_config.halflife_days,
                np.mean(temporal_weights),
            )

        clustered_profile = clusterer.cluster(vectors.copy(), items, temporal_weights=temporal_weights)

        if clustered_profile.valid_cluster_count == 0:
            logger.info("No valid clusters created (library may be too small)")
            return None

        # Save to storage
        self.storage.save_clustered_profile(clustered_profile)

        logger.info(
            "Created %d valid clusters covering %d/%d papers (total_effective_size=%.2f)",
            clustered_profile.valid_cluster_count,
            clustered_profile.papers_in_valid_clusters,
            clustered_profile.total_papers,
            clustered_profile.total_effective_size,
        )

        return clustered_profile


__all__ = ["ProfileBuilder"]
