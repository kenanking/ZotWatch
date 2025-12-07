"""Profile clustering using FAISS k-means with adaptive Silhouette-based optimization."""

import logging
from collections import Counter

import faiss
import numpy as np
from sklearn.metrics import silhouette_score

from zotwatch.config.settings import ClusteringConfig
from zotwatch.core.models import ClusteredProfile, ClusterInfo, ZoteroItem
from zotwatch.utils.datetime import utc_now
from zotwatch.utils.temporal import compute_item_age_days

logger = logging.getLogger(__name__)

# FAISS recommends n_samples >= 39 * k to avoid warnings
FAISS_SAMPLES_PER_CLUSTER = 39


class ProfileClusterer:
    """Clusters user library papers into semantic groups using FAISS k-means.

    Uses adaptive Silhouette-based clustering with automatic k selection.
    Handles edge cases:
    - n=0: Returns empty profile (random recommendation will be used)
    - n=1: Treats entire library as single cluster
    - n>=2: Uses Silhouette score to find optimal k in [2, min(35, n/39)]
    """

    def __init__(
        self,
        config: ClusteringConfig,
        embedding_signature: str,
    ):
        """Initialize profile clusterer.

        Args:
            config: Clustering configuration.
            embedding_signature: Embedding provider and model signature
                                 (e.g., "voyage:voyage-3.5").
        """
        self.config = config
        self.embedding_signature = embedding_signature
        self._last_silhouette_score: float | None = None

    def cluster(
        self,
        vectors: np.ndarray,
        items: list[ZoteroItem],
        temporal_weights: np.ndarray | None = None,
    ) -> ClusteredProfile:
        """Run k-means clustering on profile embeddings.

        Args:
            vectors: Embedding matrix (N x dim), should be L2-normalized.
            items: Corresponding ZoteroItem list.
            temporal_weights: Optional temporal weights (N,) for each item.
                             If None and temporal.enabled, will compute from items.

        Returns:
            ClusteredProfile with cluster information.
        """
        n_samples = vectors.shape[0]

        # Edge case: Empty library
        if n_samples == 0:
            logger.info("Empty library, skipping clustering (will use random recommendation)")
            return ClusteredProfile(
                valid_cluster_count=0,
                total_papers=0,
                embedding_signature=self.embedding_signature,
                generated_at=utc_now(),
            )

        # Ensure vectors are float32 and normalized for spherical k-means
        vectors = vectors.astype(np.float32).copy()
        faiss.normalize_L2(vectors)

        # Handle temporal weights
        if temporal_weights is None:
            # Use uniform weights if temporal weighting is disabled
            temporal_weights = np.ones(n_samples, dtype=np.float32)
        else:
            temporal_weights = np.asarray(temporal_weights, dtype=np.float32)

        # Compute temporal config for storage
        temporal_config = None
        if self.config.temporal.enabled:
            temporal_config = {
                "enabled": self.config.temporal.enabled,
                "halflife_days": self.config.temporal.halflife_days,
                "min_weight": self.config.temporal.min_weight,
            }

        # Compute date range
        date_range = self._compute_date_range(items)

        # Edge case: Single paper - treat as one cluster
        if n_samples == 1:
            logger.info("Single paper library, treating as one cluster")
            return self._build_single_cluster_profile(vectors, items, temporal_weights, temporal_config, date_range)

        # Normal clustering flow (n >= 2)
        n_clusters = self._determine_cluster_count(n_samples, vectors)
        if self._last_silhouette_score is not None:
            logger.info(
                "Clustering %d papers into %d clusters (silhouette=%.4f)",
                n_samples,
                n_clusters,
                self._last_silhouette_score,
            )
        else:
            logger.info("Clustering %d papers into %d clusters", n_samples, n_clusters)

        # Special case: k=1 (very small library)
        if n_clusters == 1:
            return self._build_single_cluster_profile(vectors, items, temporal_weights, temporal_config, date_range)

        # Run FAISS k-means with spherical clustering (cosine similarity)
        dim = vectors.shape[1]
        kmeans = faiss.Kmeans(
            dim,
            n_clusters,
            niter=self.config.kmeans_iterations,
            verbose=False,
            gpu=False,
            spherical=True,  # Normalize centroids after each iteration
            seed=42,  # Reproducibility
        )
        kmeans.train(vectors)

        # Get cluster assignments
        _, assignments = kmeans.index.search(vectors, 1)
        assignments = assignments.flatten()

        # Build cluster info with temporal weighting
        clusters = self._build_cluster_info(
            kmeans.centroids,
            assignments,
            vectors,
            items,
            temporal_weights,
        )

        # Filter by min_cluster_size
        valid_clusters = [c for c in clusters if c.member_count >= self.config.min_cluster_size]

        # Re-assign cluster IDs after filtering
        for i, cluster in enumerate(valid_clusters):
            cluster.cluster_id = i

        # Compute total effective size
        total_effective_size = sum(c.effective_size for c in valid_clusters)

        logger.info(
            "Created %d valid clusters (≥%d papers) from %d total clusters, total_effective_size=%.2f",
            len(valid_clusters),
            self.config.min_cluster_size,
            len(clusters),
            total_effective_size,
        )

        return ClusteredProfile(
            clusters=valid_clusters,
            valid_cluster_count=len(valid_clusters),
            min_cluster_size_used=self.config.min_cluster_size,
            total_papers=n_samples,
            papers_in_valid_clusters=sum(c.member_count for c in valid_clusters),
            clustering_method="silhouette",
            embedding_signature=self.embedding_signature,
            generated_at=utc_now(),
            temporal_config_used=temporal_config,
            profile_date_range=date_range,
            total_effective_size=total_effective_size,
        )

    def _compute_date_range(self, items: list[ZoteroItem]) -> tuple[str, str] | None:
        """Compute min and max date_added from items."""
        dates = [item.date_added for item in items if item.date_added is not None]
        if not dates:
            return None
        min_date = min(dates)
        max_date = max(dates)
        return (min_date.isoformat(), max_date.isoformat())

    def _determine_cluster_count(self, n_samples: int, vectors: np.ndarray) -> int:
        """Determine optimal cluster count using Silhouette score.

        Automatically caps k to avoid FAISS warnings (n_samples >= 39 * k).

        Args:
            n_samples: Number of samples to cluster.
            vectors: Embedding vectors for silhouette search.

        Returns:
            Optimal number of clusters.
        """
        if n_samples < 2:
            return 1  # Single cluster for tiny libraries

        # FAISS recommends n_samples >= 39 * k to avoid warnings
        faiss_max_k = max(2, n_samples // FAISS_SAMPLES_PER_CLUSTER)
        max_k = min(35, faiss_max_k, self.config.max_clusters)

        if max_k < 2:
            # Not enough samples for meaningful multi-cluster analysis
            logger.info(
                "Library too small for multi-cluster analysis (%d samples, need %d for k=2)",
                n_samples,
                2 * FAISS_SAMPLES_PER_CLUSTER,
            )
            return 1

        return self._find_optimal_k_silhouette(vectors, min_k=2, max_k=max_k)

    def _find_optimal_k_silhouette(self, vectors: np.ndarray, min_k: int, max_k: int) -> int:
        """Find optimal cluster count using Silhouette score with biased k-selection.

        Tests different k values and returns the largest k within tolerance of
        the best Silhouette score. This prefers finer-grained research domains.

        Algorithm:
        1. Compute Silhouette score for each k in [min_k, max_k]
        2. Find global best score S_max
        3. Within {k | S_k >= S_max - delta} where delta = pct * |S_max|,
           select the maximum k

        Args:
            vectors: L2-normalized embedding vectors (N x dim).
            min_k: Minimum k to search.
            max_k: Maximum k to search.

        Returns:
            Optimal number of clusters (largest k within tolerance).
        """
        n_samples = vectors.shape[0]
        dim = vectors.shape[1]

        logger.info(
            "Searching for optimal k in range [%d, %d] using Silhouette score",
            min_k,
            max_k,
        )

        # Subsample for efficiency on large datasets
        subsample_threshold = self.config.subsample_threshold
        if n_samples > subsample_threshold:
            sample_idx = np.random.choice(n_samples, subsample_threshold, replace=False)
            sample_vectors = vectors[sample_idx]
            logger.debug(
                "Subsampled %d vectors to %d for silhouette search",
                n_samples,
                subsample_threshold,
            )
        else:
            sample_vectors = vectors

        # Collect all (k, score) pairs
        k_scores: list[tuple[int, float]] = []

        for k in range(min_k, max_k + 1):
            # Run k-means
            kmeans = faiss.Kmeans(
                dim,
                k,
                niter=10,  # Fewer iterations for search
                verbose=False,
                gpu=False,
                spherical=True,
                seed=42,
            )
            kmeans.train(sample_vectors)
            _, assignments = kmeans.index.search(sample_vectors, 1)
            labels = assignments.flatten()

            # Skip invalid silhouette cases (all points in one cluster)
            if len(set(labels.tolist())) < 2:
                logger.debug("k=%d skipped: single cluster produced, cannot compute silhouette", k)
                continue

            # Compute Silhouette score using sklearn (optimized C implementation)
            try:
                score = silhouette_score(sample_vectors, labels, metric="cosine")
                k_scores.append((k, score))
                logger.debug("k=%d, silhouette=%.4f", k, score)
            except ValueError as exc:
                logger.debug("k=%d silhouette failed (%s); skipping", k, exc)
                continue

        # If no valid silhouette scores, fall back to single cluster
        if not k_scores:
            logger.info("Silhouette search produced no valid k; falling back to k=1")
            self._last_silhouette_score = None
            return 1

        # Find global best score
        global_best_score = max(score for _, score in k_scores)
        global_best_k = max(k for k, score in k_scores if score == global_best_score)

        # Biased k-selection: within tolerance of best, select max k
        tolerance_fraction = self.config.biased_k_tolerance_percent
        delta = abs(global_best_score) * tolerance_fraction

        # Find all k values within tolerance of the best score
        candidates_within_tolerance = [k for k, score in k_scores if score >= global_best_score - delta]

        # Select the maximum k (prefer finer granularity)
        best_k = max(candidates_within_tolerance)
        best_score = next(score for k, score in k_scores if k == best_k)
        self._last_silhouette_score = best_score

        if best_k != global_best_k:
            logger.info(
                "Biased selection: k=%d (score=%.4f) over global best k=%d (score=%.4f) within tolerance=%.4f (%.1f%% of best)",
                best_k,
                best_score,
                global_best_k,
                global_best_score,
                delta,
                tolerance_fraction * 100,
            )
        else:
            logger.info(
                "Selected k=%d with Silhouette score=%.4f (tolerance=%.4f, %.1f%% of best)",
                best_k,
                best_score,
                delta,
                tolerance_fraction * 100,
            )

        return best_k

    def _build_single_cluster_profile(
        self,
        vectors: np.ndarray,
        items: list[ZoteroItem],
        temporal_weights: np.ndarray,
        temporal_config: dict | None,
        date_range: tuple[str, str] | None,
    ) -> ClusteredProfile:
        """Build a profile with all papers in a single cluster.

        Used when library is too small for meaningful multi-cluster analysis.

        Args:
            vectors: Embedding matrix (N x dim).
            items: Corresponding ZoteroItem list.
            temporal_weights: Temporal weights (N,) for each item.
            temporal_config: Temporal config dict for storage.
            date_range: (min_date, max_date) tuple.

        Returns:
            ClusteredProfile with a single cluster.
        """
        n_samples = vectors.shape[0]

        # Compute standard centroid as mean of all vectors
        centroid = np.mean(vectors, axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)

        # Compute weighted centroid: μ_k = Σ(w_i * v_i) / Σw_i
        effective_size = float(np.sum(temporal_weights))
        if effective_size > 0:
            weighted_sum = np.sum(vectors * temporal_weights[:, np.newaxis], axis=0)
            weighted_centroid = weighted_sum / effective_size
            weighted_centroid_norm = weighted_centroid / (np.linalg.norm(weighted_centroid) + 1e-8)
        else:
            weighted_centroid_norm = centroid_norm

        # Compute coherence (average similarity to centroid)
        similarities = np.dot(vectors, centroid_norm)
        coherence = float(np.mean(similarities))

        # Get representative titles (closest to centroid)
        title_count = min(self.config.representative_title_count, n_samples)
        top_indices = np.argsort(similarities)[-title_count:][::-1]
        representative_titles = [items[i].title for i in top_indices]

        # Extract common keywords from tags
        tag_counter: Counter[str] = Counter()
        for item in items:
            tag_counter.update(item.tags)
        keywords = [tag for tag, _ in tag_counter.most_common(10)]

        # Compute temporal metrics
        ages = [compute_item_age_days(item.date_added) for item in items]
        mean_age = float(np.mean(ages)) if ages else 0.0
        temporal_span = int(max(ages) - min(ages)) if ages else 0
        recent_ratio = sum(1 for a in ages if a <= 90) / len(ages) if ages else 0.0

        cluster = ClusterInfo(
            cluster_id=0,
            centroid=centroid_norm.tolist(),
            member_count=n_samples,
            member_keys=[item.key for item in items],
            representative_titles=representative_titles,
            keywords=keywords,
            coherence_score=coherence,
            weighted_centroid=weighted_centroid_norm.tolist(),
            effective_size=effective_size,
            mean_item_age_days=mean_age,
            temporal_span_days=temporal_span,
            recent_ratio=recent_ratio,
        )

        logger.info(
            "Created single cluster with %d papers (coherence=%.4f, effective_size=%.2f)",
            n_samples,
            coherence,
            effective_size,
        )

        return ClusteredProfile(
            clusters=[cluster],
            valid_cluster_count=1,
            min_cluster_size_used=self.config.min_cluster_size,
            total_papers=n_samples,
            papers_in_valid_clusters=n_samples,
            clustering_method="single",
            embedding_signature=self.embedding_signature,
            generated_at=utc_now(),
            temporal_config_used=temporal_config,
            profile_date_range=date_range,
            total_effective_size=effective_size,
        )

    def _build_cluster_info(
        self,
        centroids: np.ndarray,
        assignments: np.ndarray,
        vectors: np.ndarray,
        items: list[ZoteroItem],
        temporal_weights: np.ndarray,
    ) -> list[ClusterInfo]:
        """Build ClusterInfo objects from clustering results with temporal weighting.

        Args:
            centroids: Cluster centroids (K x dim).
            assignments: Cluster assignment for each sample.
            vectors: Sample embeddings (N x dim).
            items: Corresponding ZoteroItem list.
            temporal_weights: Temporal weights (N,) for each item.

        Returns:
            List of ClusterInfo objects sorted by member count.
        """
        n_clusters = centroids.shape[0]
        clusters: list[ClusterInfo] = []

        for cluster_id in range(n_clusters):
            mask = assignments == cluster_id
            member_indices = np.where(mask)[0]
            member_count = len(member_indices)

            if member_count == 0:
                continue

            # Get member items and their temporal weights
            member_items = [items[i] for i in member_indices]
            member_keys = [item.key for item in member_items]
            member_weights = temporal_weights[mask]
            member_vectors = vectors[mask]

            # Compute effective size: E_k = Σw_i
            effective_size = float(np.sum(member_weights))

            # Standard centroid (from k-means)
            centroid = centroids[cluster_id]
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)

            # Compute weighted centroid: μ_k = Σ(w_i * v_i) / Σw_i
            if effective_size > 0:
                weighted_sum = np.sum(member_vectors * member_weights[:, np.newaxis], axis=0)
                weighted_centroid = weighted_sum / effective_size
                weighted_centroid_norm = weighted_centroid / (np.linalg.norm(weighted_centroid) + 1e-8)
            else:
                weighted_centroid_norm = centroid_norm

            # Compute coherence (average cosine similarity to centroid)
            similarities = np.dot(member_vectors, centroid_norm)
            coherence = float(np.mean(similarities))

            # Extract representative titles (closest to centroid)
            title_count = min(self.config.representative_title_count, member_count)
            top_indices = np.argsort(similarities)[-title_count:][::-1]
            representative_titles = [member_items[i].title for i in top_indices]

            # Extract common keywords from tags
            tag_counter: Counter[str] = Counter()
            for item in member_items:
                tag_counter.update(item.tags)
            keywords = [tag for tag, _ in tag_counter.most_common(10)]

            # Compute temporal metrics
            ages = [compute_item_age_days(item.date_added) for item in member_items]
            mean_age = float(np.mean(ages)) if ages else 0.0
            temporal_span = int(max(ages) - min(ages)) if ages else 0
            recent_ratio = sum(1 for a in ages if a <= 90) / len(ages) if ages else 0.0

            clusters.append(
                ClusterInfo(
                    cluster_id=cluster_id,
                    centroid=centroid_norm.tolist(),
                    member_count=member_count,
                    member_keys=member_keys,
                    representative_titles=representative_titles,
                    keywords=keywords,
                    coherence_score=coherence,
                    weighted_centroid=weighted_centroid_norm.tolist(),
                    effective_size=effective_size,
                    mean_item_age_days=mean_age,
                    temporal_span_days=temporal_span,
                    recent_ratio=recent_ratio,
                )
            )

        # Sort by member count descending
        clusters.sort(key=lambda c: c.member_count, reverse=True)
        return clusters


__all__ = ["ProfileClusterer"]
