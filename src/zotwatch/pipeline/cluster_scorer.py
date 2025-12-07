"""Cluster-based similarity scoring."""

import logging
import math

import numpy as np
from pydantic import BaseModel

from zotwatch.config.settings import ClusteringConfig
from zotwatch.core.models import ClusteredProfile

logger = logging.getLogger(__name__)


class ClusterScore(BaseModel):
    """Cluster-based score breakdown (used for micro/macro fusion)."""

    final_score: float  # Cluster similarity (same as macro_score, helpful for debugging)
    cluster_similarities: list[tuple[int, float]]  # (cluster_id, similarity) pairs
    top_cluster_id: int | None = None
    macro_score: float = 0.0  # Normalized macro score in [0, 1]
    raw_macro_score: float = 0.0  # Unnormalized macro score


class ClusterScorer:
    """Computes cluster-based macro similarity for fusion scoring.

    - S_raw_macro(p, C_k) = sim(p, μ_k) × ln(1 + E_k)
    - S_macro(p) = max_k(S_raw_macro)
    - Normalized by max(ln(1 + E_k)) for [0, 1] range
    """

    def __init__(
        self,
        clustered_profile: ClusteredProfile,
        config: ClusteringConfig,
    ):
        """Initialize cluster scorer.

        Args:
            clustered_profile: Clustered profile with centroids.
            config: Clustering configuration.
        """
        self.profile = clustered_profile
        self.config = config

        # Pre-compute centroid matrix for efficient batch scoring
        if clustered_profile.clusters:
            # Use weighted centroids if available, fallback to standard centroids
            self.centroids = np.array(
                [c.weighted_centroid if c.weighted_centroid else c.centroid for c in clustered_profile.clusters],
                dtype=np.float32,
            )
            self.cluster_sizes = np.array([c.member_count for c in clustered_profile.clusters])
            self.cluster_ids = [c.cluster_id for c in clustered_profile.clusters]

            # Load effective sizes for macro score computation
            self.effective_sizes = np.array(
                [c.effective_size for c in clustered_profile.clusters],
                dtype=np.float32,
            )

            # Normalize centroids
            norms = np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-8
            self.centroids_norm = self.centroids / norms

            # Precompute ln(1 + E_k) for each cluster
            self.log_effective_sizes = np.array([math.log(1 + e) for e in self.effective_sizes], dtype=np.float32)

            # Normalization factor: max(ln(1 + E_k))
            self.max_log_effective_size = float(np.max(self.log_effective_sizes))

            # Total effective size for reference
            self.total_effective_size = clustered_profile.total_effective_size

            logger.debug(
                "Initialized ClusterScorer with %d clusters, total_effective_size=%.2f",
                len(clustered_profile.clusters),
                self.total_effective_size,
            )
        else:
            self.centroids = None
            self.centroids_norm = None
            self.cluster_sizes = None
            self.cluster_ids = []
            self.effective_sizes = None
            self.log_effective_sizes = None
            self.max_log_effective_size = 0.0
            self.total_effective_size = 0.0

    def score(self, vectors: np.ndarray) -> list[ClusterScore]:
        """Compute cluster-based macro scores for candidate vectors."""
        if self.centroids_norm is None or len(self.centroids_norm) == 0:
            # No valid clusters, return zeros
            return [ClusterScore(final_score=0.0, cluster_similarities=[]) for _ in range(vectors.shape[0])]

        # Normalize vectors for cosine similarity
        vectors = vectors.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        vectors_norm = vectors / norms

        # Compute similarities to all centroids (N x K)
        similarities = np.dot(vectors_norm, self.centroids_norm.T)

        scores: list[ClusterScore] = []

        for i in range(similarities.shape[0]):
            sims = similarities[i]
            # Build (cluster_id, similarity) pairs
            cluster_sims = [(self.cluster_ids[j], float(sims[j])) for j in range(len(sims))]
            cluster_sims.sort(key=lambda x: x[1], reverse=True)

            # Compute macro score: S_raw_macro = max_k(sim_k * ln(1 + E_k))
            raw_macro_scores = sims * self.log_effective_sizes
            best_macro_idx = int(np.argmax(raw_macro_scores))
            raw_macro_score = float(raw_macro_scores[best_macro_idx])

            # Normalize macro score by max(ln(1 + E_k)) to get [0, 1] range
            if self.max_log_effective_size > 0:
                macro_score = raw_macro_score / self.max_log_effective_size
            else:
                macro_score = 0.0

            # Get top cluster from macro scoring perspective
            macro_top_cluster_id = self.cluster_ids[best_macro_idx]

            scores.append(
                ClusterScore(
                    final_score=macro_score,  # Use macro score as the final cluster similarity
                    cluster_similarities=cluster_sims[:5],  # Top 5 for debugging
                    top_cluster_id=macro_top_cluster_id,  # Use macro-best cluster
                    macro_score=macro_score,
                    raw_macro_score=raw_macro_score,
                )
            )

        return scores

    def score_single(self, vector: np.ndarray) -> ClusterScore:
        """Score a single vector.

        Args:
            vector: Single embedding vector (dim,).

        Returns:
            ClusterScore for the vector.
        """
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        return self.score(vector)[0]


__all__ = ["ClusterScorer", "ClusterScore"]
