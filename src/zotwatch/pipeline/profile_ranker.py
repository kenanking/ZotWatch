"""Profile-based ranking pipeline."""

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from zotwatch.config.settings import Settings
from zotwatch.core.models import CandidateWork, RankedWork
from zotwatch.infrastructure.embedding import (
    CachingEmbeddingProvider,
    EmbeddingCache,
    FaissIndex,
    create_embedding_provider,
)
from zotwatch.infrastructure.embedding.base import BaseEmbeddingProvider
from zotwatch.infrastructure.storage import ProfileStorage
from zotwatch.pipeline.cluster_scorer import ClusterScorer
from zotwatch.pipeline.journal_scorer import JournalScorer
from zotwatch.utils.temporal import compute_batch_weights

logger = logging.getLogger(__name__)


@dataclass
class RankerArtifacts:
    """Paths to ranker artifact files."""

    index_path: Path


@dataclass
class ComputedThresholds:
    """Computed threshold values for current batch."""

    must_read: float
    consider: float
    mode: str  # "fixed" or "dynamic"


class ProfileRanker:
    """Ranks candidate works by embedding similarity to user's library profile."""

    def __init__(
        self,
        base_dir: Path | str,
        settings: Settings,
        vectorizer: BaseEmbeddingProvider | None = None,
        embedding_cache: EmbeddingCache | None = None,
    ):
        """Initialize profile ranker.

        Args:
            base_dir: Base directory for data files.
            settings: Application settings.
            vectorizer: Optional base embedding provider (defaults to VoyageEmbedding).
            embedding_cache: Optional embedding cache. If provided, wraps vectorizer
                            with CachingEmbeddingProvider for candidate source type.
        """
        self.base_dir = Path(base_dir)
        self.settings = settings
        self._cache = embedding_cache

        # Create base vectorizer
        base_vectorizer = vectorizer or create_embedding_provider(settings.embedding)

        # Wrap with cache if provided
        if embedding_cache is not None:
            self.vectorizer: BaseEmbeddingProvider = CachingEmbeddingProvider(
                provider=base_vectorizer,
                cache=embedding_cache,
                source_type="candidate",
                ttl_days=settings.embedding.candidate_ttl_days,
            )
        else:
            self.vectorizer = base_vectorizer

        self.artifacts = RankerArtifacts(
            index_path=self.base_dir / "data" / "faiss.index",
        )
        self.index = FaissIndex.load(self.artifacts.index_path)
        self._journal_scorer = JournalScorer(self.base_dir)
        self._last_computed_thresholds: ComputedThresholds | None = None

        # Load cluster scorer if clustering is enabled
        self._cluster_scorer: ClusterScorer | None = None
        self._load_cluster_scorer()

        # Load temporal weights for scoring
        self._item_temporal_weights: dict[int, float] = {}
        self._load_temporal_weights()

    @property
    def computed_thresholds(self) -> ComputedThresholds | None:
        """Return thresholds computed during the last rank() call."""
        return self._last_computed_thresholds

    def _load_cluster_scorer(self) -> None:
        """Load clustered profile and initialize cluster scorer if available."""
        if not self.settings.profile.clustering.enabled:
            return

        # Get embedding signature
        embedding_signature = self.settings.embedding.signature

        # Try to load from storage
        storage_path = self.base_dir / "data" / "profile.sqlite"
        if not storage_path.exists():
            logger.debug("Profile storage not found, skipping cluster scorer")
            return

        try:
            storage = ProfileStorage(storage_path)
            clustered_profile = storage.get_clustered_profile(embedding_signature)
            storage.close()

            if clustered_profile and clustered_profile.valid_cluster_count > 0:
                self._cluster_scorer = ClusterScorer(
                    clustered_profile,
                    self.settings.profile.clustering,
                )
                logger.info(
                    "Loaded cluster scorer with %d clusters for fusion scoring",
                    clustered_profile.valid_cluster_count,
                )
            else:
                logger.debug("No valid clustered profile found")
        except Exception as e:
            logger.warning("Failed to load cluster scorer: %s", e)

    def _load_temporal_weights(self) -> None:
        """Load precomputed temporal weights for profile items.

        Computes temporal weights for all items in the profile and maps them
        to their FAISS index positions for micro score computation.
        """
        storage_path = self.base_dir / "data" / "profile.sqlite"
        if not storage_path.exists():
            logger.debug("Profile storage not found, skipping temporal weights")
            return

        try:
            storage = ProfileStorage(storage_path)
            items = storage.get_items_with_abstract()
            storage.close()

            if not items:
                return

            # Compute temporal weights (or uniform if disabled)
            if self.settings.profile.clustering.temporal.enabled:
                halflife = self.settings.profile.clustering.temporal.halflife_days
                min_weight = self.settings.profile.clustering.temporal.min_weight
                weights = compute_batch_weights(items, halflife, min_weight)
                weights_mode = f"temporal (halflife={halflife}d)"
            else:
                weights = [1.0] * len(items)
                weights_mode = "uniform (temporal disabled)"

            index_size = self.index.ntotal
            weight_count = len(weights)

            if index_size != weight_count:
                if index_size == 0:
                    logger.debug("FAISS index is empty; skipping temporal weights")
                    return
                if weight_count > index_size:
                    logger.warning(
                        "Temporal weight count (%d) exceeds index size (%d); truncating to match index",
                        weight_count,
                        index_size,
                    )
                    weights = weights[:index_size]
                else:
                    logger.warning(
                        "Temporal weight count (%d) smaller than index size (%d); padding with 1.0",
                        weight_count,
                        index_size,
                    )
                    weights.extend([1.0] * (index_size - weight_count))

            # Map index position to weight (aligned with embedded items)
            self._item_temporal_weights = {i: w for i, w in enumerate(weights[:index_size])}

            logger.info(
                "Loaded %s weights for %d items",
                weights_mode,
                len(self._item_temporal_weights),
            )
        except Exception as e:
            logger.warning("Failed to load temporal weights: %s", e)

    def _compute_micro_score(
        self,
        candidate_vector: np.ndarray,
        k: int = 5,
    ) -> float:
        """Compute micro score using k-NN with temporal weighting.

        S_micro = Σ(sim_r * w_r) / (Σw_r + ε)

        Args:
            candidate_vector: Candidate embedding (dim,).
            k: Number of nearest neighbors (L in the formula).

        Returns:
            Micro similarity score in [0, 1].
        """
        if self.index is None or self.index.ntotal == 0:
            return 0.0

        # Search k nearest neighbors
        vector_2d = candidate_vector.reshape(1, -1)
        distances, indices = self.index.search(vector_2d, top_k=k)

        sims = distances[0]  # Cosine similarities
        neighbor_indices = indices[0]

        # Get temporal weights for neighbors
        neighbor_weights = [
            self._item_temporal_weights.get(int(idx), 1.0)
            for idx in neighbor_indices
            if idx >= 0  # FAISS returns -1 for missing neighbors
        ]
        valid_sims = [sims[i] for i, idx in enumerate(neighbor_indices) if idx >= 0]

        if not neighbor_weights:
            return 0.0

        # Weighted average: S_micro = Σ(sim_r * w_r) / (Σw_r + ε)
        weight_sum = sum(neighbor_weights) + 1e-8
        micro_score = sum(s * w for s, w in zip(valid_sims, neighbor_weights)) / weight_sum

        return float(micro_score)

    def _compute_thresholds(self, scores: list[float]) -> ComputedThresholds:
        """Compute thresholds based on configuration mode.

        Args:
            scores: All computed scores for the current batch.

        Returns:
            ComputedThresholds with must_read and consider threshold values.
        """
        thresholds_config = self.settings.scoring.thresholds

        if thresholds_config.mode == "fixed":
            return ComputedThresholds(
                must_read=thresholds_config.must_read,
                consider=thresholds_config.consider,
                mode="fixed",
            )

        # Dynamic mode: compute from percentiles
        dynamic = thresholds_config.dynamic

        if len(scores) < 2:
            # Fallback for very small batches
            logger.warning("Batch too small for dynamic thresholds, using fixed fallback")
            return ComputedThresholds(
                must_read=thresholds_config.must_read,
                consider=thresholds_config.consider,
                mode="fixed",
            )

        scores_array = np.array(scores)

        # Compute percentile-based thresholds
        # For top 5%, we want 95th percentile (95% of values are below this)
        must_read_threshold = float(np.percentile(scores_array, dynamic.must_read_percentile))
        consider_threshold = float(np.percentile(scores_array, dynamic.consider_percentile))

        # Apply minimum thresholds to avoid labeling low-quality papers
        must_read_threshold = max(must_read_threshold, dynamic.min_must_read)
        consider_threshold = max(consider_threshold, dynamic.min_consider)

        # Ensure must_read > consider
        if must_read_threshold <= consider_threshold:
            must_read_threshold = consider_threshold + 0.01

        logger.info(
            "Dynamic thresholds computed: must_read=%.3f (p%.0f), consider=%.3f (p%.0f)",
            must_read_threshold,
            dynamic.must_read_percentile,
            consider_threshold,
            dynamic.consider_percentile,
        )

        return ComputedThresholds(
            must_read=must_read_threshold,
            consider=consider_threshold,
            mode="dynamic",
        )

    def _assign_label(self, score: float, thresholds: ComputedThresholds) -> str:
        """Assign label based on score and computed thresholds."""
        if score >= thresholds.must_read:
            return "must_read"
        elif score >= thresholds.consider:
            return "consider"
        return "ignore"

    def _is_empty_profile(self) -> bool:
        """Check if user library profile is empty (no indexed papers)."""
        return self.index is None or self.index.ntotal == 0

    def _random_rank(self, candidates: list[CandidateWork]) -> list[RankedWork]:
        """Randomly rank candidates when profile is empty.

        Used for new users with empty Zotero libraries. All candidates
        are shuffled and assigned 'consider' label to encourage exploration.

        Args:
            candidates: List of candidate works to rank.

        Returns:
            Randomly ordered list of RankedWork with 'consider' labels.
        """
        logger.info(
            "Empty profile detected, using random recommendation for %d candidates",
            len(candidates),
        )

        # Shuffle candidates randomly
        shuffled = list(candidates)
        random.shuffle(shuffled)

        # Build RankedWork with zero similarity scores
        ranked: list[RankedWork] = []
        for candidate in shuffled:
            if_score, raw_if, is_cn = self._journal_scorer.compute_score(candidate)
            ranked.append(
                RankedWork(
                    **candidate.model_dump(),
                    score=0.0,
                    similarity=0.0,
                    impact_factor_score=if_score,
                    impact_factor=raw_if,
                    is_chinese_core=is_cn,
                    label="consider",  # All get 'consider' to encourage exploration
                )
            )

        # Store thresholds as N/A for random mode
        self._last_computed_thresholds = ComputedThresholds(
            must_read=0.0,
            consider=0.0,
            mode="random",
        )

        return ranked

    def rank(self, candidates: list[CandidateWork]) -> list[RankedWork]:
        """Rank candidates by embedding similarity.

        Always use micro k-NN + macro cluster fusion:
        - similarity = α * S_micro + (1 - α) * S_macro
        - score = 0.8 * similarity + 0.2 * IF
        Fallback to single-neighbor when clusters or temporal weights are unavailable;
        fallback to random when the profile is empty.
        """
        if not candidates:
            return []

        # Check for empty profile - use random recommendation
        if self._is_empty_profile():
            return self._random_rank(candidates)

        # Encode candidates using unified interface (caching handled automatically)
        texts = [c.content_for_embedding() for c in candidates]
        vectors = self.vectorizer.encode(texts)
        logger.info("Scoring %d candidate works", len(candidates))

        fusion_config = self.settings.scoring.fusion
        use_fusion = self._cluster_scorer is not None

        # First pass: compute all scores
        scores_data: list[
            tuple[CandidateWork, float, float, float, float | None, bool, float | None, float | None, int | None]
        ] = []

        if use_fusion:
            alpha = fusion_config.micro_weight
            knn_k = fusion_config.knn_neighbors

            # Compute macro scores for all candidates at once
            cluster_scores = self._cluster_scorer.score(vectors)

            logger.info(
                "Using fusion scoring (α=%.2f, k=%d neighbors)",
                alpha,
                knn_k,
            )

            for i, candidate in enumerate(candidates):
                # Micro score: k-NN with temporal weighting
                micro = self._compute_micro_score(vectors[i], k=knn_k)

                # Macro score: cluster-based (already normalized)
                macro = cluster_scores[i].macro_score
                top_cluster = cluster_scores[i].top_cluster_id

                # Fusion: similarity = α * S_micro + (1-α) * S_macro
                similarity = alpha * micro + (1 - alpha) * macro

                # Final score: 0.8 * similarity + 0.2 * IF
                if_score, raw_if, is_cn = self._journal_scorer.compute_score(candidate)
                score = 0.8 * similarity + 0.2 * if_score

                scores_data.append(
                    (
                        candidate,
                        score,
                        similarity,
                        if_score,
                        raw_if,
                        is_cn,
                        micro,
                        macro,
                        top_cluster,
                    )
                )

        else:
            # Fallback to original single-neighbor approach
            distances, _ = self.index.search(vectors, top_k=1)

            for i, candidate in enumerate(candidates):
                similarity = float(distances[i][0]) if distances[i].size else 0.0
                if_score, raw_if, is_cn = self._journal_scorer.compute_score(candidate)
                score = 0.8 * similarity + 0.2 * if_score
                scores_data.append(
                    (
                        candidate,
                        score,
                        similarity,
                        if_score,
                        raw_if,
                        is_cn,
                        None,
                        None,
                        None,
                    )
                )

        # Compute thresholds from score distribution
        all_scores = [s[1] for s in scores_data]
        computed_thresholds = self._compute_thresholds(all_scores)
        self._last_computed_thresholds = computed_thresholds

        # Second pass: assign labels with computed thresholds
        ranked: list[RankedWork] = []
        for candidate, score, similarity, if_score, raw_if, is_cn, micro, macro, top_cluster in scores_data:
            label = self._assign_label(score, computed_thresholds)
            ranked.append(
                RankedWork(
                    **candidate.model_dump(),
                    score=score,
                    similarity=similarity,
                    impact_factor_score=if_score,
                    impact_factor=raw_if,
                    is_chinese_core=is_cn,
                    label=label,
                    micro_score=micro,
                    macro_score=macro,
                    matched_cluster_id=top_cluster,
                )
            )

        ranked.sort(key=lambda w: w.score, reverse=True)

        # Log label distribution
        label_counts = {"must_read": 0, "consider": 0, "ignore": 0}
        for w in ranked:
            label_counts[w.label] += 1
        total = len(ranked)
        logger.info(
            "Label distribution (%s mode): must_read=%d (%.1f%%), consider=%d (%.1f%%), ignore=%d (%.1f%%)",
            computed_thresholds.mode,
            label_counts["must_read"],
            100 * label_counts["must_read"] / total if total else 0,
            label_counts["consider"],
            100 * label_counts["consider"] / total if total else 0,
            label_counts["ignore"],
            100 * label_counts["ignore"] / total if total else 0,
        )

        return ranked


__all__ = ["ProfileRanker", "ComputedThresholds"]
