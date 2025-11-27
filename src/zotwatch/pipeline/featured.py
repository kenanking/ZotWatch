"""Featured paper selection based on user interests."""

import logging
from typing import List

import numpy as np

from zotwatch.config.settings import Settings
from zotwatch.core.models import CandidateWork, FeaturedWork
from zotwatch.infrastructure.embedding import FaissIndex, VoyageReranker
from zotwatch.infrastructure.embedding.base import BaseEmbeddingProvider
from zotwatch.llm import InterestRefiner

logger = logging.getLogger(__name__)


class FeaturedSelector:
    """Selects featured papers based on user interests using FAISS recall + Rerank."""

    def __init__(
        self,
        settings: Settings,
        vectorizer: BaseEmbeddingProvider,
        reranker: VoyageReranker,
        interest_refiner: InterestRefiner,
    ):
        """Initialize featured selector.

        Args:
            settings: Application settings
            vectorizer: Embedding provider for encoding
            reranker: Voyage reranker for semantic re-ranking
            interest_refiner: LLM-based interest refiner
        """
        self.settings = settings
        self.vectorizer = vectorizer
        self.reranker = reranker
        self.interest_refiner = interest_refiner

    def select(self, candidates: List[CandidateWork]) -> List[FeaturedWork]:
        """Select featured papers using interest-based reranking.

        Pipeline:
        1. Refine user interests using LLM
        2. Filter by exclude keywords
        3. Encode query and build temporary FAISS index
        4. FAISS recall top-K candidates
        5. Rerank using Voyage API
        6. Return top featured papers

        Args:
            candidates: List of candidate works to select from

        Returns:
            List of featured works sorted by relevance
        """
        if not candidates:
            return []

        interests_config = self.settings.scoring.interests

        # Step 1: Refine interests using LLM
        logger.info("Refining user interests with LLM...")
        refined = self.interest_refiner.refine(interests_config.description)
        logger.info(
            "Refined query: %s (include: %d, exclude: %d keywords)",
            refined.refined_query[:100] + "..." if len(refined.refined_query) > 100 else refined.refined_query,
            len(refined.include_keywords),
            len(refined.exclude_keywords),
        )

        # Step 2: Filter by exclude keywords
        filtered = self._apply_exclusions(candidates, refined.exclude_keywords)
        logger.info(
            "After exclusion filter: %d/%d candidates remain",
            len(filtered),
            len(candidates),
        )

        if not filtered:
            logger.warning("No candidates remaining after exclusion filter")
            return []

        # Step 3: Encode query and candidates
        logger.info("Encoding query and %d candidates...", len(filtered))
        query_vec = self.vectorizer.encode([refined.refined_query])
        candidate_texts = [c.content_for_embedding() for c in filtered]
        candidate_vecs = self.vectorizer.encode(candidate_texts)

        # Step 4: FAISS recall or use all candidates
        # When top_k_recall == -1, skip FAISS and use all candidates directly
        if interests_config.top_k_recall == -1:
            logger.info("top_k_recall=-1, skipping FAISS recall, using all %d candidates", len(filtered))
            recalled = filtered
            similarities = {}
            # Compute cosine similarities for all candidates
            query_norm = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
            candidate_norms = candidate_vecs / np.linalg.norm(candidate_vecs, axis=1, keepdims=True)
            sim_scores = np.dot(query_norm, candidate_norms.T)[0]
            for i, c in enumerate(filtered):
                similarities[c.identifier] = float(sim_scores[i])
        else:
            # Build temporary FAISS index and recall top-K
            temp_index, _ = FaissIndex.from_vectors(candidate_vecs.astype("float32"))
            top_k_recall = min(interests_config.top_k_recall, len(filtered))

            distances, indices = temp_index.search(query_vec, top_k=top_k_recall)

            # Get recalled candidates with their similarity scores
            recalled = []
            similarities = {}
            for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                if idx < len(filtered):
                    recalled.append(filtered[idx])
                    similarities[filtered[idx].identifier] = float(dist)

            logger.info("FAISS recalled %d candidates", len(recalled))

        if not recalled:
            return []

        # Step 5: Rerank using Voyage API
        logger.info("Reranking with Voyage API...")
        documents = [f"{c.title}\n{c.abstract or ''}" for c in recalled]
        top_k_featured = min(interests_config.top_k_featured, len(recalled))

        rerank_results = self.reranker.rerank(
            query=refined.refined_query,
            documents=documents,
            top_k=top_k_featured,
        )

        # Step 6: Build featured works
        featured = []
        for idx, score in rerank_results:
            work = recalled[idx]
            featured.append(
                FeaturedWork(
                    **work.model_dump(),
                    score=score,  # Use rerank score as primary score
                    similarity=similarities.get(work.identifier, 0.0),
                    rerank_score=score,
                    label="featured",
                )
            )

        logger.info(
            "Selected %d featured papers (top rerank score: %.4f)",
            len(featured),
            featured[0].rerank_score if featured else 0.0,
        )

        return featured

    def _apply_exclusions(
        self,
        candidates: List[CandidateWork],
        exclude_keywords: List[str],
    ) -> List[CandidateWork]:
        """Filter out candidates matching exclude keywords.

        Args:
            candidates: List of candidates to filter
            exclude_keywords: Keywords to exclude

        Returns:
            Filtered list of candidates
        """
        if not exclude_keywords:
            return candidates

        exclude_lower = [kw.lower() for kw in exclude_keywords]
        result = []

        for c in candidates:
            text = f"{c.title} {c.abstract or ''}".lower()
            if not any(kw in text for kw in exclude_lower):
                result.append(c)

        return result


__all__ = ["FeaturedSelector"]
