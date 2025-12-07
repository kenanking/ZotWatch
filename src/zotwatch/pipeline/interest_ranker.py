"""Interest-based paper selection using FAISS recall + Rerank."""

import logging
from pathlib import Path

from zotwatch.config.settings import Settings
from zotwatch.core.models import CandidateWork, InterestWork
from zotwatch.infrastructure.embedding import FaissIndex
from zotwatch.infrastructure.embedding.base import BaseEmbeddingProvider, BaseReranker
from zotwatch.llm import InterestRefiner
from zotwatch.pipeline.journal_scorer import JournalScorer

logger = logging.getLogger(__name__)


class InterestRanker:
    """Selects papers based on user interests using FAISS recall + Rerank."""

    def __init__(
        self,
        settings: Settings,
        vectorizer: BaseEmbeddingProvider,
        reranker: BaseReranker,
        interest_refiner: InterestRefiner,
        base_dir: Path | str | None = None,
    ):
        """Initialize interest ranker.

        Args:
            settings: Application settings
            vectorizer: Embedding provider for encoding (typically CachingEmbeddingProvider)
            reranker: Reranker for semantic re-ranking (Voyage or DashScope)
            interest_refiner: LLM-based interest refiner
            base_dir: Base directory for data files (for loading journal whitelist)
        """
        self.settings = settings
        self.vectorizer = vectorizer
        self.reranker = reranker
        self.interest_refiner = interest_refiner
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self._journal_scorer = JournalScorer(self.base_dir)

    def select(self, candidates: list[CandidateWork]) -> list[InterestWork]:
        """Select interest-based papers using FAISS recall + reranking.

        Pipeline:
        1. Refine user interests using LLM
        2. Filter by exclude keywords
        3. Encode query (with input_type="query" for Voyage) and candidates
        4. FAISS recall max_documents candidates (capped at reranker limit)
        5. Rerank using reranker API (single batch)
        6. Return top interest-based papers

        Args:
            candidates: List of candidate works to select from

        Returns:
            List of interest works sorted by relevance
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
        # Use encode_query() for the refined query (uses input_type="query" for Voyage)
        logger.info("Encoding query and %d candidates...", len(filtered))
        query_vec = self.vectorizer.encode_query([refined.refined_query])
        candidate_texts = [c.content_for_embedding() for c in filtered]
        candidate_vecs = self.vectorizer.encode(candidate_texts)

        # Step 4: FAISS recall to limit candidates for reranking
        # Validate and cap max_documents to not exceed reranker API limit
        max_docs = interests_config.max_documents
        if max_docs > self.reranker.max_documents:
            logger.warning(
                "max_documents (%d) exceeds reranker limit (%d), capping to %d",
                max_docs,
                self.reranker.max_documents,
                self.reranker.max_documents,
            )
            max_docs = self.reranker.max_documents

        # Build temporary FAISS index and recall top-K
        temp_index, _ = FaissIndex.from_vectors(candidate_vecs.astype("float32"))
        recall_count = min(max_docs, len(filtered))

        distances, indices = temp_index.search(query_vec, top_k=recall_count)

        # Get recalled candidates with their similarity scores
        recalled = []
        similarities = {}
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(filtered):
                recalled.append(filtered[idx])
                similarities[filtered[idx].identifier] = float(dist)

        logger.info("FAISS recalled %d candidates (max_documents=%d)", len(recalled), max_docs)

        if not recalled:
            return []

        # Step 5: Rerank using configured provider
        provider = self.settings.scoring.rerank.provider
        model = getattr(self.reranker, "model", "n/a")
        logger.info(f"Reranking with {provider.capitalize()} API (model: {model})...")
        documents = [f"{c.title}\n{c.abstract or ''}" for c in recalled]
        top_k_interest = min(interests_config.top_k_interest, len(recalled))

        rerank_results = self.reranker.rerank(
            query=refined.refined_query,
            documents=documents,
            top_k=top_k_interest,
        )

        # Step 6: Build interest works with IF scores
        interest_results = []
        for idx, score in rerank_results:
            work = recalled[idx]
            if_score, raw_if, is_cn = self._journal_scorer.compute_score(work)
            interest_results.append(
                InterestWork(
                    **work.model_dump(),
                    score=score,  # Use rerank score as primary score
                    similarity=similarities.get(work.identifier, 0.0),
                    impact_factor_score=if_score,
                    impact_factor=raw_if,
                    is_chinese_core=is_cn,
                    rerank_score=score,
                    label="interest",
                )
            )

        logger.info(
            "Selected %d interest papers (top rerank score: %.4f)",
            len(interest_results),
            interest_results[0].rerank_score if interest_results else 0.0,
        )

        return interest_results

    def _apply_exclusions(
        self,
        candidates: list[CandidateWork],
        exclude_keywords: list[str],
    ) -> list[CandidateWork]:
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


__all__ = ["InterestRanker"]
