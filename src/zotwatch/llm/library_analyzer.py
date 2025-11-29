"""Library profile analysis service using LLM."""

import json
import logging

from zotwatch.core.models import (
    DomainDistribution,
    ResearcherProfile,
    ResearcherProfileInsights,
    ZoteroItem,
)

from .base import BaseLLMProvider
from .prompts import DOMAIN_CLASSIFICATION_PROMPT, PROFILE_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


class LibraryAnalyzer:
    """Service for analyzing user's research library and generating insights."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        model: str | None = None,
    ):
        """Initialize the library analyzer.

        Args:
            llm: LLM provider instance.
            model: Optional model name to use.
        """
        self.llm = llm
        self.model = model

    def classify_domains(
        self,
        items: list[ZoteroItem],
        max_domains: int = 10,
        max_papers: int = 200,
    ) -> list[DomainDistribution]:
        """Use LLM to classify papers into research domains.

        Args:
            items: List of Zotero items to classify.
            max_domains: Maximum number of domains to return.
            max_papers: Maximum number of papers to include in prompt.

        Returns:
            List of domain distributions.
        """
        if not items:
            return []

        # Limit papers for API efficiency
        papers_to_classify = items[:max_papers]

        # Build papers list for prompt
        papers_list = self._format_papers_for_classification(papers_to_classify)

        prompt = DOMAIN_CLASSIFICATION_PROMPT.format(
            paper_count=len(papers_to_classify),
            papers_list=papers_list,
            max_domains=max_domains,
        )

        try:
            response = self.llm.complete(prompt, model=self.model)
            domains = self._parse_domains_response(response.content)

            # Calculate percentages
            total = len(items)
            for domain in domains:
                domain.percentage = (domain.paper_count / total) * 100 if total > 0 else 0

            return domains
        except Exception as e:
            logger.warning("Failed to classify domains: %s", e)
            return []

    def generate_insights(
        self,
        profile: ResearcherProfile,
    ) -> ResearcherProfileInsights | None:
        """Generate natural language insights from profile statistics.

        Args:
            profile: ResearcherProfile with statistics.

        Returns:
            ResearcherProfileInsights or None if generation fails.
        """
        if profile.total_papers == 0:
            return None

        prompt = PROFILE_ANALYSIS_PROMPT.format(
            total_papers=profile.total_papers,
            collection_duration=profile.collection_duration or "未知",
            year_range=f"{profile.year_range[0]}-{profile.year_range[1]}",
            top_domains=self._format_domains(profile.domains[:5]),
            top_authors=self._format_authors(profile.authors[:10]),
            top_venues=self._format_venues(profile.venues[:10]),
            top_keywords=self._format_keywords(profile.keywords[:20]),
            quarterly_trends=self._format_trends(profile.quarterly_trends),
            recent_analysis=self._format_recent(profile.recent_analysis),
        )

        try:
            response = self.llm.complete(prompt, model=self.model)
            return self._parse_insights_response(response.content)
        except Exception as e:
            logger.warning("Failed to generate profile insights: %s", e)
            return None

    def _format_papers_for_classification(
        self,
        items: list[ZoteroItem],
    ) -> str:
        """Format papers list for domain classification prompt."""
        lines = []
        for i, item in enumerate(items, 1):
            # Include title and tags for classification
            tags_str = ", ".join(item.tags[:5]) if item.tags else "无标签"
            lines.append(f"{i}. {item.title}\n   标签：{tags_str}")

        return "\n\n".join(lines)

    def _format_domains(self, domains: list[DomainDistribution]) -> str:
        """Format domains for prompt."""
        if not domains:
            return "暂无领域分类数据"
        lines = [f"- {d.domain}: {d.paper_count}篇 ({d.percentage:.1f}%)" for d in domains]
        return "\n".join(lines)

    def _format_authors(self, authors: list) -> str:
        """Format authors for prompt."""
        if not authors:
            return "暂无作者数据"
        lines = [f"- {a.author}: {a.paper_count}篇" for a in authors]
        return "\n".join(lines)

    def _format_venues(self, venues: list) -> str:
        """Format venues for prompt."""
        if not venues:
            return "暂无期刊/会议数据"
        lines = [f"- {v.venue} [{v.venue_type}]: {v.paper_count}篇" for v in venues]
        return "\n".join(lines)

    def _format_keywords(self, keywords: list) -> str:
        """Format keywords for prompt."""
        if not keywords:
            return "暂无关键词数据"
        return ", ".join(f"{k.keyword}({k.count})" for k in keywords)

    def _format_trends(self, trends: list) -> str:
        """Format quarterly trends for prompt."""
        if not trends:
            return "暂无趋势数据"

        # Only show last 12 quarters
        recent_trends = trends[-12:]
        lines = [f"- {t.quarter}: {t.paper_count}篇" for t in recent_trends if t.paper_count > 0]
        if not lines:
            return "近期无新增论文"
        return "\n".join(lines)

    def _format_recent(self, recent) -> str:
        """Format recent papers analysis for prompt."""
        if not recent:
            return "暂无近期数据"

        lines = [f"- 近期新增论文：{recent.paper_count}篇"]
        if recent.new_keywords:
            lines.append(f"- 新增关键词：{', '.join(recent.new_keywords)}")
        else:
            lines.append("- 新增关键词：无明显新增")

        return "\n".join(lines)

    def _parse_domains_response(self, content: str) -> list[DomainDistribution]:
        """Parse LLM response into domain distributions."""
        try:
            content = self._clean_json_response(content)
            data = json.loads(content)

            domains = []
            for d in data.get("domains", []):
                domains.append(
                    DomainDistribution(
                        domain=d.get("domain", "未命名"),
                        paper_count=d.get("paper_count", 0),
                        sample_titles=d.get("sample_titles", [])[:3],
                    )
                )

            return domains
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse domains response: %s", e)
            return []

    def _parse_insights_response(self, content: str) -> ResearcherProfileInsights:
        """Parse LLM response into profile insights."""
        try:
            content = self._clean_json_response(content)
            data = json.loads(content)

            return ResearcherProfileInsights(
                research_focus_summary=data.get("research_focus_summary", "暂无分析"),
                strength_areas=data.get("strength_areas", "暂无分析"),
                interdisciplinary_notes=data.get("interdisciplinary_notes", "暂无分析"),
                trend_observations=data.get("trend_observations", "暂无分析"),
                recommendations=data.get("recommendations", "暂无分析"),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse insights response: %s", e)
            return ResearcherProfileInsights(
                research_focus_summary="分析生成失败，请重试。",
                strength_areas="分析生成失败，请重试。",
                interdisciplinary_notes="分析生成失败，请重试。",
                trend_observations="分析生成失败，请重试。",
                recommendations="分析生成失败，请重试。",
            )

    def _clean_json_response(self, content: str) -> str:
        """Clean JSON response by removing markdown formatting."""
        content = content.strip()
        if content.startswith("```"):
            # Remove markdown code block
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
        return content


__all__ = ["LibraryAnalyzer"]
