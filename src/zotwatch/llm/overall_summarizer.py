"""Overall section summary generation service."""

import json
import logging

from zotwatch.core.models import (
    InterestWork,
    OverallSummary,
    RankedWork,
    TopicSummary,
)
from zotwatch.utils.datetime import utc_now

from .base import BaseLLMProvider
from .prompts import OVERALL_SUMMARY_PROMPT

logger = logging.getLogger(__name__)


class OverallSummarizer:
    """Service for generating overall section summaries."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        model: str | None = None,
    ):
        self.llm = llm
        self.model = model

    def summarize_section(
        self,
        works: list[RankedWork | InterestWork],
        section_type: str,
    ) -> OverallSummary:
        """Generate overall summary for a section of papers.

        Args:
            works: List of papers to summarize
            section_type: "interest" or "similarity"

        Returns:
            OverallSummary with summary text and key themes
        """
        if not works:
            return OverallSummary(
                section_type=section_type,
                overview="No papers available for summary.",
                paper_count=0,
                generated_at=utc_now(),
                model_used="none",
                tokens_used=0,
            )

        # Build papers list for prompt
        papers_list = self._format_papers_list(works)

        section_label = "精选推荐" if section_type == "interest" else "相似度推荐"

        prompt = OVERALL_SUMMARY_PROMPT.format(
            paper_count=len(works),
            section_type=section_label,
            papers_list=papers_list,
        )

        response = self.llm.complete(prompt, model=self.model)

        return self._parse_response(
            response.content,
            section_type=section_type,
            paper_count=len(works),
            model_used=response.model,
            tokens_used=response.tokens_used,
        )

    def _format_papers_list(
        self,
        works: list[RankedWork | InterestWork],
        max_papers: int = 10,
    ) -> str:
        """Format papers list for prompt."""
        lines = []
        for i, work in enumerate(works[:max_papers], 1):
            abstract = work.abstract or ""
            abstract_snippet = abstract[:200]
            suffix = "..." if len(abstract) > 200 else ""
            lines.append(f"{i}. {work.title}\n   摘要片段：{abstract_snippet}{suffix}")

        if len(works) > max_papers:
            lines.append(f"...还有 {len(works) - max_papers} 篇论文")

        return "\n\n".join(lines)

    def _parse_response(
        self,
        content: str,
        section_type: str,
        paper_count: int,
        model_used: str,
        tokens_used: int,
    ) -> OverallSummary:
        """Parse LLM response into OverallSummary with topics."""
        try:
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)

            topics = [
                TopicSummary(
                    topic_name=t.get("topic_name", "未命名"),
                    paper_count=t.get("paper_count", 0),
                    description=t.get("description", ""),
                )
                for t in data.get("topics", [])
            ]

            return OverallSummary(
                section_type=section_type,
                overview=data.get("overview", ""),
                topics=topics,
                paper_count=paper_count,
                generated_at=utc_now(),
                model_used=model_used,
                tokens_used=tokens_used,
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse overall summary: %s", e)
            return OverallSummary(
                section_type=section_type,
                overview=f"本期共推荐 {paper_count} 篇论文。",
                topics=[],
                paper_count=paper_count,
                generated_at=utc_now(),
                model_used=model_used,
                tokens_used=tokens_used,
            )


__all__ = ["OverallSummarizer"]
