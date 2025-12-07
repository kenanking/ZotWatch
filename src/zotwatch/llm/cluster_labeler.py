"""Cluster labeling service using LLM."""

import json
import logging
import re

from zotwatch.core.models import ClusterInfo

from .base import BaseLLMProvider
from .prompts import BATCH_CLUSTER_LABEL_PROMPT, CLUSTER_LABEL_PROMPT

logger = logging.getLogger(__name__)


class ClusterLabeler:
    """Generates labels for semantic clusters using LLM."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        model: str | None = None,
    ):
        """Initialize cluster labeler.

        Args:
            llm: LLM provider instance.
            model: Optional model override.
        """
        self.llm = llm
        self.model = model

    def label_cluster(self, cluster: ClusterInfo) -> str:
        """Generate a label for a single cluster.

        Args:
            cluster: ClusterInfo with representative titles and keywords.

        Returns:
            Generated label string.
        """
        titles = "\n".join(f"- {t}" for t in cluster.representative_titles[:5])
        keywords = ", ".join(cluster.keywords[:10]) if cluster.keywords else "N/A"

        prompt = CLUSTER_LABEL_PROMPT.format(titles=titles, keywords=keywords)

        try:
            response = self.llm.complete(prompt, model=self.model, max_tokens=100)
            label = response.content.strip()
            # Remove quotes if present
            label = label.strip("\"'")
            logger.debug("Generated label for cluster %d: %s", cluster.cluster_id, label)
            return label
        except Exception as e:
            logger.warning("Failed to generate label for cluster %d: %s", cluster.cluster_id, e)
            # Fallback: use first keyword or first title word
            if cluster.keywords:
                return cluster.keywords[0]
            elif cluster.representative_titles:
                return cluster.representative_titles[0][:30] + "..."
            return f"Cluster {cluster.cluster_id}"

    def label_clusters_batch(self, clusters: list[ClusterInfo]) -> list[str]:
        """Generate labels for multiple clusters in a single LLM call.

        Args:
            clusters: List of ClusterInfo objects to label.

        Returns:
            List of generated labels in the same order.
        """
        if not clusters:
            return []

        # Build cluster info string
        clusters_info_parts = []
        for i, cluster in enumerate(clusters):
            titles = ", ".join(f'"{t}"' for t in cluster.representative_titles[:3])
            keywords = ", ".join(cluster.keywords[:5]) if cluster.keywords else "N/A"
            clusters_info_parts.append(
                f"Cluster {i + 1} ({cluster.member_count} papers):\n  Titles: {titles}\n  Keywords: {keywords}"
            )

        clusters_info = "\n\n".join(clusters_info_parts)
        prompt = BATCH_CLUSTER_LABEL_PROMPT.format(clusters_info=clusters_info)

        try:
            response = self.llm.complete(prompt, model=self.model, max_tokens=500)
            labels = self._parse_batch_response(response.content, len(clusters))
            logger.info("Generated %d cluster labels via batch LLM call", len(labels))
            return labels
        except Exception as e:
            logger.warning("Batch labeling failed, falling back to individual calls: %s", e)
            return [self.label_cluster(c) for c in clusters]

    def _parse_batch_response(self, content: str, expected_count: int) -> list[str]:
        """Parse batch labeling LLM response.

        Args:
            content: LLM response content.
            expected_count: Expected number of labels.

        Returns:
            List of labels.
        """
        content = content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)

        try:
            labels = json.loads(content)
            if isinstance(labels, list) and len(labels) == expected_count:
                return [str(label).strip("\"'") for label in labels]
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract labels line by line
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        labels = []
        for line in lines:
            # Remove list markers and quotes
            label = re.sub(r"^[\d\.\-\*\s]+", "", line).strip().strip("\"'")
            if label:
                labels.append(label)
            if len(labels) >= expected_count:
                break

        # Pad with generic labels if needed
        while len(labels) < expected_count:
            labels.append(f"Cluster {len(labels) + 1}")

        return labels[:expected_count]


__all__ = ["ClusterLabeler"]
