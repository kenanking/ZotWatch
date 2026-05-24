"""LLM-based abstract extraction from HTML content."""

import logging
import re

from zotwatch.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

# Extraction prompt for extracting abstract from HTML
EXTRACT_PROMPT = """Extract the abstract from the following academic paper webpage HTML.

Requirements:
1. Return ONLY the plain text abstract, no HTML tags
2. If there are multiple abstract paragraphs, merge them into one complete abstract
3. Remove any header words like "Abstract", "Summary", etc.
4. If you cannot find an abstract, return exactly "NOT_FOUND"
5. Do not add any explanation or extra content, return only the abstract text

HTML content (truncated):
{html}
"""


class LLMAbstractExtractor:
    """Extract abstracts from HTML using LLM."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        max_html_chars: int = 15000,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ):
        """Initialize the extractor.

        Args:
            llm: LLM provider for extraction.
            max_html_chars: Maximum HTML characters to send to LLM.
            max_tokens: Maximum tokens for LLM response.
            temperature: LLM temperature (low for extraction).
        """
        self.llm = llm
        self.max_html_chars = max_html_chars
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _preprocess_html(self, html: str) -> str:
        """Clean HTML to reduce token usage.

        Strategy:
        1. Try to extract the abstract section directly
        2. If found, use a smaller context around it
        3. Otherwise, clean and truncate the full HTML
        """
        # First, try to find the abstract section directly
        abstract_section = self._extract_abstract_section(html)
        if abstract_section:
            logger.debug("Found abstract section, using targeted extraction")
            return abstract_section

        # Fallback: clean the full HTML
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
        html = re.sub(r"<nav[^>]*>.*?</nav>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<header[^>]*>.*?</header>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<footer[^>]*>.*?</footer>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"\s+", " ", html)
        return html[: self.max_html_chars]

    def _extract_abstract_section(self, html: str) -> str | None:
        """Try to extract the abstract section from HTML.

        Strategy (in order of preference):
        1. og:description meta tag (often has full abstract, e.g., IEEE)
        2. description meta tag
        3. ID-based div/section patterns
        4. Class-based patterns
        """
        # 1. Try og:description meta tag first (often has full abstract)
        og_match = re.search(
            r'<meta[^>]*(?:property=["\']og:description["\'][^>]*content=["\']([^"\']+)["\']'
            r'|content=["\']([^"\']+)["\'][^>]*property=["\']og:description["\'])',
            html,
            re.IGNORECASE,
        )
        if og_match:
            og_abstract = (og_match.group(1) or og_match.group(2)).strip()
            # Verify it looks like an abstract (not just site description)
            if len(og_abstract) > 200 and not og_abstract.startswith("IEEE"):
                logger.debug("Found abstract in og:description (%d chars)", len(og_abstract))
                return f"Abstract from page metadata: {og_abstract}"

        # 2. Try description meta tag
        desc_match = re.search(
            r'<meta[^>]*(?:name=["\']description["\'][^>]*content=["\']([^"\']+)["\']'
            r'|content=["\']([^"\']+)["\'][^>]*name=["\']description["\'])',
            html,
            re.IGNORECASE,
        )
        if desc_match:
            desc_abstract = desc_match.group(1).strip()
            if len(desc_abstract) > 200:
                logger.debug("Found abstract in meta description (%d chars)", len(desc_abstract))
                return f"Abstract from page metadata: {desc_abstract}"

        # 3. Try div/section patterns
        patterns = [
            # H2 header patterns (ScienceDirect, Elsevier)
            r"<h2[^>]*>\s*Abstract\s*</h2>\s*(.*?)</div",
            r"<h2[^>]*>\s*Abstract\s*</h2>\s*<div[^>]*>(.*?)</div",
            # ID-based patterns (most reliable)
            r'id=["\']?abstracts?["\']?[^>]*>(.*?)</(?:div|section)',
            r'id=["\']?abstract-content["\']?[^>]*>(.*?)</(?:div|section)',
            # Class-based patterns
            r'class=["\'][^"\']*abstract[^"\']*["\'][^>]*>(.*?)</(?:div|section|p)',
            r'class=["\'][^"\']*Abstract[^"\']*["\'][^>]*>(.*?)</(?:div|section|p)',
            # Section with data-title
            r'data-title=["\']Abstract["\'][^>]*>(.*?)</section',
            # Springer, Nature patterns
            r'<section[^>]*aria-labelledby=["\'][^"\']*abstract[^"\']*["\'][^>]*>(.*?)</section',
        ]

        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
            if match:
                section = match.group(1) if match.lastindex else match.group(0)
                # Clean the extracted section
                section = re.sub(r"<[^>]+>", " ", section)  # Remove HTML tags
                section = re.sub(r"\s+", " ", section).strip()
                # Skip if it contains "Show More" (truncated)
                if "show more" in section.lower():
                    logger.debug("Skipping truncated abstract div, will try meta tags")
                    continue
                if len(section) > 100:  # Minimum meaningful abstract length
                    logger.debug("Found abstract in HTML section (%d chars)", len(section))
                    return f"Abstract section from page: {section[:5000]}"

        return None

    def extract(self, html: str, title: str | None = None) -> str | None:
        """Extract abstract from HTML content.

        Args:
            html: Raw HTML content.
            title: Optional paper title for context.

        Returns:
            Extracted abstract or None if not found.
        """
        if not html:
            return None

        cleaned_html = self._preprocess_html(html)
        prompt = EXTRACT_PROMPT.format(html=cleaned_html)

        if title:
            prompt = f"Paper title: {title}\n\n{prompt}"

        try:
            response = self.llm.complete(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            result = response.content.strip()

            # Check for NOT_FOUND response
            if "NOT_FOUND" in result or len(result) < 50:
                logger.debug("LLM extraction returned no abstract")
                return None

            logger.debug("LLM extracted abstract (%d chars)", len(result))
            return result

        except Exception as e:
            logger.warning("LLM extraction failed: %s", e)
            return None


__all__ = ["LLMAbstractExtractor"]
