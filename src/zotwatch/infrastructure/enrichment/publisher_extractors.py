"""Publisher-specific abstract extraction rules.

This module provides rule-based abstract extraction for major academic publishers.
The rules are tried in order, and if all fail, the caller can fall back to LLM extraction.

Supported publishers:
- ACM Digital Library (dl.acm.org)
- IEEE Xplore (ieeexplore.ieee.org)
- Springer/Nature (link.springer.com, nature.com, springeropen.com)
- Elsevier/ScienceDirect (sciencedirect.com)
- SPIE (spiedigitallibrary.org)
- MDPI (mdpi.com)
- Taylor & Francis (tandfonline.com)
- Wiley (onlinelibrary.wiley.com)
- arXiv (arxiv.org)
"""

import html
import logging
import re
from typing import Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Publisher patterns: domain -> extraction configuration
PUBLISHER_CONFIGS: Dict[str, Dict] = {
    "acm": {
        "domains": ["dl.acm.org"],
        "meta_tags": [
            ("property", "og:description"),
            ("name", "dcterms.abstract"),
            ("name", "description"),
        ],
        "selectors": [
            r'<div[^>]*role=["\']paragraph["\'][^>]*>(.*?)</div>',
            r'<section[^>]*class=["\'][^"\']*abstract[^"\']*["\'][^>]*>(.*?)</section>',
        ],
    },
    "ieee": {
        "domains": ["ieeexplore.ieee.org"],
        "selectors_first": True,
        "meta_tags": [
            ("property", "og:description"),
            ("property", "twitter:description"),
            ("name", "description"),
        ],
        "selectors": [
            r'"abstract"\s*:\s*"((?:[^"\\]|\\.)+)"',
            r'<div[^>]*class=["\'][^"\']*abstract-text[^"\']*["\'][^>]*>(.*?)</div>',
        ],
    },
    "springer": {
        "domains": ["link.springer.com", "nature.com", "springeropen.com", "biomedcentral.com"],
        "meta_tags": [
            ("name", "dc.description"),
            ("property", "og:description"),
            ("name", "description"),
        ],
        "selectors": [
            r'<div[^>]*id=["\']Abs1-content["\'][^>]*>(.*?)</div>',
            r'<section[^>]*aria-labelledby=["\']Abs1["\'][^>]*>(.*?)</section>',
            r'<div[^>]*class=["\'][^"\']*c-article-section__content[^"\']*["\'][^>]*>(.*?)</div>',
            r'<p[^>]*id=["\']Par1["\'][^>]*>(.*?)</p>',
        ],
    },
    "elsevier": {
        "domains": ["sciencedirect.com", "linkinghub.elsevier.com"],
        # Elsevier's og:description is truncated (~150 chars), so try selectors first
        "selectors_first": True,
        "meta_tags": [
            ("property", "og:description"),
            ("name", "dc.description"),
            ("name", "description"),
        ],
        "selectors": [
            # ScienceDirect abstract sections - order matters!
            # 1. Preview pages: "abstract author" with sp[N] content div
            r'<div[^>]*class=["\']abstract author["\'][^>]*>.*?<h2[^>]*>Abstract</h2>.*?<div[^>]*id=["\']sp\d+["\'][^>]*>(.*?)</div>',
            # 2. Preview pages: "abstract author" with abss[N] content div
            r'<div[^>]*class=["\']abstract author["\'][^>]*>.*?<h2[^>]*>Abstract</h2>.*?<div[^>]*id=["\']abss\d+["\'][^>]*>(.*?)</div>',
            # 3. Full article pages: "abstract author" with u-margin-s-bottom content div (dynamic IDs like d1e####)
            r'<div[^>]*class=["\']abstract author["\'][^>]*>.*?<h2[^>]*>Abstract</h2>.*?<div[^>]*class=["\']u-margin-s-bottom["\'][^>]*>(.*?)</div>',
            # 4. Legacy patterns for other ScienceDirect layouts
            r'<div[^>]*id=["\']abs000\d["\'][^>]*>(.*?)</div>',
            r'<section[^>]*id=["\']abstracts?["\'][^>]*>.*?<div[^>]*>(.*?)</div>',
        ],
    },
    "spie": {
        "domains": ["spiedigitallibrary.org"],
        "meta_tags": [
            ("name", "citation_abstract"),
            ("property", "og:description"),
            ("name", "description"),
        ],
        "selectors": [
            r'<div[^>]*class=["\'][^"\']*abstractSection[^"\']*["\'][^>]*>(.*?)</div>',
        ],
    },
    "mdpi": {
        "domains": ["mdpi.com"],
        "meta_tags": [
            ("name", "dc.description"),
            ("property", "og:description"),
        ],
        "selectors": [
            r'<div[^>]*class=["\'][^"\']*art-abstract[^"\']*["\'][^>]*>(.*?)</div>',
            r'<section[^>]*class=["\'][^"\']*html-abstract[^"\']*["\'][^>]*>(.*?)</section>',
        ],
    },
    "taylor_francis": {
        "domains": ["tandfonline.com"],
        # Taylor & Francis og:description is truncated (~200 chars), try selectors first
        "selectors_first": True,
        "meta_tags": [
            ("property", "og:description"),
            ("name", "dc.description"),
        ],
        "selectors": [
            # hlFld-Abstract: div > h2 > p structure (most common T&F layout)
            r'<div[^>]*class=["\'][^"\']*hlFld-Abstract[^"\']*["\'][^>]*>.*?<p[^>]*>(.*?)</p>',
            # abstractSection with h2 header then paragraph
            r'<div[^>]*class=["\'][^"\']*abstractSection[^"\']*["\'][^>]*>.*?<p[^>]*>(.*?)</p>',
            # abstractInFull with any content before paragraph
            r'<div[^>]*class=["\'][^"\']*abstractInFull[^"\']*["\'][^>]*>.*?<p[^>]*>(.*?)</p>',
            # Generic abstract section - capture all content before closing div
            r'<div[^>]*class=["\'][^"\']*abstractSection[^"\']*["\'][^>]*>(.*?)</div>',
        ],
    },
    "wiley": {
        "domains": ["onlinelibrary.wiley.com"],
        "meta_tags": [
            ("property", "og:description"),
            ("name", "dc.description"),
        ],
        "selectors": [
            r'<section[^>]*class=["\'][^"\']*article-section__abstract[^"\']*["\'][^>]*>(.*?)</section>',
            r'<div[^>]*class=["\'][^"\']*abstract-group[^"\']*["\'][^>]*>(.*?)</div>',
        ],
    },
    "arxiv": {
        "domains": ["arxiv.org"],
        "meta_tags": [
            ("property", "og:description"),
            ("name", "citation_abstract"),
        ],
        "selectors": [
            r'<blockquote[^>]*class=["\'][^"\']*abstract[^"\']*["\'][^>]*>(.*?)</blockquote>',
        ],
    },
}

# Generic patterns for unknown publishers
GENERIC_META_TAGS = [
    ("name", "citation_abstract"),
    ("property", "og:description"),
    ("name", "dc.description"),
    ("name", "description"),
]

GENERIC_SELECTORS = [
    r'<div[^>]*id=["\']abstracts?["\'][^>]*>(.*?)</div>',
    r'<section[^>]*class=["\'][^"\']*abstract[^"\']*["\'][^>]*>(.*?)</section>',
    r'<div[^>]*class=["\'][^"\']*abstract[^"\']*["\'][^>]*>(.*?)</div>',
]


def detect_publisher(url: str) -> str:
    """Detect publisher from URL.

    Args:
        url: Page URL.

    Returns:
        Publisher key (e.g., "acm", "ieee") or "unknown".
    """
    if not url:
        return "unknown"

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
    except Exception:
        return "unknown"

    for publisher, config in PUBLISHER_CONFIGS.items():
        for pub_domain in config["domains"]:
            if pub_domain in domain:
                return publisher

    return "unknown"


def _clean_html_text(text: str) -> str:
    """Clean extracted HTML text.

    Args:
        text: Raw extracted text (may contain HTML entities, JSON escapes, and extra whitespace).

    Returns:
        Cleaned plain text.
    """
    if not text:
        return ""

    # Decode JSON escape sequences (for content extracted from JavaScript/JSON)
    # Order matters: handle double backslash first to avoid incorrect substitutions
    # e.g., \\n should become \n (literal), not a space
    text = re.sub(r"\\\\", "\x00", text)  # Temporarily replace \\ with placeholder
    text = text.replace(r"\"", '"')
    text = text.replace(r"\n", " ")
    text = text.replace(r"\t", " ")
    text = text.replace(r"\r", "")
    text = text.replace("\x00", "\\")  # Restore backslashes

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove "Abstract" header
    text = re.sub(r"^\s*Abstract\s*:?\s*", "", text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _extract_meta_tag(html_content: str, attr_name: str, attr_value: str) -> Optional[str]:
    """Extract content from a meta tag.

    Args:
        html_content: HTML content.
        attr_name: Attribute name ("name" or "property").
        attr_value: Attribute value to match.

    Returns:
        Meta tag content or None.
    """
    # Pattern 1: content before attr (e.g., <meta content="..." property="og:description">)
    pattern1 = rf'<meta[^>]*content=["\']([^"\']+)["\'][^>]*{attr_name}=["\']?{attr_value}["\']?[^>]*>'
    match = re.search(pattern1, html_content, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Pattern 2: attr before content (e.g., <meta property="og:description" content="...">)
    pattern2 = rf'<meta[^>]*{attr_name}=["\']?{attr_value}["\']?[^>]*content=["\']([^"\']+)["\'][^>]*>'
    match = re.search(pattern2, html_content, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None


def _extract_from_selector(html_content: str, selector_pattern: str) -> Optional[str]:
    """Extract abstract using regex selector pattern.

    Args:
        html_content: HTML content.
        selector_pattern: Regex pattern with capture group for content.

    Returns:
        Extracted and cleaned text or None.
    """
    match = re.search(selector_pattern, html_content, re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1)
        cleaned = _clean_html_text(text)
        # Minimum length check
        if len(cleaned) >= 100:
            return cleaned
    return None


def extract_abstract(html_content: str, url: str) -> Optional[str]:
    """Extract abstract using publisher-specific rules.

    This function tries rule-based extraction first:
    1. Detect publisher from URL
    2. Try publisher-specific extraction (meta tags or selectors, order depends on config)
    3. Try generic meta tags
    4. Try generic selectors

    Some publishers (e.g., Elsevier) have truncated meta descriptions, so we try
    selectors first for those publishers (controlled by `selectors_first` config).

    If all rules fail, returns None so caller can fall back to LLM.

    Args:
        html_content: Raw HTML content.
        url: Page URL (for publisher detection).

    Returns:
        Extracted abstract or None.
    """
    if not html_content:
        return None

    publisher = detect_publisher(url)
    logger.debug("Detected publisher: %s for URL: %s", publisher, url)

    # Get publisher-specific config or use generic
    if publisher != "unknown":
        config = PUBLISHER_CONFIGS[publisher]
        meta_tags = config.get("meta_tags", [])
        selectors = config.get("selectors", [])
        selectors_first = config.get("selectors_first", False)
    else:
        meta_tags = []
        selectors = []
        selectors_first = False

    # For publishers with truncated meta descriptions, try selectors first
    if selectors_first:
        # Try publisher-specific selectors first
        for selector in selectors:
            content = _extract_from_selector(html_content, selector)
            if content:
                logger.info(
                    "Extracted abstract from %s selector (%d chars)",
                    publisher,
                    len(content),
                )
                return content

        # Fall back to meta tags
        for attr_name, attr_value in meta_tags:
            content = _extract_meta_tag(html_content, attr_name, attr_value)
            if content and len(content) >= 100:
                logger.info(
                    "Extracted abstract from %s meta tag [%s=%s] (%d chars)",
                    publisher,
                    attr_name,
                    attr_value,
                    len(content),
                )
                return _clean_html_text(content)
    else:
        # Default order: meta tags first, then selectors
        for attr_name, attr_value in meta_tags:
            content = _extract_meta_tag(html_content, attr_name, attr_value)
            if content and len(content) >= 100:
                logger.info(
                    "Extracted abstract from %s meta tag [%s=%s] (%d chars)",
                    publisher,
                    attr_name,
                    attr_value,
                    len(content),
                )
                return _clean_html_text(content)

        # Try publisher-specific selectors
        for selector in selectors:
            content = _extract_from_selector(html_content, selector)
            if content:
                logger.info(
                    "Extracted abstract from %s selector (%d chars)",
                    publisher,
                    len(content),
                )
                return content

    # Try generic meta tags
    for attr_name, attr_value in GENERIC_META_TAGS:
        content = _extract_meta_tag(html_content, attr_name, attr_value)
        if content and len(content) >= 100:
            logger.info(
                "Extracted abstract from generic meta tag [%s=%s] (%d chars)",
                attr_name,
                attr_value,
                len(content),
            )
            return _clean_html_text(content)

    # Try generic selectors
    for selector in GENERIC_SELECTORS:
        content = _extract_from_selector(html_content, selector)
        if content:
            logger.info("Extracted abstract from generic selector (%d chars)", len(content))
            return content

    logger.debug("Rule-based extraction failed, will need LLM fallback")
    return None


class PublisherExtractor:
    """Publisher-aware abstract extractor.

    Tries rule-based extraction first, with optional LLM fallback.
    """

    def __init__(self, use_llm_fallback: bool = True):
        """Initialize extractor.

        Args:
            use_llm_fallback: Whether to allow LLM fallback (handled by caller).
        """
        self.use_llm_fallback = use_llm_fallback

    def extract(self, html_content: str, url: str) -> Optional[str]:
        """Extract abstract using rules.

        Args:
            html_content: Raw HTML content.
            url: Page URL.

        Returns:
            Extracted abstract or None.
        """
        return extract_abstract(html_content, url)

    def detect_publisher(self, url: str) -> str:
        """Detect publisher from URL.

        Args:
            url: Page URL.

        Returns:
            Publisher key.
        """
        return detect_publisher(url)


__all__ = [
    "PublisherExtractor",
    "extract_abstract",
    "detect_publisher",
    "PUBLISHER_CONFIGS",
]
