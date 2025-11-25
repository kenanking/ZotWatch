"""OpenAlex source implementation."""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import requests

from zotwatch.config.settings import Settings
from zotwatch.core.models import CandidateWork

from .base import BaseSource, SourceRegistry, clean_title, parse_date

logger = logging.getLogger(__name__)


@SourceRegistry.register
class OpenAlexSource(BaseSource):
    """OpenAlex scholarly works source."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.config = settings.sources.openalex
        self.session = requests.Session()

    @property
    def name(self) -> str:
        return "openalex"

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def fetch(self, days_back: int | None = None) -> List[CandidateWork]:
        """Fetch OpenAlex works."""
        if days_back is None:
            days_back = self.config.days_back

        since = datetime.now(timezone.utc) - timedelta(days=days_back)
        url = "https://api.openalex.org/works"
        params = {
            "filter": f"from_publication_date:{since.date().isoformat()}",
            "sort": "publication_date:desc",
            "per-page": 200,
            "mailto": self.config.mailto,
        }

        logger.info("Fetching OpenAlex works since %s", since.date())
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("results", []):
            work = self._parse_openalex_item(item)
            if work:
                results.append(work)

        logger.info("Fetched %d OpenAlex works", len(results))
        return results

    def _parse_openalex_item(self, item: dict) -> Optional[CandidateWork]:
        """Parse OpenAlex API item to CandidateWork."""
        title = clean_title(item.get("display_name"))
        if not title:
            return None

        work_id = item.get("id") or item.get("ids", {}).get("openalex")
        primary_location = item.get("primary_location") or {}
        source_info = primary_location.get("source") or {}
        landing_page = primary_location.get("landing_page_url")

        return CandidateWork(
            source="openalex",
            identifier=work_id or item.get("doi") or title,
            title=title,
            abstract=_extract_openalex_abstract(item),
            authors=[auth.get("author", {}).get("display_name", "") for auth in item.get("authorships", [])],
            doi=item.get("doi"),
            url=source_info.get("url") or landing_page,
            published=parse_date(item.get("publication_date")),
            venue=source_info.get("display_name"),
            metrics={"cited_by": float(item.get("cited_by_count", 0))},
            extra={"concepts": [c.get("display_name") for c in item.get("concepts", [])]},
        )


def _extract_openalex_abstract(item: dict) -> str | None:
    """Extract abstract from OpenAlex item."""
    abstract = item.get("abstract")
    if isinstance(abstract, dict):
        text = abstract.get("text")
        if text:
            return text
    if isinstance(abstract, str) and abstract.strip():
        return abstract.strip()

    # Handle inverted index format
    inverted = item.get("abstract_inverted_index")
    if isinstance(inverted, dict) and inverted:
        try:
            size = max(pos for positions in inverted.values() for pos in positions) + 1
        except ValueError:
            size = 0
        tokens = ["" for _ in range(size)]
        for word, positions in inverted.items():
            for pos in positions:
                if 0 <= pos < size:
                    tokens[pos] = word
        summary = " ".join(filter(None, tokens)).strip()
        return summary or None

    return None


__all__ = ["OpenAlexSource"]
