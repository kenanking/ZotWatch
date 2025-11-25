"""Crossref source implementation."""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import requests

from zotwatch.config.settings import Settings
from zotwatch.core.models import CandidateWork

from .base import BaseSource, SourceRegistry, clean_html, clean_title, parse_date

logger = logging.getLogger(__name__)


@SourceRegistry.register
class CrossrefSource(BaseSource):
    """Crossref journal articles source."""

    def __init__(self, settings: Settings, profile_path: Optional[Path] = None):
        super().__init__(settings)
        self.config = settings.sources.crossref
        self.session = requests.Session()
        self.profile_path = profile_path
        self._top_venues: Optional[List[str]] = None

    @property
    def name(self) -> str:
        return "crossref"

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def top_venues(self) -> List[str]:
        """Load top venues from profile."""
        if self._top_venues is not None:
            return self._top_venues

        if not self.profile_path or not self.profile_path.exists():
            self._top_venues = []
            return self._top_venues

        try:
            data = json.loads(self.profile_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load profile when reading top venues: %s", exc)
            self._top_venues = []
            return self._top_venues

        venues: List[str] = []
        for entry in data.get("top_venues", []):
            name = entry.get("venue") if isinstance(entry, dict) else None
            if name:
                venues.append(name)

        unique = list(dict.fromkeys(venues)) if venues else []
        if unique:
            logger.info("Loaded %d top venues from profile", len(unique))
        self._top_venues = unique[:20]
        return self._top_venues

    def set_profile_path(self, path: Path) -> None:
        """Set profile path for top venues loading."""
        self.profile_path = path
        self._top_venues = None

    def fetch(self, days_back: int | None = None) -> List[CandidateWork]:
        """Fetch Crossref works."""
        if days_back is None:
            days_back = self.config.days_back

        results = self._fetch_general(days_back)
        results.extend(self._fetch_top_venues(days_back))
        return results

    def _fetch_general(self, days_back: int) -> List[CandidateWork]:
        """Fetch general Crossref works."""
        since = datetime.now(timezone.utc) - timedelta(days=days_back)
        url = "https://api.crossref.org/works"
        params = {
            "filter": f"from-pub-date:{since.date().isoformat()}",
            "sort": "created",
            "order": "desc",
            "rows": 200,
            "mailto": self.config.mailto,
            "select": "DOI,title,author,abstract,container-title,created,URL,type,is-referenced-by-count",
        }

        logger.info("Fetching Crossref works since %s", since.date())
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        message = resp.json().get("message", {})

        results = []
        for item in message.get("items", []):
            work = self._parse_crossref_item(item)
            if work:
                results.append(work)

        logger.info("Fetched %d Crossref works", len(results))
        return results

    def _fetch_top_venues(self, days_back: int) -> List[CandidateWork]:
        """Fetch works from top venues."""
        if not self.top_venues:
            return []

        since = datetime.now(timezone.utc) - timedelta(days=days_back)
        results: List[CandidateWork] = []

        for venue in self.top_venues:
            params = {
                "filter": f"from-pub-date:{since.date().isoformat()},container-title:{venue}",
                "sort": "created",
                "order": "desc",
                "rows": 100,
                "mailto": self.config.mailto,
                "select": "DOI,title,author,abstract,container-title,created,URL,type,is-referenced-by-count",
            }
            try:
                resp = self.session.get(
                    "https://api.crossref.org/works",
                    params=params,
                    timeout=30,
                )
                resp.raise_for_status()
            except Exception as exc:
                logger.warning("Failed to fetch Crossref top venue %s: %s", venue, exc)
                continue

            message = resp.json().get("message", {})
            for item in message.get("items", []):
                work = self._parse_crossref_item(item, venue_override=venue)
                if work:
                    work.extra["source"] = "top_venue"
                    results.append(work)

        if results:
            logger.info("Fetched %d additional works from top venues", len(results))
        return results

    def _parse_crossref_item(
        self,
        item: dict,
        venue_override: Optional[str] = None,
    ) -> Optional[CandidateWork]:
        """Parse Crossref API item to CandidateWork."""
        title = clean_title((item.get("title") or [""])[0])
        if not title:
            return None

        doi = item.get("DOI")
        authors = [" ".join(filter(None, [p.get("given"), p.get("family")])).strip() for p in item.get("author", [])]

        return CandidateWork(
            source="crossref",
            identifier=doi or item.get("URL", "unknown"),
            title=title,
            abstract=clean_html(item.get("abstract")),
            authors=[a for a in authors if a],
            doi=doi,
            url=item.get("URL"),
            published=parse_date(item.get("created", {}).get("date-time")),
            venue=venue_override or (item.get("container-title") or [None])[0],
            metrics={"is-referenced-by": float(item.get("is-referenced-by-count", 0))},
            extra={"type": item.get("type")},
        )


__all__ = ["CrossrefSource"]
