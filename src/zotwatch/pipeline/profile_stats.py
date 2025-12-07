"""Profile statistics extraction from Zotero library."""

import hashlib
import json
import logging
from collections import Counter
from datetime import datetime, timedelta

from zotwatch.core.models import (
    AuthorStats,
    KeywordStats,
    QuarterlyTrend,
    RecentPapersAnalysis,
    ResearcherProfile,
    VenueStats,
    YearDistribution,
    ZoteroItem,
)
from zotwatch.utils.datetime import ensure_aware, utc_now

logger = logging.getLogger(__name__)

# Keywords that indicate a conference venue
CONFERENCE_KEYWORDS = frozenset(
    {
        "conference",
        "symposium",
        "workshop",
        "proceedings",
        "annual meeting",
        "congress",
    }
)


class ProfileStatsExtractor:
    """Extract statistics from user's Zotero library."""

    def __init__(self, years_back: int = 3, recent_days: int = 30):
        """Initialize the extractor.

        Args:
            years_back: Number of years to include in quarterly trends.
            recent_days: Number of days to consider for "recent" analysis.
        """
        self.years_back = years_back
        self.recent_days = recent_days

    def compute_library_hash(self, items: list[ZoteroItem]) -> str:
        """Compute hash of library state for cache invalidation.

        Uses item keys and versions to detect changes.
        """
        content = json.dumps(
            sorted([f"{i.key}:{i.version}" for i in items]),
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def extract_all(
        self,
        items: list[ZoteroItem],
        author_min_count: int = 10,
    ) -> ResearcherProfile:
        """Extract all statistics from library items.

        Args:
            items: List of Zotero items to analyze.
            author_min_count: Minimum appearances for "frequent author" count.

        Returns:
            ResearcherProfile with extracted statistics (without LLM insights).
        """
        if not items:
            return ResearcherProfile(
                total_papers=0,
                year_range=(0, 0),
                generated_at=utc_now(),
            )

        years = [i.year for i in items if i.year]
        year_range = (min(years), max(years)) if years else (0, 0)

        library_hash = self.compute_library_hash(items)
        frequent_author_count = self.count_frequent_authors(items, author_min_count)
        collection_duration = self._calculate_collection_duration(items)

        return ResearcherProfile(
            total_papers=len(items),
            year_range=year_range,
            collection_duration=collection_duration,
            frequent_author_count=frequent_author_count,
            keywords=self._extract_keywords(items),
            authors=self._extract_authors(items),
            venues=self._extract_venues(items),
            quarterly_trends=self._extract_quarterly_trends(items),
            year_distribution=self._extract_year_distribution(items),
            recent_analysis=self._analyze_recent(items),
            generated_at=utc_now(),
            library_hash=library_hash,
        )

    def _extract_keywords(
        self,
        items: list[ZoteroItem],
        top_n: int = 50,
    ) -> list[KeywordStats]:
        """Extract keyword frequency from tags.

        Args:
            items: Zotero items to analyze.
            top_n: Maximum number of keywords to return.

        Returns:
            List of keyword statistics sorted by frequency.
        """
        tag_counter: Counter[str] = Counter()

        for item in items:
            for tag in item.tags:
                # Filter empty or single-char tags
                if tag and len(tag.strip()) > 1:
                    # Normalize: strip whitespace, keep original case for display
                    normalized = tag.strip()
                    tag_counter[normalized] += 1

        return [KeywordStats(keyword=kw, count=cnt, source="tag") for kw, cnt in tag_counter.most_common(top_n)]

    def count_frequent_authors(
        self,
        items: list[ZoteroItem],
        min_count: int = 10,
    ) -> int:
        """Count authors appearing at least min_count times.

        Args:
            items: Zotero items to analyze.
            min_count: Minimum appearance threshold.

        Returns:
            Number of authors with at least min_count appearances.
        """
        author_counts: Counter[str] = Counter()

        for item in items:
            for author in item.creators:
                if author and author.strip():
                    author_counts[author.strip()] += 1

        return sum(1 for count in author_counts.values() if count >= min_count)

    def _extract_authors(
        self,
        items: list[ZoteroItem],
        top_n: int = 30,
    ) -> list[AuthorStats]:
        """Extract author frequency and activity years.

        Args:
            items: Zotero items to analyze.
            top_n: Maximum number of authors to return.

        Returns:
            List of author statistics sorted by paper count.
        """
        author_data: dict[str, dict] = {}

        for item in items:
            year = item.year
            for author in item.creators:
                if not author or not author.strip():
                    continue

                author_name = author.strip()
                if author_name not in author_data:
                    author_data[author_name] = {"count": 0, "years": set()}

                author_data[author_name]["count"] += 1
                if year:
                    author_data[author_name]["years"].add(year)

        # Sort by count descending
        sorted_authors = sorted(
            author_data.items(),
            key=lambda x: x[1]["count"],
            reverse=True,
        )[:top_n]

        return [
            AuthorStats(
                author=name,
                paper_count=data["count"],
                years_active=sorted(data["years"]),
            )
            for name, data in sorted_authors
        ]

    def _extract_venues(
        self,
        items: list[ZoteroItem],
        top_n: int = 20,
    ) -> list[VenueStats]:
        """Extract venue (journal/conference) frequency.

        Args:
            items: Zotero items to analyze.
            top_n: Maximum number of venues to return.

        Returns:
            List of venue statistics sorted by paper count.
        """
        venue_counter: Counter[str] = Counter()

        for item in items:
            # Try to extract venue from raw data
            raw_data = item.raw.get("data", {})
            venue = (
                raw_data.get("publicationTitle")
                or raw_data.get("conferenceName")
                or raw_data.get("proceedingsTitle")
                or raw_data.get("journalAbbreviation")
            )
            if venue and venue.strip():
                venue_counter[venue.strip()] += 1

        # Determine venue type based on keywords
        results: list[VenueStats] = []
        for venue, count in venue_counter.most_common(top_n):
            venue_lower = venue.lower()
            is_conference = any(kw in venue_lower for kw in CONFERENCE_KEYWORDS)
            results.append(
                VenueStats(
                    venue=venue,
                    paper_count=count,
                    venue_type="conference" if is_conference else "journal",
                )
            )

        return results

    def _extract_quarterly_trends(
        self,
        items: list[ZoteroItem],
    ) -> list[QuarterlyTrend]:
        """Extract quarterly publication trends for last N years.

        Args:
            items: Zotero items to analyze.

        Returns:
            List of quarterly trends sorted chronologically.
        """
        current_year = datetime.now().year
        start_year = current_year - self.years_back + 1

        # Group items by quarter
        quarter_counts: Counter[str] = Counter()

        for item in items:
            if not item.year or item.year < start_year:
                continue

            # Try to get month from raw date
            raw_data = item.raw.get("data", {})
            date_str = raw_data.get("date", "")
            quarter = self._get_quarter(item.year, date_str)
            quarter_counts[quarter] += 1

        # Generate all quarters in range (even if count is 0)
        all_quarters: list[QuarterlyTrend] = []
        for year in range(start_year, current_year + 1):
            for q in range(1, 5):
                quarter_key = f"{year}-Q{q}"
                # Don't include future quarters
                if year == current_year and q > (datetime.now().month - 1) // 3 + 1:
                    break
                all_quarters.append(
                    QuarterlyTrend(
                        quarter=quarter_key,
                        paper_count=quarter_counts.get(quarter_key, 0),
                        top_domains=[],  # Will be filled by LLM if needed
                    )
                )

        return all_quarters

    def _extract_year_distribution(
        self,
        items: list[ZoteroItem],
        years_back: int = 20,
    ) -> list[YearDistribution]:
        """Extract paper count by publication year for the last N years.

        Args:
            items: Zotero items to analyze.
            years_back: Number of years to include (default: 20).

        Returns:
            List of year distribution sorted chronologically.
        """
        current_year = datetime.now().year
        start_year = current_year - years_back + 1

        # Count papers by year
        year_counts: Counter[int] = Counter()
        for item in items:
            if item.year and item.year >= start_year:
                year_counts[item.year] += 1

        # Generate all years in range (including zero counts)
        return [
            YearDistribution(year=year, paper_count=year_counts.get(year, 0))
            for year in range(start_year, current_year + 1)
        ]

    def _get_quarter(self, year: int, date_str: str) -> str:
        """Determine quarter from year and date string.

        Args:
            year: Publication year.
            date_str: Date string from Zotero (e.g., "2024-03-15").

        Returns:
            Quarter string (e.g., "2024-Q1").
        """
        month = 1  # Default to Q1 if month unknown

        if date_str:
            parts = date_str.split("-")
            if len(parts) >= 2:
                try:
                    month = int(parts[1])
                except ValueError:
                    pass

        quarter = (month - 1) // 3 + 1
        return f"{year}-Q{quarter}"

    def _analyze_recent(
        self,
        items: list[ZoteroItem],
    ) -> RecentPapersAnalysis:
        """Analyze recently added papers using dateAdded when available.

        Falls back to publication year filtering when dateAdded is missing.
        """
        cutoff = utc_now() - timedelta(days=self.recent_days)
        current_year = datetime.now().year

        recent_items: list[ZoteroItem] = []
        for item in items:
            if item.date_added:
                added = ensure_aware(item.date_added)
                if added and added >= cutoff:
                    recent_items.append(item)
            elif item.year and item.year >= current_year:
                recent_items.append(item)

        # Find keywords that appear only/mainly in recent papers
        recent_tags: Counter[str] = Counter()
        all_tags: Counter[str] = Counter()

        for item in items:
            for tag in item.tags:
                if tag and len(tag.strip()) > 1:
                    normalized = tag.strip()
                    all_tags[normalized] += 1
                    if item in recent_items:
                        recent_tags[normalized] += 1

        # Find "new" keywords (only appear in recent papers)
        new_keywords = [tag for tag, cnt in recent_tags.most_common(10) if cnt > 0 and all_tags[tag] == cnt][:5]

        return RecentPapersAnalysis(
            period_days=self.recent_days,
            paper_count=len(recent_items),
            new_keywords=new_keywords,
            emerging_domains=[],  # Will be filled by LLM
        )

    def _calculate_collection_duration(self, items: list[ZoteroItem]) -> str | None:
        """Calculate collection duration as 'X年Y月' format.

        Args:
            items: Zotero items with date_added field.

        Returns:
            Duration string in 'X年Y月' format, or None if no date_added data.
        """
        dates = [i.date_added for i in items if i.date_added]
        if not dates:
            return None

        earliest = min(dates)
        latest = max(dates)

        # Calculate difference in months
        total_months = (latest.year - earliest.year) * 12 + (latest.month - earliest.month)
        years = total_months // 12
        months = total_months % 12

        if years == 0:
            return f"{months}月"
        elif months == 0:
            return f"{years}年"
        else:
            return f"{years}年{months}月"
