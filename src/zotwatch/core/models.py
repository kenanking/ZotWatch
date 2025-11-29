"""Core domain models for ZotWatch."""

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field


class ZoteroItem(BaseModel):
    """Represents an item from user's Zotero library."""

    key: str
    version: int
    title: str
    abstract: str | None = None
    creators: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    collections: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    date_added: datetime | None = None  # When item was added to Zotero library
    raw: dict[str, object] = Field(default_factory=dict)
    content_hash: str | None = None  # Hash of content used for embedding

    def content_for_embedding(self) -> str:
        """Generate text content for embedding."""
        parts = [self.title]
        if self.abstract:
            parts.append(self.abstract)
        if self.creators:
            parts.append("; ".join(self.creators))
        if self.tags:
            parts.append("; ".join(self.tags))
        return "\n".join(filter(None, parts))

    @classmethod
    def from_zotero_api(cls, item: dict[str, object]) -> "ZoteroItem":
        """Parse item from Zotero API response."""
        data = item.get("data", {})
        creators = [
            " ".join(filter(None, [c.get("firstName"), c.get("lastName")])).strip() for c in data.get("creators", [])
        ]

        # Parse dateAdded (format: "2024-01-15T10:30:00Z")
        date_added = None
        if date_added_str := data.get("dateAdded"):
            try:
                date_added = datetime.fromisoformat(date_added_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        return cls(
            key=data.get("key") or item.get("key"),
            version=data.get("version") or item.get("version", 0),
            title=data.get("title") or "",
            abstract=data.get("abstractNote"),
            creators=[c for c in creators if c],
            tags=[t.get("tag") for t in data.get("tags", []) if isinstance(t, dict)],
            collections=data.get("collections", []),
            year=_safe_int(data.get("date")),
            doi=data.get("DOI"),
            url=data.get("url"),
            date_added=date_added,
            raw=item,
        )


def _safe_int(value: str | None) -> int | None:
    """Safely parse year from date string."""
    if not value:
        return None
    for part in value.split("-"):
        if part.isdigit():
            return int(part)
    return None


class CandidateWork(BaseModel):
    """Represents a candidate paper from external sources."""

    source: str
    identifier: str
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    doi: str | None = None
    url: str | None = None
    published: datetime | None = None
    venue: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    extra: dict[str, object] = Field(default_factory=dict)

    def content_for_embedding(self) -> str:
        """Generate text content for embedding."""
        parts = [self.title]
        if self.abstract:
            parts.append(self.abstract)
        if self.authors:
            parts.append("; ".join(self.authors))
        return "\n".join(filter(None, parts))


class RankedWork(CandidateWork):
    """Extends CandidateWork with scoring information."""

    score: float  # Final combined score (similarity + IF weighted)
    similarity: float  # Embedding similarity
    impact_factor_score: float = 0.0  # Normalized IF score (0-1)
    impact_factor: float | None = None  # Raw IF value (None for arXiv/CN/unknown)
    is_chinese_core: bool = False  # True if Chinese core journal
    label: str  # must_read/consider/ignore
    summary: "PaperSummary | None" = None


class InterestWork(RankedWork):
    """Interest-based paper with rerank score."""

    rerank_score: float


class RefinedInterests(BaseModel):
    """LLM-refined research interests."""

    refined_query: str
    include_keywords: list[str] = Field(default_factory=list)
    exclude_keywords: list[str] = Field(default_factory=list)


@dataclass
class ProfileArtifacts:
    """Paths to profile artifact files."""

    sqlite_path: str
    faiss_path: str


# LLM Summary Models


class BulletSummary(BaseModel):
    """Short bullet-point summary."""

    research_question: str
    methodology: str
    key_findings: str
    innovation: str
    relevance_note: str | None = None


class DetailedAnalysis(BaseModel):
    """Expanded detailed analysis."""

    background: str
    methodology_details: str
    results: str
    limitations: str
    future_directions: str | None = None
    relevance_to_interests: str


class PaperSummary(BaseModel):
    """Complete paper summary with both formats."""

    paper_id: str
    bullets: BulletSummary
    detailed: DetailedAnalysis
    model_used: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    tokens_used: int = 0


class TopicSummary(BaseModel):
    """Summary for a single research topic."""

    topic_name: str  # e.g., "大语言模型"
    paper_count: int
    description: str  # 1-2 sentences describing key research points


class OverallSummary(BaseModel):
    """Overall summary for a section of papers."""

    section_type: str  # "featured" or "similarity"
    overview: str  # First sentence: topic distribution summary
    topics: list[TopicSummary] = Field(default_factory=list)
    paper_count: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: str
    tokens_used: int = 0


# Update forward reference
RankedWork.model_rebuild()


# Researcher Profile Models


class DomainDistribution(BaseModel):
    """Research domain distribution item."""

    domain: str
    paper_count: int
    percentage: float = 0.0
    sample_titles: list[str] = Field(default_factory=list)


class KeywordStats(BaseModel):
    """Keyword frequency statistics."""

    keyword: str
    count: int
    source: str = "tag"  # "tag" or "extracted"


class AuthorStats(BaseModel):
    """Author appearance statistics."""

    author: str
    paper_count: int
    years_active: list[int] = Field(default_factory=list)


class VenueStats(BaseModel):
    """Venue (journal/conference) statistics."""

    venue: str
    paper_count: int
    venue_type: str = "journal"  # "journal" or "conference"


class QuarterlyTrend(BaseModel):
    """Quarterly publication trend."""

    quarter: str  # Format: "2024-Q1"
    paper_count: int
    top_domains: list[str] = Field(default_factory=list)


class RecentPapersAnalysis(BaseModel):
    """Analysis of recently added papers."""

    period_days: int
    paper_count: int
    new_keywords: list[str] = Field(default_factory=list)
    emerging_domains: list[str] = Field(default_factory=list)


class ResearcherProfileInsights(BaseModel):
    """LLM-generated natural language insights."""

    research_focus_summary: str  # Main research directions
    strength_areas: str  # Research strengths
    interdisciplinary_notes: str  # Cross-domain observations
    trend_observations: str  # How interests have evolved
    recommendations: str  # Suggested directions


class ResearcherProfile(BaseModel):
    """Complete researcher profile analysis."""

    # Statistics
    total_papers: int
    year_range: tuple[int, int] = (0, 0)
    collection_duration: str | None = None  # "X年Y月" format
    frequent_author_count: int = 0  # Authors with ≥N appearances

    # Distributions
    domains: list[DomainDistribution] = Field(default_factory=list)
    keywords: list[KeywordStats] = Field(default_factory=list)
    authors: list[AuthorStats] = Field(default_factory=list)
    venues: list[VenueStats] = Field(default_factory=list)
    quarterly_trends: list[QuarterlyTrend] = Field(default_factory=list)

    # Recent activity
    recent_analysis: RecentPapersAnalysis | None = None

    # LLM insights
    insights: ResearcherProfileInsights | None = None

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: str | None = None
    library_hash: str | None = None  # Hash of library state for cache invalidation


__all__ = [
    "ZoteroItem",
    "CandidateWork",
    "RankedWork",
    "InterestWork",
    "RefinedInterests",
    "ProfileArtifacts",
    "BulletSummary",
    "DetailedAnalysis",
    "PaperSummary",
    "TopicSummary",
    "OverallSummary",
    # Researcher Profile Models
    "DomainDistribution",
    "KeywordStats",
    "AuthorStats",
    "VenueStats",
    "QuarterlyTrend",
    "RecentPapersAnalysis",
    "ResearcherProfileInsights",
    "ResearcherProfile",
]
