"""Core domain models and interfaces."""

from .exceptions import (
    CacheError,
    ConfigurationError,
    EmbeddingError,
    LLMError,
    LLMRateLimitError,
    NetworkError,
    ProfileBuildError,
    RateLimitError,
    SourceFetchError,
    StorageError,
    ValidationError,
    ZotWatchError,
)
from .models import (
    BulletSummary,
    CandidateWork,
    DetailedAnalysis,
    PaperSummary,
    ProfileArtifacts,
    RankedWork,
    ZoteroItem,
)
from .protocols import (
    ItemStorage,
    LLMResponse,
    SummaryStorage,
)

__all__ = [
    # Models
    "ZoteroItem",
    "CandidateWork",
    "RankedWork",
    "ProfileArtifacts",
    "BulletSummary",
    "DetailedAnalysis",
    "PaperSummary",
    # Protocols
    "LLMResponse",
    "ItemStorage",
    "SummaryStorage",
    # Exceptions
    "ZotWatchError",
    "ConfigurationError",
    "ValidationError",
    "NetworkError",
    "RateLimitError",
    "SourceFetchError",
    "EmbeddingError",
    "LLMError",
    "LLMRateLimitError",
    "StorageError",
    "CacheError",
    "ProfileBuildError",
]
