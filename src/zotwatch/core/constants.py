"""Core constants for ZotWatch."""

# Network timeouts (seconds)
DEFAULT_HTTP_TIMEOUT = 30.0
DEFAULT_LLM_TIMEOUT = 60.0

# Pagination limits
ZOTERO_API_PAGE_SIZE = 100
CROSSREF_API_PAGE_SIZE = 200

# Embedding dimensions
VOYAGE_EMBEDDING_DIM = 1024
DASHSCOPE_EMBEDDING_DIM = 1024

# Cache TTL defaults (days)
DEFAULT_CACHE_TTL_DAYS = 30

# Minimum text lengths for validation
MIN_ABSTRACT_LENGTH = 100
MIN_CONTENT_LENGTH_FOR_LLM = 200

# Parallel fetching configuration
DEFAULT_MAX_WORKERS = 5  # Max concurrent source fetches
DEFAULT_TIMEOUT_PER_SOURCE = 300  # 5 minutes per source (seconds)

__all__ = [
    "DEFAULT_HTTP_TIMEOUT",
    "DEFAULT_LLM_TIMEOUT",
    "ZOTERO_API_PAGE_SIZE",
    "CROSSREF_API_PAGE_SIZE",
    "VOYAGE_EMBEDDING_DIM",
    "DASHSCOPE_EMBEDDING_DIM",
    "DEFAULT_CACHE_TTL_DAYS",
    "MIN_ABSTRACT_LENGTH",
    "MIN_CONTENT_LENGTH_FOR_LLM",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_TIMEOUT_PER_SOURCE",
]
