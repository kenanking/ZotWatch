"""Data source implementations."""

from .arxiv import ArxivSource
from .base import SourceRegistry, get_enabled_sources
from .crossref import CrossrefSource
from .zotero import ZoteroClient, ZoteroIngestor

__all__ = [
    "SourceRegistry",
    "get_enabled_sources",
    "ArxivSource",
    "CrossrefSource",
    "ZoteroClient",
    "ZoteroIngestor",
]
