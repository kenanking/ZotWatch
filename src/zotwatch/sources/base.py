"""Base source definitions and registry."""

import html
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional, Type

from zotwatch.config.settings import Settings
from zotwatch.core.models import CandidateWork

logger = logging.getLogger(__name__)


class BaseSource(ABC):
    """Abstract base class for candidate sources."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique source identifier."""
        ...

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Whether this source is enabled in config."""
        ...

    @abstractmethod
    def fetch(self, days_back: int = 7) -> List[CandidateWork]:
        """Fetch candidates from this source."""
        ...

    def validate_config(self) -> bool:
        """Validate source-specific configuration."""
        return True


class SourceRegistry:
    """Registry for dynamically discovering and loading sources."""

    _sources: Dict[str, Type[BaseSource]] = {}

    @classmethod
    def register(cls, source_class: Type[BaseSource]) -> Type[BaseSource]:
        """Decorator to register a source."""
        # Get name from class
        _ = object.__new__(source_class)
        name = source_class.__name__.lower().replace("source", "")
        cls._sources[name] = source_class
        return source_class

    @classmethod
    def get_source(cls, name: str) -> Optional[Type[BaseSource]]:
        """Get source class by name."""
        return cls._sources.get(name.lower())

    @classmethod
    def get_enabled_sources(cls, settings: Settings) -> List[BaseSource]:
        """Return instantiated sources that are enabled in config."""
        enabled = []
        for name, source_class in cls._sources.items():
            source = source_class(settings)
            if source.enabled:
                enabled.append(source)
        return enabled

    @classmethod
    def all_sources(cls) -> Dict[str, Type[BaseSource]]:
        """Get all registered sources."""
        return cls._sources.copy()


def get_enabled_sources(settings: Settings) -> List[BaseSource]:
    """Convenience function to get enabled sources."""
    return SourceRegistry.get_enabled_sources(settings)


# Helper functions for parsing


def clean_title(value: str | None) -> str:
    """Clean and normalize title string."""
    if not value:
        return ""
    return value.strip()


def ensure_aware(dt: datetime | None) -> datetime | None:
    """Ensure datetime is timezone-aware."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def parse_date(value) -> datetime | None:
    """Parse various date formats to datetime."""
    if not value:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        try:
            return ensure_aware(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            try:
                return ensure_aware(datetime.strptime(value, "%Y-%m-%d"))
            except ValueError:
                return None
    return None


def clean_html(value: str | None) -> str | None:
    """Clean HTML tags from string."""
    if not value:
        return None
    text = html.unescape(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


__all__ = [
    "BaseSource",
    "SourceRegistry",
    "get_enabled_sources",
    "clean_title",
    "ensure_aware",
    "parse_date",
    "clean_html",
]
