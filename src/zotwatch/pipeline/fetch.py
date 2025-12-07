"""Candidate fetching pipeline."""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from pathlib import Path

from zotwatch.config.settings import Settings
from zotwatch.core.constants import DEFAULT_MAX_WORKERS, DEFAULT_TIMEOUT_PER_SOURCE
from zotwatch.core.models import CandidateWork
from zotwatch.sources.base import get_enabled_sources

logger = logging.getLogger(__name__)


def fetch_candidates(settings: Settings) -> list[CandidateWork]:
    """Fetch candidates from all enabled sources (with automatic parallelization).

    When multiple sources are enabled, fetches them concurrently using ThreadPoolExecutor.

    Args:
        settings: Application settings

    Returns:
        List of candidate works from all sources
    """
    sources = list(get_enabled_sources(settings))

    if len(sources) == 0:
        logger.warning("No enabled sources found")
        return []

    # Use parallel fetching when 2+ sources enabled
    if len(sources) >= 2:
        return _fetch_parallel(sources)
    else:
        # Single source: no need for parallelization
        return _fetch_sequential(sources)


def _fetch_sequential(sources: list) -> list[CandidateWork]:
    """Fetch sources sequentially (original behavior).

    Args:
        sources: List of source instances

    Returns:
        Combined list of candidates from all sources
    """
    results: list[CandidateWork] = []

    for source in sources:
        try:
            candidates = source.fetch()
            results.extend(candidates)
            logger.info("Fetched %d candidates from %s", len(candidates), source.name)
        except Exception as exc:
            logger.error("Failed to fetch from %s: %s", source.name, exc)

    logger.info("Fetched %d total candidate works (sequential mode)", len(results))
    return results


def _fetch_parallel(sources: list) -> list[CandidateWork]:
    """Fetch sources in parallel using ThreadPoolExecutor.

    Args:
        sources: List of source instances

    Returns:
        Combined list of candidates from all sources
    """
    results: list[CandidateWork] = []
    errors: dict[str, Exception] = {}

    max_workers = min(len(sources), DEFAULT_MAX_WORKERS)
    logger.info("Fetching from %d sources in parallel (max_workers=%d)", len(sources), max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all fetch tasks
        future_to_source = {executor.submit(source.fetch): source for source in sources}

        # Collect results as they complete
        try:
            for future in as_completed(future_to_source, timeout=DEFAULT_TIMEOUT_PER_SOURCE):
                source = future_to_source[future]
                try:
                    candidates = future.result(timeout=DEFAULT_TIMEOUT_PER_SOURCE)
                    results.extend(candidates)
                    logger.info("Fetched %d candidates from %s", len(candidates), source.name)
                except TimeoutError:
                    error_msg = f"Timeout after {DEFAULT_TIMEOUT_PER_SOURCE}s"
                    errors[source.name] = TimeoutError(error_msg)
                    logger.error("Failed to fetch from %s: %s", source.name, error_msg)
                except Exception as exc:
                    errors[source.name] = exc
                    logger.error("Failed to fetch from %s: %s", source.name, exc)
        except TimeoutError:
            # as_completed() iterator timed out - no futures completed within timeout window
            error_msg = f"as_completed() timed out after {DEFAULT_TIMEOUT_PER_SOURCE}s (no futures completed)"
            logger.error(error_msg)
            # Mark all pending futures as timed out
            for future, source in future_to_source.items():
                if source.name not in errors and not future.done():
                    errors[source.name] = TimeoutError(error_msg)
                    logger.error("Source %s timed out (no result within timeout window)", source.name)

    if errors:
        logger.warning(
            "Parallel fetch completed with %d/%d source failures: %s", len(errors), len(sources), list(errors.keys())
        )

    logger.info("Fetched %d total candidate works (parallel mode)", len(results))
    return results


class CandidateFetcher:
    """Wrapper for candidate fetching."""

    def __init__(self, settings: Settings, base_dir: Path):
        self.settings = settings
        self.base_dir = Path(base_dir)

    def fetch_all(self) -> list[CandidateWork]:
        """Fetch all candidates."""
        return fetch_candidates(self.settings)


__all__ = ["fetch_candidates", "CandidateFetcher"]
