"""Zotero ingestion pipeline."""

import logging

from zotwatch.config.settings import Settings
from zotwatch.infrastructure.storage import ProfileStorage
from zotwatch.sources.zotero import IngestStats, ZoteroIngestor

logger = logging.getLogger(__name__)


def ingest_zotero(
    storage: ProfileStorage,
    settings: Settings,
    *,
    full: bool = False,
) -> IngestStats:
    """Ingest items from Zotero library.

    Args:
        storage: Profile storage instance
        settings: Application settings
        full: If True, perform full rebuild; otherwise incremental sync

    Returns:
        IngestStats with operation statistics
    """
    ingestor = ZoteroIngestor(storage, settings)
    stats = ingestor.run(full=full)

    logger.info(
        "Ingest stats: fetched=%d updated=%d removed=%d",
        stats.fetched,
        stats.updated,
        stats.removed,
    )

    return stats


__all__ = ["ingest_zotero", "IngestStats"]
