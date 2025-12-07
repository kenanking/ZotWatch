"""SQLite storage implementation."""

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Self

from zotwatch.core.exceptions import ValidationError
from zotwatch.core.models import ClusteredProfile, PaperSummary, ResearcherProfile, ZoteroItem
from zotwatch.utils.datetime import utc_now

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    key TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT,
    creators TEXT,
    tags TEXT,
    collections TEXT,
    year INTEGER,
    doi TEXT,
    url TEXT,
    raw_json TEXT NOT NULL,
    content_hash TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS summaries (
    paper_id TEXT PRIMARY KEY,
    bullets_json TEXT NOT NULL,
    detailed_json TEXT NOT NULL,
    model_used TEXT NOT NULL,
    tokens_used INTEGER DEFAULT 0,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS profile_analysis (
    id INTEGER PRIMARY KEY DEFAULT 1,
    library_hash TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS title_translations (
    paper_id TEXT PRIMARY KEY,
    original_title TEXT NOT NULL,
    translated_title TEXT NOT NULL,
    target_language TEXT NOT NULL,
    model_used TEXT NOT NULL,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS clustered_profile (
    id INTEGER PRIMARY KEY DEFAULT 1,
    embedding_signature TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_items_version ON items(version);
CREATE INDEX IF NOT EXISTS idx_items_content_hash ON items(content_hash);
CREATE INDEX IF NOT EXISTS idx_summaries_expires ON summaries(expires_at);
CREATE INDEX IF NOT EXISTS idx_profile_hash ON profile_analysis(library_hash);
"""


class ProfileStorage:
    """SQLite storage for profile data.

    Note: Embedding storage has been moved to EmbeddingCache.
    This class now focuses on item metadata and summaries.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def initialize(self) -> None:
        """Initialize database schema."""
        conn = self.connect()
        conn.executescript(SCHEMA)
        conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager and close resources."""
        self.close()

    # Metadata helpers

    def get_metadata(self, key: str) -> str | None:
        """Get metadata value by key."""
        cur = self.connect().execute("SELECT value FROM metadata WHERE key = ?", (key,))
        row = cur.fetchone()
        return row["value"] if row else None

    def set_metadata(self, key: str, value: str) -> None:
        """Set metadata value."""
        self.connect().execute(
            "REPLACE INTO metadata(key, value) VALUES(?, ?)",
            (key, value),
        )
        self.connect().commit()

    def last_modified_version(self) -> int | None:
        """Get last modified version from Zotero sync."""
        value = self.get_metadata("last_modified_version")
        return int(value) if value else None

    def set_last_modified_version(self, version: int) -> None:
        """Set last modified version."""
        self.set_metadata("last_modified_version", str(version))

    # Item helpers

    def upsert_item(self, item: ZoteroItem, content_hash: str | None = None) -> None:
        """Insert or update item."""
        data = (
            item.key,
            item.version,
            item.title,
            item.abstract,
            json.dumps(item.creators),
            json.dumps(item.tags),
            json.dumps(item.collections),
            item.year,
            item.doi,
            item.url,
            json.dumps(item.raw),
            content_hash,
        )
        self.connect().execute(
            """
            INSERT INTO items(
                key, version, title, abstract, creators, tags, collections, year, doi, url, raw_json, content_hash
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                version=excluded.version,
                title=excluded.title,
                abstract=excluded.abstract,
                creators=excluded.creators,
                tags=excluded.tags,
                collections=excluded.collections,
                year=excluded.year,
                doi=excluded.doi,
                url=excluded.url,
                raw_json=excluded.raw_json,
                content_hash=excluded.content_hash,
                updated_at=CURRENT_TIMESTAMP
            """,
            data,
        )
        self.connect().commit()

    def remove_items(self, keys: Iterable[str]) -> None:
        """Remove items by keys."""
        keys = list(keys)
        if not keys:
            return
        placeholders = ",".join("?" for _ in keys)
        self.connect().execute(f"DELETE FROM items WHERE key IN ({placeholders})", keys)
        self.connect().commit()

    def iter_items(self) -> Iterable[ZoteroItem]:
        """Iterate over all items."""
        cur = self.connect().execute("SELECT * FROM items")
        for row in cur:
            yield _row_to_item(row)

    def get_item(self, key: str) -> ZoteroItem | None:
        """Get item by key."""
        cur = self.connect().execute("SELECT * FROM items WHERE key = ?", (key,))
        row = cur.fetchone()
        return _row_to_item(row) if row else None

    def get_all_items(self) -> list[ZoteroItem]:
        """Get all items as a list."""
        return list(self.iter_items())

    def get_items_with_abstract(self) -> list[ZoteroItem]:
        """Get items that have non-empty abstracts (used for profile/index builds)."""
        # Stable order by key to keep embedding/weight alignment deterministic
        cur = self.connect().execute(
            "SELECT * FROM items WHERE abstract IS NOT NULL AND TRIM(abstract) != '' ORDER BY key"
        )
        return [_row_to_item(row) for row in cur]

    def count_items(self) -> int:
        """Count total items."""
        cur = self.connect().execute("SELECT COUNT(*) FROM items")
        return cur.fetchone()[0]

    def get_all_content_hashes(self) -> dict[str, str]:
        """Get mapping of item keys to content hashes.

        Used by EmbeddingCache to determine which items need re-embedding.
        """
        cur = self.connect().execute("SELECT key, content_hash FROM items WHERE content_hash IS NOT NULL")
        return {row["key"]: row["content_hash"] for row in cur}

    # Summary helpers

    def get_summary(self, paper_id: str) -> PaperSummary | None:
        """Get cached summary by paper ID."""
        cur = self.connect().execute(
            "SELECT * FROM summaries WHERE paper_id = ?",
            (paper_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return _row_to_summary(row)

    def save_summary(self, paper_id: str, summary: PaperSummary) -> None:
        """Save summary to cache."""
        self.connect().execute(
            """
            INSERT INTO summaries(paper_id, bullets_json, detailed_json, model_used, tokens_used, generated_at)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                bullets_json=excluded.bullets_json,
                detailed_json=excluded.detailed_json,
                model_used=excluded.model_used,
                tokens_used=excluded.tokens_used,
                generated_at=excluded.generated_at
            """,
            (
                paper_id,
                summary.bullets.model_dump_json(),
                summary.detailed.model_dump_json(),
                summary.model_used,
                summary.tokens_used,
                summary.generated_at.isoformat(),
            ),
        )
        self.connect().commit()

    def has_summary(self, paper_id: str) -> bool:
        """Check if summary exists."""
        cur = self.connect().execute(
            "SELECT 1 FROM summaries WHERE paper_id = ?",
            (paper_id,),
        )
        return cur.fetchone() is not None

    # Profile analysis helpers

    def get_profile_analysis(self, library_hash: str) -> ResearcherProfile | None:
        """Get cached profile analysis if hash matches.

        Args:
            library_hash: Hash of the current library state.

        Returns:
            Cached ResearcherProfile if hash matches, None otherwise.
        """
        cur = self.connect().execute(
            "SELECT profile_json FROM profile_analysis WHERE library_hash = ?",
            (library_hash,),
        )
        row = cur.fetchone()
        if row:
            return ResearcherProfile.model_validate_json(row["profile_json"])
        return None

    def save_profile_analysis(self, profile: ResearcherProfile) -> None:
        """Save profile analysis to cache.

        Args:
            profile: ResearcherProfile to cache.
        """
        if not profile.library_hash:
            raise ValidationError("Profile must have library_hash set for caching")

        self.connect().execute(
            """
            INSERT OR REPLACE INTO profile_analysis (id, library_hash, profile_json, generated_at)
            VALUES (1, ?, ?, CURRENT_TIMESTAMP)
            """,
            (profile.library_hash, profile.model_dump_json()),
        )
        self.connect().commit()

    def clear_profile_cache(self) -> None:
        """Clear profile analysis cache."""
        self.connect().execute("DELETE FROM profile_analysis")
        self.connect().commit()

    # Clustered profile helpers

    def get_clustered_profile(self, embedding_signature: str) -> ClusteredProfile | None:
        """Get cached clustered profile if signature matches.

        Args:
            embedding_signature: Embedding provider and model signature
                                 (e.g., "voyage:voyage-3.5").

        Returns:
            Cached ClusteredProfile if signature matches, None otherwise.
        """
        cur = self.connect().execute(
            "SELECT profile_json FROM clustered_profile WHERE embedding_signature = ?",
            (embedding_signature,),
        )
        row = cur.fetchone()
        if row:
            return ClusteredProfile.model_validate_json(row["profile_json"])
        return None

    def save_clustered_profile(self, profile: ClusteredProfile) -> None:
        """Save clustered profile to cache.

        Args:
            profile: ClusteredProfile to cache.
        """
        if not profile.embedding_signature:
            raise ValidationError("ClusteredProfile must have embedding_signature set for caching")

        self.connect().execute(
            """
            INSERT OR REPLACE INTO clustered_profile (id, embedding_signature, profile_json, generated_at)
            VALUES (1, ?, ?, CURRENT_TIMESTAMP)
            """,
            (profile.embedding_signature, profile.model_dump_json()),
        )
        self.connect().commit()

    def clear_clustered_profile_cache(self) -> None:
        """Clear clustered profile cache."""
        self.connect().execute("DELETE FROM clustered_profile")
        self.connect().commit()

    # Translation cache helpers

    def get_translation(self, paper_id: str, target_language: str) -> str | None:
        """Get cached translation by paper ID and target language."""
        cur = self.connect().execute(
            "SELECT translated_title FROM title_translations WHERE paper_id = ? AND target_language = ?",
            (paper_id, target_language),
        )
        row = cur.fetchone()
        return row["translated_title"] if row else None

    def save_translation(
        self,
        paper_id: str,
        original: str,
        translated: str,
        target_language: str,
        model: str,
    ) -> None:
        """Save translation to cache."""
        self.connect().execute(
            """
            INSERT INTO title_translations(paper_id, original_title, translated_title, target_language, model_used)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                original_title=excluded.original_title,
                translated_title=excluded.translated_title,
                target_language=excluded.target_language,
                model_used=excluded.model_used,
                generated_at=CURRENT_TIMESTAMP
            """,
            (paper_id, original, translated, target_language, model),
        )
        self.connect().commit()

    def get_translations_batch(self, paper_ids: list[str], target_language: str) -> dict[str, str]:
        """Get multiple cached translations at once."""
        if not paper_ids:
            return {}
        placeholders = ",".join("?" for _ in paper_ids)
        cur = self.connect().execute(
            f"SELECT paper_id, translated_title FROM title_translations WHERE paper_id IN ({placeholders}) AND target_language = ?",
            (*paper_ids, target_language),
        )
        return {row["paper_id"]: row["translated_title"] for row in cur}

    def save_translations_batch(
        self,
        translations: list[dict],
        target_language: str,
        model: str,
    ) -> None:
        """Save multiple translations at once.

        Args:
            translations: List of dicts with keys: paper_id, original, translated
            target_language: Target language code
            model: Model used for translation
        """
        if not translations:
            return
        conn = self.connect()
        conn.executemany(
            """
            INSERT INTO title_translations(paper_id, original_title, translated_title, target_language, model_used)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                original_title=excluded.original_title,
                translated_title=excluded.translated_title,
                target_language=excluded.target_language,
                model_used=excluded.model_used,
                generated_at=CURRENT_TIMESTAMP
            """,
            [(t["paper_id"], t["original"], t["translated"], target_language, model) for t in translations],
        )
        conn.commit()


def _row_to_item(row: sqlite3.Row) -> ZoteroItem:
    """Convert database row to ZoteroItem."""
    from datetime import datetime

    raw = json.loads(row["raw_json"])

    # Parse dateAdded from raw data
    date_added = None
    if raw_data := raw.get("data"):
        if date_added_str := raw_data.get("dateAdded"):
            try:
                date_added = datetime.fromisoformat(date_added_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

    return ZoteroItem(
        key=row["key"],
        version=row["version"],
        title=row["title"],
        abstract=row["abstract"],
        creators=json.loads(row["creators"] or "[]"),
        tags=json.loads(row["tags"] or "[]"),
        collections=json.loads(row["collections"] or "[]"),
        year=row["year"],
        doi=row["doi"],
        url=row["url"],
        date_added=date_added,
        raw=raw,
        content_hash=row["content_hash"],
    )


def _row_to_summary(row: sqlite3.Row) -> PaperSummary:
    """Convert database row to PaperSummary."""
    from datetime import datetime

    from zotwatch.core.models import BulletSummary, DetailedAnalysis

    return PaperSummary(
        paper_id=row["paper_id"],
        bullets=BulletSummary.model_validate_json(row["bullets_json"]),
        detailed=DetailedAnalysis.model_validate_json(row["detailed_json"]),
        model_used=row["model_used"],
        tokens_used=row["tokens_used"],
        generated_at=datetime.fromisoformat(row["generated_at"]) if row["generated_at"] else utc_now(),
    )


__all__ = ["ProfileStorage"]
