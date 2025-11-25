"""SQLite storage implementation."""

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional

from zotwatch.core.models import PaperSummary, ZoteroItem


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

CREATE INDEX IF NOT EXISTS idx_items_version ON items(version);
CREATE INDEX IF NOT EXISTS idx_items_content_hash ON items(content_hash);
CREATE INDEX IF NOT EXISTS idx_summaries_expires ON summaries(expires_at);
"""


class ProfileStorage:
    """SQLite storage for profile data.

    Note: Embedding storage has been moved to EmbeddingCache.
    This class now focuses on item metadata and summaries.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

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
        # Migrate: drop old embedding columns if they exist
        self._migrate_drop_embedding_columns()
        conn.commit()

    def _migrate_drop_embedding_columns(self) -> None:
        """Drop deprecated embedding columns from items table."""
        conn = self.connect()
        # Check if old columns exist
        cur = conn.execute("PRAGMA table_info(items)")
        columns = {row["name"] for row in cur.fetchall()}

        if "embedding" in columns or "embedding_hash" in columns:
            # SQLite doesn't support DROP COLUMN directly in older versions
            # We need to recreate the table
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS items_new (
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

                INSERT INTO items_new (key, version, title, abstract, creators, tags, collections, year, doi, url, raw_json, content_hash, updated_at)
                SELECT key, version, title, abstract, creators, tags, collections, year, doi, url, raw_json, content_hash, updated_at
                FROM items;

                DROP TABLE items;

                ALTER TABLE items_new RENAME TO items;

                CREATE INDEX IF NOT EXISTS idx_items_version ON items(version);
                CREATE INDEX IF NOT EXISTS idx_items_content_hash ON items(content_hash);
            """)

        # Drop old candidate_embeddings table if exists
        conn.execute("DROP TABLE IF EXISTS candidate_embeddings")

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # Metadata helpers

    def get_metadata(self, key: str) -> Optional[str]:
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

    def last_modified_version(self) -> Optional[int]:
        """Get last modified version from Zotero sync."""
        value = self.get_metadata("last_modified_version")
        return int(value) if value else None

    def set_last_modified_version(self, version: int) -> None:
        """Set last modified version."""
        self.set_metadata("last_modified_version", str(version))

    # Item helpers

    def upsert_item(self, item: ZoteroItem, content_hash: Optional[str] = None) -> None:
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

    def get_item(self, key: str) -> Optional[ZoteroItem]:
        """Get item by key."""
        cur = self.connect().execute("SELECT * FROM items WHERE key = ?", (key,))
        row = cur.fetchone()
        return _row_to_item(row) if row else None

    def get_all_items(self) -> List[ZoteroItem]:
        """Get all items as a list."""
        return list(self.iter_items())

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

    def get_summary(self, paper_id: str) -> Optional[PaperSummary]:
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


def _row_to_item(row: sqlite3.Row) -> ZoteroItem:
    """Convert database row to ZoteroItem."""
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
        raw=json.loads(row["raw_json"]),
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
        generated_at=datetime.fromisoformat(row["generated_at"]) if row["generated_at"] else datetime.utcnow(),
    )


__all__ = ["ProfileStorage"]
