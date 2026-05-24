"""Base class for SQLite-backed cache implementations."""

import logging
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

logger = logging.getLogger(__name__)


class BaseSQLiteCache(ABC):
    """Abstract base class for SQLite-backed caches.

    Provides common functionality for SQLite connection management,
    thread-safe write operations, and resource cleanup.

    Subclasses must implement:
    - _ensure_schema(): Create necessary tables and indexes
    - _get_expires_column(): Return the column name for expiration timestamps
    - _get_table_name(): Return the main table name
    """

    def __init__(self, db_path: Path | str) -> None:
        """Initialize the cache.

        Args:
            db_path: Path to SQLite database file.
        """
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._write_lock = threading.Lock()  # Protects concurrent writes
        self._ensure_parent_directory()
        self._ensure_schema()

    def _ensure_parent_directory(self) -> None:
        """Ensure the parent directory exists for the database file."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        """Get or create database connection.

        Returns:
            Active SQLite connection with Row factory enabled.
        """
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    @abstractmethod
    def _ensure_schema(self) -> None:
        """Create tables and indexes if they don't exist.

        Subclasses must implement this to define their schema.
        """
        ...

    @abstractmethod
    def _get_expires_column(self) -> str:
        """Return the column name for expiration timestamps.

        Returns:
            Column name (e.g., "expires_at").
        """
        ...

    @abstractmethod
    def _get_table_name(self) -> str:
        """Return the main table name.

        Returns:
            Table name (e.g., "embeddings" or "paper_metadata").
        """
        ...

    def cleanup_expired(self) -> int:
        """Remove expired entries from the cache.

        Returns:
            Number of deleted rows.
        """
        table = self._get_table_name()
        expires_col = self._get_expires_column()

        conn = self._connect()
        cur = conn.execute(
            f"""
            DELETE FROM {table}
            WHERE {expires_col} IS NOT NULL AND {expires_col} <= datetime('now')
            """
        )
        count = cur.rowcount
        conn.commit()
        if count > 0:
            logger.info("Cleaned up %d expired %s cache entries", count, table)
        return count

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


__all__ = ["BaseSQLiteCache"]
