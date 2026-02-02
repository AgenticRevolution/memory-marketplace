"""
Temporal Storage - Fast Working Memory

Default: SQLite (zero dependencies)
Optional: Redis (for production scale)

This is the "working memory" tier - fast access for recent/active data.
"""

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class TemporalStore(ABC):
    """Abstract base for temporal storage backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value with optional TTL in seconds."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a key."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all data."""
        pass

    # Queue operations for pattern learning
    @abstractmethod
    def push(self, queue: str, value: Any) -> None:
        """Push to a queue (FIFO)."""
        pass

    @abstractmethod
    def pop(self, queue: str, timeout: int = 0) -> Optional[Any]:
        """Pop from a queue."""
        pass

    @abstractmethod
    def queue_length(self, queue: str) -> int:
        """Get queue length."""
        pass


class SQLiteTemporalStore(TemporalStore):
    """
    SQLite-based temporal storage.

    Zero external dependencies - works out of the box.
    Good for development and small-to-medium workloads.
    """

    def __init__(self, db_path: str = "~/.memory-nexus-lite/temporal.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()
        logger.info(f"SQLite temporal store initialized at {self.db_path}")

    @property
    def _conn(self) -> sqlite3.Connection:
        """Thread-local connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS kv_store (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_kv_expires
                ON kv_store(expires_at) WHERE expires_at IS NOT NULL;

                CREATE TABLE IF NOT EXISTS queues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    queue_name TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_queue_name
                ON queues(queue_name, id);
            """)

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        with self._conn:
            self._conn.execute(
                "DELETE FROM kv_store WHERE expires_at IS NOT NULL AND expires_at < ?",
                (datetime.utcnow().isoformat(),)
            )

    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        self._cleanup_expired()
        cursor = self._conn.execute(
            "SELECT value FROM kv_store WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)",
            (key, datetime.utcnow().isoformat())
        )
        row = cursor.fetchone()
        if row:
            try:
                return json.loads(row["value"])
            except json.JSONDecodeError:
                return row["value"]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value with optional TTL in seconds."""
        expires_at = None
        if ttl:
            expires_at = (datetime.utcnow() + timedelta(seconds=ttl)).isoformat()

        value_str = json.dumps(value) if not isinstance(value, str) else value

        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO kv_store (key, value, expires_at)
                   VALUES (?, ?, ?)""",
                (key, value_str, expires_at)
            )
        return True

    def delete(self, key: str) -> bool:
        """Delete a key."""
        with self._conn:
            cursor = self._conn.execute("DELETE FROM kv_store WHERE key = ?", (key,))
            return cursor.rowcount > 0

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None

    def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern (simple wildcard support)."""
        self._cleanup_expired()

        # Convert glob pattern to SQL LIKE pattern
        sql_pattern = pattern.replace("*", "%").replace("?", "_")

        cursor = self._conn.execute(
            "SELECT key FROM kv_store WHERE key LIKE ? AND (expires_at IS NULL OR expires_at > ?)",
            (sql_pattern, datetime.utcnow().isoformat())
        )
        return [row["key"] for row in cursor.fetchall()]

    def clear(self) -> None:
        """Clear all data."""
        with self._conn:
            self._conn.execute("DELETE FROM kv_store")
            self._conn.execute("DELETE FROM queues")

    def push(self, queue: str, value: Any) -> None:
        """Push to a queue (FIFO)."""
        value_str = json.dumps(value) if not isinstance(value, str) else value
        with self._conn:
            self._conn.execute(
                "INSERT INTO queues (queue_name, value) VALUES (?, ?)",
                (queue, value_str)
            )

    def pop(self, queue: str, timeout: int = 0) -> Optional[Any]:
        """Pop from a queue (oldest first)."""
        with self._conn:
            cursor = self._conn.execute(
                """SELECT id, value FROM queues
                   WHERE queue_name = ? ORDER BY id ASC LIMIT 1""",
                (queue,)
            )
            row = cursor.fetchone()
            if row:
                self._conn.execute("DELETE FROM queues WHERE id = ?", (row["id"],))
                try:
                    return json.loads(row["value"])
                except json.JSONDecodeError:
                    return row["value"]
        return None

    def queue_length(self, queue: str) -> int:
        """Get queue length."""
        cursor = self._conn.execute(
            "SELECT COUNT(*) as count FROM queues WHERE queue_name = ?",
            (queue,)
        )
        return cursor.fetchone()["count"]

    def close(self) -> None:
        """Close connection."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()


class RedisTemporalStore(TemporalStore):
    """
    Redis-based temporal storage.

    Requires: pip install redis
    Use for production scale and multi-process deployments.
    """

    def __init__(self, url: str = "redis://localhost:6379/0"):
        try:
            import redis
        except ImportError:
            raise ImportError(
                "Redis support requires: pip install redis\n"
                "Or use SQLiteTemporalStore for zero dependencies."
            )

        self.client = redis.from_url(url, decode_responses=True)
        logger.info(f"Redis temporal store connected to {url}")

    def get(self, key: str) -> Optional[Any]:
        value = self.client.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        value_str = json.dumps(value) if not isinstance(value, str) else value
        if ttl:
            return bool(self.client.setex(key, ttl, value_str))
        return bool(self.client.set(key, value_str))

    def delete(self, key: str) -> bool:
        return bool(self.client.delete(key))

    def exists(self, key: str) -> bool:
        return bool(self.client.exists(key))

    def keys(self, pattern: str = "*") -> List[str]:
        return self.client.keys(pattern)

    def clear(self) -> None:
        self.client.flushdb()

    def push(self, queue: str, value: Any) -> None:
        value_str = json.dumps(value) if not isinstance(value, str) else value
        self.client.rpush(queue, value_str)

    def pop(self, queue: str, timeout: int = 0) -> Optional[Any]:
        if timeout:
            result = self.client.blpop(queue, timeout)
            if result:
                try:
                    return json.loads(result[1])
                except json.JSONDecodeError:
                    return result[1]
        else:
            value = self.client.lpop(queue)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
        return None

    def queue_length(self, queue: str) -> int:
        return self.client.llen(queue)

    def close(self) -> None:
        self.client.close()


def create_temporal_store(backend: str = "sqlite", **kwargs) -> TemporalStore:
    """
    Factory function to create temporal store.

    Args:
        backend: "sqlite" or "redis://..." URL
        **kwargs: Additional arguments for the backend

    Returns:
        TemporalStore instance
    """
    if backend == "sqlite" or backend.startswith("sqlite://"):
        db_path = kwargs.get("db_path", "~/.memory-nexus-lite/temporal.db")
        if backend.startswith("sqlite://"):
            db_path = backend.replace("sqlite://", "")
        return SQLiteTemporalStore(db_path)

    elif backend.startswith("redis://"):
        return RedisTemporalStore(backend)

    else:
        raise ValueError(f"Unknown temporal backend: {backend}")
