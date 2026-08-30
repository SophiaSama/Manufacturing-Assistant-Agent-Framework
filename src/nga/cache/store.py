"""SQLite-backed cache store with TTL and LRU eviction.

Single-instance deployment, so a local SQLite file is consistent with the
rest of the stack (app_state.db, checkpoint-sqlite). WAL mode enabled.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key        TEXT PRIMARY KEY,
    v          BLOB NOT NULL,
    exp        REAL,
    created    REAL NOT NULL,
    size       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cache_exp ON cache(exp);
"""


class SqliteCacheStore:
    """Thread-safe SQLite KV store with TTL + LRU eviction."""

    def __init__(self, db_path: str, *, max_entries: int = 100_000) -> None:
        self.db_path = str(db_path)
        self.max_entries = max_entries
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- internal ----------------------------------------------------------
    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _fetchone(self, sql: str, params: tuple = ()) -> Any:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def _purge_expired(self) -> None:
        now = time.time()
        self._execute("DELETE FROM cache WHERE exp IS NOT NULL AND exp < ?", (now,))

    def _evict_lru(self) -> None:
        """Evict oldest-created rows until under max_entries."""
        with self._lock:
            (count,) = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()
            if count < self.max_entries:
                return
            excess = count - self.max_entries + 1  # at least one row
            self._execute(
                "DELETE FROM cache WHERE rowid IN ("
                "  SELECT rowid FROM cache ORDER BY created ASC LIMIT ?"
                ")",
                (max(excess, 1),),
            )

    # -- public ------------------------------------------------------------
    def get(self, key: str) -> bytes | None:
        row = self._fetchone(
            "SELECT v, exp FROM cache WHERE key = ?", (key,)
        )
        if row is None:
            return None
        value, exp = row
        if exp is not None and exp < time.time():
            self._execute("DELETE FROM cache WHERE key = ?", (key,))
            return None
        return bytes(value)

    def get_json(self, key: str) -> Any | None:
        raw = self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._execute("DELETE FROM cache WHERE key = ?", (key,))
            return None

    def set(self, key: str, value: bytes, ttl: float | None = None) -> None:
        exp = None if ttl is None else time.time() + ttl
        self._purge_expired()
        self._execute(
            "INSERT OR REPLACE INTO cache (key, v, exp, created, size) "
            "VALUES (?, ?, ?, ?, ?)",
            (key, value, exp, time.time(), len(value)),
        )
        self._evict_lru()

    def set_json(self, key: str, value: Any, ttl: float | None = None) -> None:
        self.set(key, json.dumps(value, default=str).encode("utf-8"), ttl=ttl)

    def delete_prefix(self, prefix: str) -> int:
        """Delete all keys starting with `prefix`. Returns rows deleted."""
        cur = self._execute("DELETE FROM cache WHERE key LIKE ?", (prefix + "%",))
        return cur.rowcount

    def size(self) -> int:
        (count,) = self._fetchone("SELECT COUNT(*) FROM cache")
        return int(count)

    def stats(self) -> dict[str, int]:
        (count,) = self._fetchone("SELECT COUNT(*) FROM cache")
        (bytes_total,) = self._fetchone(
            "SELECT COALESCE(SUM(size), 0) FROM cache"
        )
        return {"entries": int(count), "size_bytes": int(bytes_total)}

    def clear(self) -> None:
        self._execute("DELETE FROM cache")

    def close(self) -> None:
        with self._lock:
            self._conn.close()
