"""Cache context (contextvars) so tools can resolve the active cache
without rebuilding the agent graph.

Resolution order inside a tool:
    1. explicit `cache` passed at tool construction (prod wiring)
    2. active cache from CacheScope (eval hot mode)
    3. None → no caching (cold)
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator

from nga.cache.metrics import CacheMetrics

_active_cache: contextvars.ContextVar = contextvars.ContextVar(
    "nga_active_cache", default=None
)
_active_env: contextvars.ContextVar = contextvars.ContextVar(
    "nga_cache_env", default="prod"
)


class NgaCache:
    """Facade combining the SQLite store with metrics."""

    def __init__(
        self,
        db_path: str,
        *,
        env: str = "prod",
        max_entries: int = 100_000,
    ) -> None:
        from nga.cache.store import SqliteCacheStore

        self.env = env
        self.store = SqliteCacheStore(db_path, max_entries=max_entries)
        self.metrics = CacheMetrics()

    def snapshot(self) -> dict:
        return self.metrics.snapshot()

    def reset_metrics(self) -> None:
        self.metrics.reset()

    def stats(self) -> dict:
        return self.store.stats()

    def clear(self, prefix: str | None = None) -> int:
        if prefix is None:
            self.store.clear()
            return 0
        return self.store.delete_prefix(prefix)

    def close(self) -> None:
        self.store.close()


def get_active_cache() -> NgaCache | None:
    """Resolve the active cache from the contextvar, if any."""
    return _active_cache.get()


def get_active_env() -> str:
    return _active_env.get()


@contextmanager
def cache_scope(cache: NgaCache | None, env: str = "prod") -> Iterator[None]:
    """Set the active cache for the duration of a block (e.g., one eval question)."""
    token_cache = _active_cache.set(cache)
    token_env = _active_env.set(env)
    try:
        yield
    finally:
        _active_cache.reset(token_cache)
        _active_env.reset(token_env)
