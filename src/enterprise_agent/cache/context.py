"""Cache context (contextvars) for Enterprise Agent.

Resolution order inside a tool:
    1. explicit `cache` passed at tool construction
    2. active cache from CacheScope
    3. None → no caching (cold)
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator

from enterprise_agent.cache.metrics import CacheMetrics

_active_cache: contextvars.ContextVar = contextvars.ContextVar(
    "enterprise_active_cache", default=None
)
_active_env: contextvars.ContextVar = contextvars.ContextVar(
    "enterprise_cache_env", default="prod"
)


class EnterpriseCache:
    """Facade combining the SQLite store with metrics."""

    def __init__(
        self,
        db_path: str,
        *,
        env: str = "prod",
        max_entries: int = 100_000,
    ) -> None:
        from enterprise_agent.cache.store import SqliteCacheStore

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


# Backward compatibility alias
NgaCache = EnterpriseCache


def get_active_cache() -> EnterpriseCache | None:
    return _active_cache.get()


def get_active_env() -> str:
    return _active_env.get()


@contextmanager
def cache_scope(
    cache: EnterpriseCache | None,
    *,
    env: str | None = None,
) -> Iterator[EnterpriseCache | None]:
    """Context manager setting active cache and env on the current contextvar."""
    token_cache = _active_cache.set(cache)
    target_env = env or (cache.env if cache is not None else "prod")
    token_env = _active_env.set(target_env)
    try:
        yield cache
    finally:
        _active_cache.reset(token_cache)
        _active_env.reset(token_env)
