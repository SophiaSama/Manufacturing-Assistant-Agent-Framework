"""NGA cache module — SQLite-backed, env-namespaced, RBAC-scoped caching.

Public API:
    NgaCache, cache_scope, get_active_cache
    CachedEmbeddings, cached_retrieve, cached_graph_evidence, cached_run_query
    compute_corpus_version, get_corpus_version, get_db_etag
    make_cache_from_settings

Design: docs/cache-design.md
"""

from __future__ import annotations

from nga.cache.context import (
    NgaCache,
    cache_scope,
    get_active_cache,
    get_active_env,
)
from nga.cache.keys import canonical_sql, canonical_text, make_key
from nga.cache.layers import (
    CachedEmbeddings,
    cached_graph_evidence,
    cached_retrieve,
    cached_run_query,
    compute_corpus_version,
    get_corpus_version,
    get_db_etag,
)
from nga.cache.metrics import CacheMetrics
from nga.cache.store import SqliteCacheStore

__all__ = [
    "NgaCache",
    "cache_scope",
    "get_active_cache",
    "get_active_env",
    "CachedEmbeddings",
    "cached_graph_evidence",
    "cached_retrieve",
    "cached_run_query",
    "compute_corpus_version",
    "get_corpus_version",
    "get_db_etag",
    "canonical_sql",
    "canonical_text",
    "make_key",
    "CacheMetrics",
    "SqliteCacheStore",
]


def make_cache_from_settings(
    settings: object,
    *,
    env: str = "prod",
) -> NgaCache | None:
    """Build an NgaCache from Settings, or None when caching is disabled."""
    if not getattr(settings, "cache_enabled", False):
        return None
    db_path = (
        getattr(settings, "cache_eval_db_path", "data/cache/eval_cache.db")
        if env == "eval"
        else getattr(settings, "cache_db_path", "data/cache/nga_cache.db")
    )
    return NgaCache(db_path, env=env, max_entries=getattr(settings, "cache_max_entries", 100_000))
