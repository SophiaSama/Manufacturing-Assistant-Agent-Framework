"""Enterprise Agent cache module — SQLite-backed, env-namespaced, RBAC-scoped caching.

Public API:
    EnterpriseCache, NgaCache, cache_scope, get_active_cache, get_active_env
    CachedEmbeddings, cached_retrieve, cached_graph_evidence, cached_run_query
    compute_corpus_version, get_corpus_version, get_db_etag
    make_key, canonical_sql, canonical_text, canonical_json
    CacheMetrics, SqliteCacheStore
"""

from __future__ import annotations

from enterprise_agent.cache.context import (
    EnterpriseCache,
    NgaCache,
    cache_scope,
    get_active_cache,
    get_active_env,
)
from enterprise_agent.cache.keys import (
    canonical_json,
    canonical_sql,
    canonical_text,
    make_key,
)
from enterprise_agent.cache.layers import (
    CachedEmbeddings,
    cached_graph_evidence,
    cached_retrieve,
    cached_run_query,
    compute_corpus_version,
    get_corpus_version,
    get_db_etag,
)
from enterprise_agent.cache.metrics import CacheMetrics
from enterprise_agent.cache.store import SqliteCacheStore

# Generic alias
MultiTierCache = EnterpriseCache


__all__ = [
    "EnterpriseCache",
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
    "canonical_json",
    "make_key",
    "CacheMetrics",
    "SqliteCacheStore",
]


def make_cache_from_settings(
    settings: object,
    *,
    env: str = "prod",
) -> EnterpriseCache | None:
    """Build an EnterpriseCache from Settings, or None when caching is disabled."""
    if not getattr(settings, "cache_enabled", False):
        return None
    db_path = (
        getattr(settings, "cache_eval_db_path", "data/cache/eval_cache.db")
        if env == "eval"
        else getattr(settings, "cache_db_path", "data/cache/nga_cache.db")
    )
    return EnterpriseCache(db_path, env=env, max_entries=getattr(settings, "cache_max_entries", 100_000))
