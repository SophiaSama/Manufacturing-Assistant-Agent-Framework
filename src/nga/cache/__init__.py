"""NGA cache module — backward-compatibility shims re-exporting from enterprise_agent.cache."""

from __future__ import annotations

from enterprise_agent.cache import (
    CachedEmbeddings,
    CacheMetrics,
    EnterpriseCache,
    NgaCache,
    SqliteCacheStore,
    cache_scope,
    cached_graph_evidence,
    cached_retrieve,
    cached_run_query,
    canonical_json,
    canonical_sql,
    canonical_text,
    compute_corpus_version,
    get_active_cache,
    get_active_env,
    get_corpus_version,
    get_db_etag,
    make_cache_from_settings,
    make_key,
)

__all__ = [
    "NgaCache",
    "EnterpriseCache",
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
    "make_cache_from_settings",
]
