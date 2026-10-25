"""Backward-compatibility re-export for cache context."""

from enterprise_agent.cache.context import (
    EnterpriseCache,
    NgaCache,
    cache_scope,
    get_active_cache,
    get_active_env,
)

__all__ = [
    "EnterpriseCache",
    "NgaCache",
    "cache_scope",
    "get_active_cache",
    "get_active_env",
]
