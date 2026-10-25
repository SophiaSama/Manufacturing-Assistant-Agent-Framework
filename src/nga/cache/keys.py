"""Backward-compatibility re-export for cache keys."""

from enterprise_agent.cache.keys import (
    canonical_json,
    canonical_sql,
    canonical_text,
    make_key,
)

__all__ = ["canonical_json", "canonical_sql", "canonical_text", "make_key"]
