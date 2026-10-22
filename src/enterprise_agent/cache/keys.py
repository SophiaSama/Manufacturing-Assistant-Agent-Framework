"""Cache key canonicalization for Enterprise Agent.

    key = "{env}:{corpus_version}:{layer}:{rbac}:{sha256(canonical_input)}"
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_WS_RE = re.compile(r"\s+")


def canonical_text(text: str) -> str:
    """Canonicalize free text for exact-match keys."""
    return _WS_RE.sub(" ", text).strip()


def canonical_json(payload: Any) -> str:
    """Canonicalize a JSON-serializable payload (stable key ordering)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def canonical_sql(sql: str) -> str:
    """Canonicalize a SQL statement via sqlglot AST round-trip."""
    try:
        import sqlglot

        parsed = sqlglot.parse_one(sql, read="sqlite")
        if parsed is not None:
            return parsed.sql(dialect="sqlite")
    except Exception:
        pass
    return canonical_text(sql)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def make_key(
    *,
    env: str,
    corpus_version: str,
    layer: str,
    rbac_level: int,
    canonical: str,
) -> str:
    """Build a versioned, namespaced, RBAC-scoped cache key."""
    return (
        f"{env}:{corpus_version}:{layer}:lvl{rbac_level}:"
        f"{_sha256(canonical.encode('utf-8'))}"
    )
