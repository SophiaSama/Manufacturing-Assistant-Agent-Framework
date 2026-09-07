"""Cache layer adapters: embedding (L0), retrieval (L1), SQL (L2).

Every adapter follows the same contract:
    - resolve the active cache (explicit param > contextvar > None)
    - record hit/miss + elapsed ms in cache metrics
    - never raise on cache failures — a cache error degrades to a miss
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Callable

from langchain_core.embeddings import Embeddings

from nga.cache.context import NgaCache, get_active_cache, get_active_env
from nga.cache.keys import (
    canonical_json,
    canonical_sql,
    canonical_text,
    make_key,
)

logger = logging.getLogger("nga.cache.layers")

# Layer identifiers (must match CacheMetrics.VALID_LAYERS)
L_EMB = "emb"
L_RETR = "retr"
L_SQL = "sql"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_cache(explicit: NgaCache | None) -> NgaCache | None:
    return explicit if explicit is not None else get_active_cache()


def _env(cache: NgaCache | None) -> str:
    return cache.env if cache is not None else get_active_env()


def _key(cache: NgaCache, layer: str, rbac_level: int, canonical: str) -> str:
    return make_key(
        env=_env(cache),
        corpus_version=get_corpus_version(),
        layer=layer,
        rbac_level=rbac_level,
        canonical=canonical,
    )


# ---------------------------------------------------------------------------
# Corpus version + DB etag
# ---------------------------------------------------------------------------

_corpus_version_cache: dict[str, str] = {}


def compute_corpus_version(store_dir: str) -> str:
    """Hash (relpath, mtime, size) over a vector-store directory.

    Re-ingesting documents changes these files → new corpus version →
    old retrieval/embedding keys become unaddressable (design §7.3).
    """
    root = Path(store_dir)
    hasher = hashlib.sha256()
    if not root.exists():
        return "cv-empty"
    files = sorted(
        p for p in root.rglob("*") if p.is_file()
    )
    for p in files:
        try:
            st = p.stat()
            rel = p.relative_to(root)
        except OSError:
            continue
        hasher.update(f"{rel}:{st.st_mtime_ns}:{st.st_size}\n".encode())
    return f"cv-{hasher.hexdigest()[:12]}"


def get_corpus_version() -> str:
    """Memoized corpus version. Falls back to a stable default when the
    store directory is not configured (unit tests, pure-SQL flows)."""
    cache = _corpus_version_cache
    key = "default"
    try:
        from nga.config import Settings

        settings = Settings.from_env()
        store_dir = settings.vector_store_dir
        if store_dir:
            key = str(Path(store_dir).resolve())
        if key not in cache:
            cache[key] = compute_corpus_version(store_dir)
        return cache[key]
    except Exception:
        return cache.get(key, "cv-default")


_db_etag_cache: dict[str, tuple[float, int]] = {}
_etag_connections: dict[str, Any] = {}


def get_db_etag(db_path: str, ttl_s: float = 30.0) -> int:
    """SQLite data version as a cheap change detector.

    PRAGMA data_version increments when the database file is modified by
    another connection. It is only meaningful on a PERSISTENT connection
    (fresh connections report a fresh baseline), so we keep one read-using
    connection per path. Cached for `ttl_s` so it isn't queried per SQL
    statement.
    """
    now = time.time()
    hit = _db_etag_cache.get(db_path)
    if hit and now - hit[0] < ttl_s:
        return hit[1]

    import sqlite3

    version = 0
    if Path(db_path).exists():
        try:
            con = _etag_connections.get(db_path)
            if con is None:
                con = sqlite3.connect(db_path)
                _etag_connections[db_path] = con
            (version,) = con.execute("PRAGMA data_version").fetchone()
        except Exception:
            _etag_connections.pop(db_path, None)
    _db_etag_cache[db_path] = (now, int(version))
    return int(version)


# ---------------------------------------------------------------------------
# L0 — embedding cache
# ---------------------------------------------------------------------------

class CachedEmbeddings(Embeddings):
    """LangChain Embeddings wrapper caching embed_query/embed_documents.

    Wrap the object passed to Chroma so query-time and ingestion-time
    embeddings deduplicate on identical text (keyed by embedding model).
    """

    def __init__(self, inner: Embeddings, *, cache: NgaCache | None = None,
                 model_id: str = "default") -> None:
        self.inner = inner
        self.cache = cache
        self.model_id = model_id

    def _canon(self, texts: list[str]) -> list[str]:
        return [canonical_text(t) for t in texts]

    def _cache_lookup(self, texts: list[str]) -> tuple[list[list[float]], list[str]]:
        cache = _resolve_cache(self.cache)
        if cache is None:
            return [], texts
        found: list[list[float]] = []
        missing: list[str] = []
        for t in texts:
            key = _key(cache, L_EMB, 0, f"{self.model_id}\u0000{t}")
            value = cache.store.get_json(key)
            if value is not None:
                found.append(value)
                cache.metrics.record_hit(L_EMB, 0.0)
            else:
                missing.append(t)
        return found, missing

    def _store_vectors(self, texts: list[str], vectors: list[list[float]]) -> None:
        cache = _resolve_cache(self.cache)
        if cache is None:
            return
        for t, v in zip(texts, vectors):
            key = _key(cache, L_EMB, 0, f"{self.model_id}\u0000{t}")
            cache.store.set_json(key, v)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        found, missing = self._cache_lookup(texts)
        if not missing:
            return found
        t0 = time.perf_counter()
        vectors = self.inner.embed_documents(missing)
        elapsed = (time.perf_counter() - t0) * 1000
        cache = _resolve_cache(self.cache)
        if cache is not None:
            # One miss per missing item (consistent with per-item hits)
            per_item = elapsed / max(len(missing), 1)
            for _ in missing:
                cache.metrics.record_miss(L_EMB, per_item)
        self._store_vectors(missing, vectors)
        return found + vectors

    def embed_query(self, text: str) -> list[float]:
        found, missing = self._cache_lookup([text])
        if found:
            return found[0]
        t0 = time.perf_counter()
        vector = self.inner.embed_query(missing[0])
        elapsed = (time.perf_counter() - t0) * 1000
        cache = _resolve_cache(self.cache)
        if cache is not None:
            cache.metrics.record_miss(L_EMB, elapsed)
        self._store_vectors(missing, [vector])
        return vector


# ---------------------------------------------------------------------------
# L1 — retrieval cache
# ---------------------------------------------------------------------------

def _retr_key(cache: NgaCache, *, query: str, categories: list[str],
              k: int, user_level: int) -> str:
    payload = canonical_json(
        {"q": canonical_text(query), "cats": sorted(categories), "k": k}
    )
    return _key(cache, L_RETR, user_level, payload)


def cached_retrieve(
    fn: Callable[..., Any],
    *,
    query: str,
    categories: list[str],
    k: int,
    user_level: int,
    cache: NgaCache | None = None,
) -> Any:
    """Cache wrapper around retrieve_documents().

    `fn` must return a JSON-serializable list of result dicts.
    """
    resolved = _resolve_cache(cache)
    if resolved is None:
        return fn()
    key = _retr_key(resolved, query=query, categories=categories, k=k, user_level=user_level)
    cached = resolved.store.get_json(key)
    if cached is not None:
        resolved.metrics.record_hit(L_RETR, 0.0)
        return cached
    t0 = time.perf_counter()
    results = fn()
    elapsed = (time.perf_counter() - t0) * 1000
    resolved.metrics.record_miss(L_RETR, elapsed)
    resolved.store.set_json(key, results)
    return results


def cached_graph_evidence(
    fn: Callable[..., Any],
    *,
    query: str,
    k_entities: int,
    max_hops: int,
    user_level: int,
    cache: NgaCache | None = None,
) -> dict[str, Any]:
    """Cache wrapper around retrieve_graph_evidence()."""
    resolved = _resolve_cache(cache)
    if resolved is None:
        return fn()
    payload = canonical_json(
        {"q": canonical_text(query), "k_entities": k_entities, "max_hops": max_hops}
    )
    key = _key(resolved, L_RETR, user_level, payload)
    cached = resolved.store.get_json(key)
    if cached is not None:
        resolved.metrics.record_hit(L_RETR, 0.0)
        return cached
    t0 = time.perf_counter()
    evidence = fn()
    elapsed = (time.perf_counter() - t0) * 1000
    resolved.metrics.record_miss(L_RETR, elapsed)
    resolved.store.set_json(key, evidence)
    return evidence


# ---------------------------------------------------------------------------
# L2 — SQL cache
# ---------------------------------------------------------------------------

def _sql_key(cache: NgaCache, *, sql: str, db_path: str, etag: int) -> str:
    payload = canonical_sql(sql)
    return _key(
        cache, L_SQL, 0, f"{Path(db_path).resolve()}::{etag}::\u0000{payload}"
    )


def cached_run_query(
    fn: Callable[..., Any],
    *,
    sql: str,
    db_path: str,
    cache: NgaCache | None = None,
    etag_ttl_s: float = 30.0,
) -> Any:
    """Cache wrapper around run_query().

    `fn` must return the JSON-serializable run_query result dict.
    The DB etag is part of the key, so a write to nga.db invalidates
    the entry deterministically (design §3.3 / §7.3).
    """
    resolved = _resolve_cache(cache)
    if resolved is None:
        return fn()
    etag = get_db_etag(db_path, ttl_s=etag_ttl_s)
    key = _sql_key(resolved, sql=sql, db_path=db_path, etag=etag)
    cached = resolved.store.get_json(key)
    if cached is not None:
        resolved.metrics.record_hit(L_SQL, 0.0)
        return cached
    t0 = time.perf_counter()
    result = fn()
    elapsed = (time.perf_counter() - t0) * 1000
    resolved.metrics.record_miss(L_SQL, elapsed)
    resolved.store.set_json(key, result)
    return result
