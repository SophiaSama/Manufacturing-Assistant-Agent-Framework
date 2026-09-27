"""Unit tests for the NGA cache module (docs/cache-design.md)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from nga.cache import (
    CacheMetrics,
    NgaCache,
    SqliteCacheStore,
    cache_scope,
    canonical_sql,
    canonical_text,
    make_key,
)
from nga.cache.layers import (
    CachedEmbeddings,
    cached_graph_evidence,
    cached_retrieve,
    cached_run_query,
)


@pytest.fixture()
def cache(tmp_path: Path) -> NgaCache:
    return NgaCache(str(tmp_path / "test_cache.db"), env="eval")


@pytest.fixture()
def cache_env(tmp_path: Path) -> str:
    return "eval"


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class TestSqliteCacheStore:
    def test_set_get_roundtrip(self, tmp_path: Path):
        store = SqliteCacheStore(str(tmp_path / "s.db"))
        store.set_json("k1", {"a": 1})
        assert store.get_json("k1") == {"a": 1}
        store.close()

    def test_expiry(self, tmp_path: Path):
        store = SqliteCacheStore(str(tmp_path / "s.db"))
        store.set_json("k1", "v", ttl=0.01)
        time.sleep(0.05)
        assert store.get_json("k1") is None
        store.close()

    def test_delete_prefix(self, tmp_path: Path):
        store = SqliteCacheStore(str(tmp_path / "s.db"))
        store.set_json("eval:cv1:emb:lvl0:aaa", 1)
        store.set_json("eval:cv1:retr:lvl4:bbb", 2)
        store.set_json("prod:cv1:retr:lvl4:ccc", 3)
        n = store.delete_prefix("eval:cv1:retr")
        assert n == 1
        assert store.get_json("eval:cv1:emb:lvl0:aaa") == 1
        assert store.get_json("prod:cv1:retr:lvl4:ccc") == 3
        store.close()

    def test_lru_eviction(self, tmp_path: Path):
        store = SqliteCacheStore(str(tmp_path / "s.db"), max_entries=10)
        for i in range(30):
            store.set_json(f"key{i}", i)
        assert store.size() <= 10
        # Oldest keys evicted, newest survive
        assert store.get_json("key29") == 29
        assert store.get_json("key0") is None
        store.close()


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

class TestKeys:
    def test_canonical_text_collapses_whitespace(self):
        assert canonical_text("  wheel   torque \n spec ") == "wheel torque spec"

    def test_canonical_text_keeps_case(self):
        assert canonical_text("SOP-OPR-101") != canonical_text("sop-opr-101")

    def test_canonical_sql_normalizes(self):
        a = canonical_sql("SELECT vin FROM vehicles WHERE build_shift='C'")
        b = canonical_sql("select   vin  from vehicles where build_shift='C'")
        assert a == b

    def test_make_key_scopes_by_rbac_level(self):
        k1 = make_key(env="eval", corpus_version="cv1", layer="retr",
                      rbac_level=1, canonical="q")
        k4 = make_key(env="eval", corpus_version="cv1", layer="retr",
                      rbac_level=4, canonical="q")
        assert k1 != k4

    def test_make_key_scopes_by_env(self):
        k1 = make_key(env="eval", corpus_version="cv1", layer="retr",
                      rbac_level=1, canonical="q")
        k2 = make_key(env="prod", corpus_version="cv1", layer="retr",
                      rbac_level=1, canonical="q")
        assert k1 != k2


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_hit_miss_recording(self):
        m = CacheMetrics()
        m.record_hit("retr", 1.0)
        m.record_miss("retr", 2.0)
        snap = m.snapshot()["retr"]
        assert snap["hits"] == 1
        assert snap["misses"] == 1
        assert snap["hit_rate"] == 0.5
        assert snap["total_ms"] == 3.0

    def test_delta(self):
        m = CacheMetrics()
        m.record_hit("sql", 10.0)
        before = m.snapshot()
        m.record_miss("sql", 5.0)
        d = m.delta(before, m.snapshot())["sql"]
        assert d == {"hits": 0, "misses": 1, "ms": 5.0}


# ---------------------------------------------------------------------------
# L0 — embeddings
# ---------------------------------------------------------------------------

class _FakeEmbeddings:
    def __init__(self):
        self.embed_query_calls = 0
        self.embed_documents_calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.embed_query_calls += 1
        return [1.0, float(len(text))]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_documents_calls += 1
        return [[1.0, float(len(t))] for t in texts]


class TestEmbeddingCache:
    def test_query_embedded_once(self, cache: NgaCache):
        inner = _FakeEmbeddings()
        wrapped = CachedEmbeddings(inner, cache=cache, model_id="m1")
        v1 = wrapped.embed_query("what is the torque spec")
        v2 = wrapped.embed_query("what is the torque spec")
        assert inner.embed_query_calls == 1
        assert v1 == v2
        assert cache.snapshot()["emb"]["hits"] == 1

    def test_documents_cached(self, cache: NgaCache):
        inner = _FakeEmbeddings()
        wrapped = CachedEmbeddings(inner, cache=cache, model_id="m1")
        texts = ["a b", "a b", "c d"]
        wrapped.embed_documents(texts)
        wrapped.embed_documents(texts)
        assert inner.embed_documents_calls == 1
        snap = cache.snapshot()["emb"]
        assert snap["hits"] == 3  # all 3 texts cached on 2nd call
        assert snap["misses"] == 3

    def test_no_cache_no_dedup(self):
        inner = _FakeEmbeddings()
        wrapped = CachedEmbeddings(inner)  # no cache
        wrapped.embed_query("x")
        wrapped.embed_query("x")
        assert inner.embed_query_calls == 2


# ---------------------------------------------------------------------------
# L1 — retrieval
# ---------------------------------------------------------------------------

class TestRetrievalCache:
    def test_retrieve_cached(self, cache: NgaCache):
        calls = []

        def fake() -> list[dict]:
            calls.append(1)
            return [{"doc_id": "SOP-OPR-101", "text": "torque 105 Nm"}]

        r1 = cached_retrieve(fake, query="wheel torque", categories=["OPR"],
                             k=5, user_level=1, cache=cache)
        r2 = cached_retrieve(fake, query="wheel torque", categories=["OPR"],
                             k=5, user_level=1, cache=cache)
        assert r1 == r2
        assert len(calls) == 1
        assert cache.snapshot()["retr"]["hits"] == 1

    def test_rbac_level_separates_entries(self, cache: NgaCache):
        calls = []

        def fake() -> list[dict]:
            calls.append(1)
            return [{"doc_id": "x"}]

        cached_retrieve(fake, query="q", categories=["OPR"], k=5, user_level=1, cache=cache)
        cached_retrieve(fake, query="q", categories=["OPR"], k=5, user_level=4, cache=cache)
        assert len(calls) == 2  # level 4 must NOT reuse level 1 entry

    def test_graph_evidence_cached(self, cache: NgaCache):
        calls = []

        def fake() -> dict:
            calls.append(1)
            return {"entities": [{"id": "E1"}], "relations": [], "community_summaries": []}

        cached_graph_evidence(fake, query="recall criteria", k_entities=5,
                              max_hops=2, user_level=1, cache=cache)
        cached_graph_evidence(fake, query="recall criteria", k_entities=5,
                              max_hops=2, user_level=1, cache=cache)
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# L2 — SQL
# ---------------------------------------------------------------------------

class TestSqlCache:
    def test_sql_cached_and_normalized(self, tmp_path: Path, cache: NgaCache):
        calls = []

        def fake() -> dict:
            calls.append(1)
            return {"rows": [{"vin": "NGA-AU25-0001"}], "row_count": 1}

        r1 = cached_run_query(fake, sql="SELECT vin FROM vehicles",
                              db_path=str(tmp_path / "nga.db"), cache=cache)
        r2 = cached_run_query(fake, sql="select  vin  from vehicles",
                              db_path=str(tmp_path / "nga.db"), cache=cache)
        assert r1 == r2
        assert len(calls) == 1  # whitespace/case variants collide on same key

    def test_etag_change_invalidates(self, tmp_path: Path, cache: NgaCache):
        import sqlite3

        calls = []

        def fake() -> dict:
            calls.append(1)
            return {"rows": []}

        db = tmp_path / "nga.db"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE t (x INTEGER)")
        con.commit()
        con.close()

        from nga.cache.layers import get_db_etag

        etag1 = get_db_etag(str(db), ttl_s=0)
        cached_run_query(fake, sql="SELECT 1", db_path=str(db), cache=cache)

        # Mutate the DB → data_version bumps
        con = sqlite3.connect(db)
        con.execute("INSERT INTO t VALUES (1)")
        con.commit()
        con.close()
        etag2 = get_db_etag(str(db), ttl_s=0)
        assert etag2 != etag1

        cached_run_query(fake, sql="SELECT 1", db_path=str(db), cache=cache)
        assert len(calls) == 2  # second call missed due to etag change

    def test_no_cache_calls_fn(self, tmp_path: Path):
        calls = []

        def fake() -> dict:
            calls.append(1)
            return {}

        cached_run_query(fake, sql="SELECT 1", db_path=str(tmp_path / "nga.db"), cache=None)
        cached_run_query(fake, sql="SELECT 1", db_path=str(tmp_path / "nga.db"), cache=None)
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# Context scope
# ---------------------------------------------------------------------------

class TestCacheScope:
    def test_scope_sets_active_cache(self, cache: NgaCache):
        from nga.cache.context import get_active_cache

        assert get_active_cache() is None
        with cache_scope(cache, env="eval"):
            assert get_active_cache() is cache
        assert get_active_cache() is None

    def test_tool_resolves_cache_via_scope(self, cache: NgaCache, tmp_path: Path):
        """Simulates eval hot mode: tools built without explicit cache still
        hit the cache because the eval runner wraps the question in a scope."""
        calls = []

        def fake() -> list[dict]:
            calls.append(1)
            return [{"doc_id": "QCR-501"}]

        # No explicit cache passed
        r1 = cached_retrieve(fake, query="recall", categories=["QCR"],
                             k=5, user_level=4)
        with cache_scope(cache, env="eval"):
            r2 = cached_retrieve(fake, query="recall", categories=["QCR"],
                                 k=5, user_level=4)
            r3 = cached_retrieve(fake, query="recall", categories=["QCR"],
                                 k=5, user_level=4)
        assert r1 == r2 == r3
        assert len(calls) == 2  # 1 cold + 1 warm; third is a hit


# ---------------------------------------------------------------------------
# Improvement metric (docs/cache-design.md §7.4.1)
# ---------------------------------------------------------------------------

class TestCacheImprovement:
    def _report(self, cache_mode: str, avg_latency: float, passed: int = 6,
                score: float = 0.9, cache: dict | None = None) -> dict:
        return {
            "run_label": cache_mode,
            "cache_mode": cache_mode,
            "cache": cache,
            "summary": {
                "avg_latency_s": avg_latency,
                "avg_score": score,
                "passed": passed,
                "total": 6,
            },
            "results": [],
        }

    def test_improvement_block(self):
        from nga.evaluation.eval_runner import _compute_cache_improvement

        cold = self._report("cold", avg_latency=2.5, passed=6)
        hot_cache = {
            "emb": {"hits": 6, "misses": 0, "hit_rate": 1.0, "total_ms": 40.0},
            "retr": {"hits": 5, "misses": 1, "hit_rate": 0.833, "total_ms": 120.0},
            "sql": {"hits": 5, "misses": 1, "hit_rate": 0.833, "total_ms": 80.0},
        }
        hot = self._report("hot", avg_latency=1.5, passed=6, cache=hot_cache)

        imp = _compute_cache_improvement(cold, hot, cold["summary"], hot["summary"])
        assert imp is not None
        assert imp["latency_reduction_pct"] == 40.0  # (1 - 1.5/2.5)*100
        assert imp["quality_parity"] == {"score_delta": 0.0, "pass_delta": 0}
        assert imp["hit_rate"]["retr"] == 0.833
        assert imp["hit_rate"]["emb"] == 1.0

    def test_no_block_without_cold_hot_pair(self):
        from nga.evaluation.eval_runner import _compute_cache_improvement

        both_cold = self._report("cold", avg_latency=1.0)
        assert _compute_cache_improvement(
            both_cold, both_cold, both_cold["summary"], both_cold["summary"]
        ) is None

    def test_works_either_order(self):
        from nga.evaluation.eval_runner import _compute_cache_improvement

        cold = self._report("cold", avg_latency=2.0)
        hot = self._report("hot", avg_latency=1.2)
        # Candidate passed first — must still detect cold baseline
        imp = _compute_cache_improvement(hot, cold, hot["summary"], cold["summary"])
        assert imp is not None and imp["latency_reduction_pct"] == 40.0


# ---------------------------------------------------------------------------
# Integration with real SQL validation
# ---------------------------------------------------------------------------

class TestSqlToolValidation:
    def test_invalid_sql_rejected_before_cache(self, cache: NgaCache, tmp_path: Path):
        from nga.tools.sql_tool import run_query

        db_path = str(tmp_path / "nga.db")
        from nga.tools.sql_tool import SqlValidationError

        with pytest.raises(SqlValidationError):
            run_query(db_path, "DROP TABLE vehicles")
