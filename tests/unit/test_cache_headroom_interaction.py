"""Unit tests verifying coexistence between NGA L1/L2 cache and Headroom compression."""

import sqlite3

import pytest

from nga.cache import NgaCache
from nga.compression.manager import HeadroomManager, SessionRawContentStore
from nga.tools.tool_factory import make_headroom_retrieve_tool, make_sql_tool


@pytest.fixture
def test_db_path(tmp_path):
    db_file = tmp_path / "test_nga.db"
    con = sqlite3.connect(str(db_file))
    con.execute("CREATE TABLE build_records (id INTEGER PRIMARY KEY, vin TEXT, station TEXT, torque REAL)")
    for i in range(50):
        con.execute(
            "INSERT INTO build_records (vin, station, torque) VALUES (?, ?, ?)",
            (f"VIN-2025-{i:04d}", f"ST-{100 + (i % 5)}", 104.5 + (i * 0.1)),
        )
    con.commit()
    con.close()
    return str(db_file)


@pytest.fixture
def cache_instance(tmp_path):
    cache_file = tmp_path / "test_cache.db"
    return NgaCache(db_path=str(cache_file))


def test_sql_cache_and_headroom_coexistence(test_db_path, cache_instance):
    session_store = SessionRawContentStore()
    headroom_mgr = HeadroomManager(
        target_ratio=0.7,
        min_tokens=20,
        store=session_store,
        enabled=True,
    )

    sql_tool = make_sql_tool(
        db_path=test_db_path,
        cache=cache_instance,
        headroom_mgr=headroom_mgr,
        session_id="session-test",
    )

    sql = "SELECT * FROM build_records WHERE torque > 105.0"

    # Turn 1: Cache miss
    res1 = sql_tool.invoke({"sql": sql})
    assert "[Compressed query_nga_database" in res1
    assert "hash=" in res1

    # Extract hash
    import re
    match1 = re.search(r"hash=([0-9a-f]{8})", res1)
    assert match1 is not None
    h1 = match1.group(1)

    # Check cache metrics: 1 miss, 0 hits
    m1 = cache_instance.snapshot()
    assert m1["sql"]["misses"] >= 1

    # Turn 2: Cache hit
    res2 = sql_tool.invoke({"sql": sql})
    assert "[Compressed query_nga_database" in res2
    match2 = re.search(r"hash=([0-9a-f]{8})", res2)
    assert match2 is not None
    h2 = match2.group(1)

    assert h1 == h2

    # Check cache metrics: now has hits
    m2 = cache_instance.snapshot()
    assert m2["sql"]["hits"] >= 1


def test_retrieve_uncompressed_evidence(test_db_path, cache_instance):
    session_store = SessionRawContentStore()
    headroom_mgr = HeadroomManager(
        target_ratio=0.7,
        min_tokens=20,
        store=session_store,
        enabled=True,
    )

    sql_tool = make_sql_tool(
        db_path=test_db_path,
        cache=cache_instance,
        headroom_mgr=headroom_mgr,
        session_id="session-42",
    )
    retrieve_tool = make_headroom_retrieve_tool(
        headroom_mgr=headroom_mgr,
        session_id="session-42",
    )

    sql = "SELECT vin, torque FROM build_records LIMIT 25"
    compressed_res = sql_tool.invoke({"sql": sql})

    import re
    match = re.search(r"hash=([0-9a-f]{8})", compressed_res)
    assert match is not None
    h = match.group(1)

    # Retrieve uncompressed evidence by hash
    raw_evidence = retrieve_tool.invoke({"content_hash": h})
    assert "VIN-2025-0000" in raw_evidence
    assert "query" in raw_evidence
    assert "rows" in raw_evidence


def test_cache_clear_does_not_affect_headroom_session(test_db_path, cache_instance):
    session_store = SessionRawContentStore()
    headroom_mgr = HeadroomManager(
        target_ratio=0.7,
        min_tokens=20,
        store=session_store,
        enabled=True,
    )

    sql_tool = make_sql_tool(
        db_path=test_db_path,
        cache=cache_instance,
        headroom_mgr=headroom_mgr,
        session_id="session-clear",
    )
    retrieve_tool = make_headroom_retrieve_tool(
        headroom_mgr=headroom_mgr,
        session_id="session-clear",
    )

    compressed = sql_tool.invoke({"sql": "SELECT * FROM build_records LIMIT 10"})
    import re
    h = re.search(r"hash=([0-9a-f]{8})", compressed).group(1)

    # Clear NGA cache
    cache_instance.clear()
    assert cache_instance.stats().get("entries", 0) == 0

    # Headroom session store still has the hash
    raw = retrieve_tool.invoke({"content_hash": h})
    assert "VIN-2025-0000" in raw
