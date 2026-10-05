"""Unit tests for Headroom compression manager and session-scoped raw content store."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from nga.compression.manager import HeadroomManager, SessionRawContentStore


class TestSessionRawContentStore:
    def test_put_and_get(self):
        store = SessionRawContentStore(max_entries_per_session=10)
        content = "Raw torque specification: 45.5 Nm ± 1.5 Nm on station 104."
        content_hash = store.put("session-1", content)

        assert len(content_hash) == 8
        assert store.get("session-1", content_hash) == content

    def test_session_isolation(self):
        store = SessionRawContentStore(max_entries_per_session=10)
        content = "Confidential engineer note for shift A."
        content_hash = store.put("session-A", content)

        # Other session should not see the hash
        assert store.get("session-B", content_hash) is None

    def test_session_entry_lru_eviction(self):
        store = SessionRawContentStore(max_entries_per_session=3)
        h1 = store.put("s1", "entry 1")
        h2 = store.put("s1", "entry 2")
        h3 = store.put("s1", "entry 3")
        assert store.entry_count("s1") == 3

        # Adding 4th entry should evict oldest (h1)
        h4 = store.put("s1", "entry 4")
        assert store.entry_count("s1") == 3
        assert store.get("s1", h1) is None
        assert store.get("s1", h2) == "entry 2"
        assert store.get("s1", h3) == "entry 3"
        assert store.get("s1", h4) == "entry 4"

    def test_session_count_eviction(self):
        store = SessionRawContentStore(max_sessions=2)
        store.put("s1", "val 1")
        store.put("s2", "val 2")
        assert store.session_count() == 2

        # 3rd session evicts oldest session (s1)
        store.put("s3", "val 3")
        assert store.session_count() == 2
        assert store.get("s1", "any") is None
        assert store.get("s2", store.put("s2", "val 2")) == "val 2"

    def test_clear_session(self):
        store = SessionRawContentStore()
        h = store.put("s1", "content")
        assert store.get("s1", h) == "content"
        store.clear_session("s1")
        assert store.get("s1", h) is None


class TestHeadroomManager:
    def test_small_payload_skips_compression(self):
        store = SessionRawContentStore()
        mgr = HeadroomManager(target_ratio=0.7, min_tokens=250, store=store)
        small_text = "Short reading: OK"

        res = mgr.compress_tool_output("query_nga_database", small_text, session_id="s1")
        assert res == small_text
        # No entry stored for skipped small payload
        assert store.entry_count("s1") == 0

    def test_disabled_manager_skips_compression(self):
        store = SessionRawContentStore()
        mgr = HeadroomManager(target_ratio=0.7, store=store, enabled=False)
        large_text = "Repeated SQL log row data: " * 100

        res = mgr.compress_tool_output("query_nga_database", large_text, session_id="s1")
        assert res == large_text
        assert store.entry_count("s1") == 0

    def test_compress_large_tool_output(self):
        store = SessionRawContentStore()
        mgr = HeadroomManager(target_ratio=0.7, min_tokens=50, store=store, enabled=True)
        large_text = (
            "SOP-OPR-101 Wheel Retention Tightening Procedure. "
            "Step 1: Check fastener alignment. Step 2: Use calibrated tool TQ-6012. "
            "Step 3: Torque to 105.0 Nm. Tolerance is +/- 5%. "
        ) * 20

        result = mgr.compress_tool_output("search_sop_documents", large_text, session_id="s-test")

        assert "[Compressed search_sop_documents" in result
        assert "hash=" in result

        # Extract hash and verify raw retrieval from session store
        import re
        match = re.search(r"hash=([0-9a-f]{8})", result)
        assert match is not None
        content_hash = match.group(1)

        raw_retrieved = store.get("s-test", content_hash)
        assert raw_retrieved == large_text

    def test_compress_history_protects_recent_turns(self):
        mgr = HeadroomManager(target_ratio=0.7, min_tokens=50, enabled=True)
        sys_msg = SystemMessage(content="You are NGA assistant.")
        user_msg = HumanMessage(content="Check station 104")
        t1 = ToolMessage(content="Historical tool output: " * 30, tool_call_id="call_1")
        a1 = AIMessage(content="Analyzed historical data")
        t2 = ToolMessage(content="Recent active tool output: " * 30, tool_call_id="call_2")
        a2 = AIMessage(content="Final active thought")

        messages = [sys_msg, user_msg, t1, a1, t2, a2]
        compressed_history = mgr.compress_message_history(messages, protect_recent=2)

        # Total count preserved
        assert len(compressed_history) == len(messages)
        # System message preserved
        assert compressed_history[0].content == sys_msg.content
        # Protected tail (last 2 messages) preserved exactly
        assert compressed_history[-2].content == t2.content
        assert compressed_history[-1].content == a2.content
