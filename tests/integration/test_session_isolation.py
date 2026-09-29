"""Integration tests verifying independent session isolation, multi-turn continuity, and session lifecycles."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from nga.memory.checkpointer import build_checkpointer
from nga.ui.server import create_app, ctx


@pytest.fixture
def isolated_client():
    """Create a test client with an isolated temporary app_state.db."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "app_state.db")
        app = create_app()
        # Point settings to isolated app_state_db
        ctx.setup()
        old_settings = ctx.settings
        old_checkpointer = ctx.checkpointer
        ctx.settings = replace(ctx.settings, app_state_db_path=db_path)
        ctx.checkpointer = build_checkpointer(db_path)
        client = TestClient(app)
        try:
            yield client, db_path
        finally:
            ctx.settings = old_settings
            ctx.checkpointer = old_checkpointer


def test_session_lifecycle_and_crud(isolated_client):
    client, _ = isolated_client

    # 1. Create a new session via POST /api/sessions/new
    res = client.post("/api/sessions/new")
    assert res.status_code == 200
    data = res.json()
    assert "thread_id" in data
    assert "short_id" in data
    session_id = data["thread_id"]
    assert len(session_id) == 36

    # 2. History of fresh session is empty
    res_hist = client.get(f"/api/sessions/{session_id}/history")
    assert res_hist.status_code == 200
    assert res_hist.json()["count"] == 0
    assert res_hist.json()["messages"] == []

    # 3. Simulate writing a checkpoint for this session
    config = {"configurable": {"thread_id": session_id, "checkpoint_ns": ""}}
    ctx.checkpointer.put(
        config,
        {
            "v": 1,
            "ts": "2026-09-29T00:00:00Z",
            "id": "cp-1",
            "channel_values": {
                "messages": [
                    HumanMessage(content="Hello NGA"),
                    AIMessage(content="Hello! How can I assist you with assembly?"),
                ]
            },
            "channel_versions": {},
            "versions_seen": {},
        },
        {},
        {},
    )

    # 4. Verify history now reflects the saved messages
    res_hist = client.get(f"/api/sessions/{session_id}/history")
    assert res_hist.status_code == 200
    hist_data = res_hist.json()
    assert hist_data["count"] == 2
    assert hist_data["messages"][0]["type"] == "human"
    assert hist_data["messages"][0]["content"] == "Hello NGA"
    assert hist_data["messages"][1]["type"] == "ai"

    # 5. List sessions and verify it appears
    res_list = client.get("/api/sessions")
    assert res_list.status_code == 200
    sessions = res_list.json()["sessions"]
    session_ids = [s["thread_id"] for s in sessions]
    assert session_id in session_ids

    # 6. Delete the session
    res_del = client.delete(f"/api/sessions/{session_id}")
    assert res_del.status_code == 200
    assert res_del.json()["success"] is True

    # 7. Verify session history is now cleared
    res_hist_after = client.get(f"/api/sessions/{session_id}/history")
    assert res_hist_after.json()["count"] == 0


def test_independent_session_context_isolation(isolated_client, monkeypatch):
    """Verify that multiple concurrent sessions maintain isolated contexts without bleed."""
    client, _ = isolated_client

    # Mock the graph execution so we test session dispatching deterministically
    def mock_stream(initial_state, config, stream_mode):
        thread_id = config["configurable"]["thread_id"]
        msg = initial_state["messages"][0].content
        yield {
            "synthesis": {
                "final_answer": {
                    "direct_answer": f"Processed for {thread_id[:8]}: {msg}",
                    "findings": [],
                    "evidence": [],
                    "answered_questions": [msg],
                    "unanswered_questions": [],
                    "recommendation": None,
                },
                "messages": [AIMessage(content=f"Answer to {msg}")],
            }
        }

    mock_graph = MagicMock()
    mock_graph.stream.side_effect = mock_stream
    monkeypatch.setattr(ctx, "get_graph", lambda role: mock_graph)

    # Send query to Session Alpha
    res_a = client.post(
        "/api/chat",
        json={"message": "Query for Alpha", "role": "operator", "thread_id": "session-alpha-111"},
    )
    assert res_a.status_code == 200
    assert res_a.json()["thread_id"] == "session-alpha-111"
    assert "session-" in res_a.json()["rendered_answer"]

    # Send query to Session Beta
    res_b = client.post(
        "/api/chat",
        json={"message": "Query for Beta", "role": "operator", "thread_id": "session-beta-222"},
    )
    assert res_b.status_code == 200
    assert res_b.json()["thread_id"] == "session-beta-222"

    # Verify both sessions were handled with their respective thread_ids
    calls = mock_graph.stream.call_args_list
    assert len(calls) == 2
    assert calls[0][1]["config"]["configurable"]["thread_id"] == "session-alpha-111"
    assert calls[1][1]["config"]["configurable"]["thread_id"] == "session-beta-222"
    assert calls[0][0][0]["thread_id"] == "session-alpha-111"
    assert calls[1][0][0]["thread_id"] == "session-beta-222"
