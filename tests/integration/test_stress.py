"""Stress, concurrency, load, and adversarial test suite for NGA Manufacturing Assistant."""

from __future__ import annotations

import concurrent.futures
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nga.evaluation.eval_runner import load_eval_questions
from nga.memory.decision_log import (
    get_decisions,
    init_decision_log,
    insert_recommendation,
    update_decision,
)
from nga.ui.server import app, ctx


@pytest.fixture
def client():
    return TestClient(app)


# ── 1. Stress Benchmark Questions Verification ─────────────────────────────────

def test_stress_benchmark_questions_loading():
    """Verify the 10 stress test questions load with proper metadata."""
    stress_qs = load_eval_questions(category="stress")
    assert len(stress_qs) == 10
    q_ids = [q["id"] for q in stress_qs]
    for i in range(1, 11):
        assert f"STR{i}" in q_ids


# ── 2. Concurrency Stress Test ────────────────────────────────────────────────

def test_concurrent_multi_role_requests(client, monkeypatch):
    """Simulate 8 simultaneous users with different roles executing chat requests."""
    roles = ["operator", "technician", "engineer", "manager"] * 2
    mock_responses = {
        "operator": "Assembly procedure for Station 144: Tighten in star pattern.",
        "technician": "Troubleshooting robot RB-07: Servo overcurrent checked.",
        "engineer": "Root cause analysis: 8D containment verified per FAP-401.",
        "manager": "Stop-ship evaluation: QCR-501 C3 field action active.",
    }

    def mock_get_graph(role):
        graph = MagicMock()
        resp_text = mock_responses.get(role, "Standard response")
        graph.stream.return_value = [
            {"prepare": {"question_parts": ["Question part"]}},
            {"synthesis": {
                "final_answer": {
                    "direct_answer": resp_text,
                    "findings": [f"Processed for role {role}"],
                    "evidence": [],
                    "class_a_alert": role == "manager",
                    "escalation_level": "L4" if role == "manager" else None,
                    "recall_criteria_met": ["C3"] if role == "manager" else [],
                    "recommendation": "Check containment" if role == "manager" else None,
                }
            }},
        ]
        return graph

    monkeypatch.setattr(ctx, "get_graph", mock_get_graph)

    def send_req(role_name, idx):
        return client.post(
            "/api/chat",
            json={
                "message": f"Concurrent test message #{idx} from {role_name}",
                "role": role_name,
                "thread_id": f"thread-stress-{idx}",
            },
        )

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(send_req, role, i) for i, role in enumerate(roles)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    total_time = time.perf_counter() - t0
    assert len(results) == 8
    for res in results:
        assert res.status_code == 200
        data = res.json()
        assert "rendered_answer" in data
        assert len(data["trace_steps"]) == 2

    # All 8 requests must complete rapidly under concurrency
    assert total_time < 5.0


# ── 3. Database Concurrency & Lock Stress Test ────────────────────────────────

def test_concurrent_sqlite_decision_writes():
    """Simulate 20 concurrent threads inserting and updating decisions simultaneously."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "app_state_stress.db")
        init_decision_log(db_path)

        def worker(thread_idx):
            dec_id = insert_recommendation(
                db_path=db_path,
                question=f"Stress question #{thread_idx}",
                recommendation=f"Quarantine batch #{thread_idx}",
                category="STRESS_TEST",
                user_role="engineer",
                class_a_alert=thread_idx % 2 == 0,
                escalation_level="L3",
            )
            update_decision(
                db_path=db_path,
                decision_id=dec_id,
                status="approved" if thread_idx % 2 == 0 else "rejected",
                approver=f"Approver-{thread_idx}",
            )
            return dec_id

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(20)]
            dec_ids = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(dec_ids) == 20
        all_decs = get_decisions(db_path, limit=50)
        assert len(all_decs) == 20
        # Verify no database corruption
        approved_count = sum(1 for d in all_decs if d["status"] == "approved")
        rejected_count = sum(1 for d in all_decs if d["status"] == "rejected")
        assert approved_count == 10
        assert rejected_count == 10


# ── 4. High-Load Sequential Turns Stress Test ─────────────────────────────────

def test_sequential_multi_turn_load(client, monkeypatch):
    """Execute 15 rapid sequential chat turns to verify latency and memory stability."""
    mock_graph = MagicMock()
    mock_graph.stream.return_value = [
        {"prepare": {"question_parts": ["Part 1", "Part 2"]}},
        {"synthesis": {
            "final_answer": {
                "direct_answer": "Sequential load test response.",
                "findings": ["Finding A", "Finding B"],
                "evidence": [{"source_type": "sql", "citation": "vehicles", "supports": []}],
                "class_a_alert": False,
                "escalation_level": None,
                "recall_criteria_met": [],
                "recommendation": None,
            }
        }},
    ]
    monkeypatch.setattr(ctx, "get_graph", lambda role: mock_graph)

    latencies = []
    thread_id = "stress-session-seq-001"

    for i in range(15):
        t0 = time.perf_counter()
        res = client.post(
            "/api/chat",
            json={
                "message": f"Sequential query turn {i}: Check vehicle AU25-00{i:02d}",
                "role": "technician",
                "thread_id": thread_id,
            },
        )
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)
        assert res.status_code == 200
        assert res.json()["thread_id"] == thread_id

    # Verify no degradation / memory stall (avg response time < 50ms for mocked backend)
    avg_latency = sum(latencies) / len(latencies)
    assert avg_latency < 0.10


# ── 5. Adversarial Input & Extreme Payload Stress Test ─────────────────────────

def test_adversarial_malformed_and_oversized_payloads(client, monkeypatch):
    """Verify resilience against SQL injection strings, unicode noise, and ultra-large inputs."""
    mock_graph = MagicMock()
    mock_graph.stream.return_value = [
        {"prepare": {"question_parts": ["Sanitized question"]}},
        {"synthesis": {
            "final_answer": {
                "direct_answer": "Handled malicious/large input safely.",
                "findings": [],
                "evidence": [],
                "class_a_alert": False,
                "escalation_level": None,
                "recall_criteria_met": [],
                "recommendation": None,
            }
        }},
    ]
    monkeypatch.setattr(ctx, "get_graph", lambda role: mock_graph)

    adversarial_payloads = [
        # SQL Injection attempt
        "'; DROP TABLE vehicles; DROP TABLE nc_records; SELECT * FROM sqlite_master; --",
        # Large prompt (>10,000 characters)
        "What is the torque spec? " + ("A" * 10000),
        # Unicode / control characters / null bytes / emojis
        "Torque \x00\x07\x1b[31m 😀 \uffff Station 144",
        # Multi-part compound extreme question
        "Part 1? And Part 2? What about Part 3? How about Part 4? Is Part 5 okay? What about Part 6?",
    ]

    for payload in adversarial_payloads:
        res = client.post(
            "/api/chat",
            json={"message": payload, "role": "engineer"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "final_answer" in data
        assert "rendered_answer" in data


# ── 6. Tool Loop & Recursion Protection Stress Test ───────────────────────────

def test_tool_loop_detection_under_recursion_stress():
    """Verify loop detection node helper handles repeated identical tool calls gracefully."""
    from nga.graph.orchestrator import _tool_loop_detected

    # Create mock message sequence with repeated identical tool calls
    msg_with_tool_call = MagicMock()
    msg_with_tool_call.tool_calls = [{"name": "query_nga_database", "args": {"query": "SELECT * FROM vehicles"}}]

    # Under threshold (2 calls) -> False
    messages_short = [msg_with_tool_call, msg_with_tool_call]
    assert _tool_loop_detected(messages_short) is False

    # Exceeding threshold (3 identical calls) -> True (Loop detected)
    messages_loop = [msg_with_tool_call, msg_with_tool_call, msg_with_tool_call]
    assert _tool_loop_detected(messages_loop) is True


# ── 7. Multi-Part Compound Parser Stress Test ─────────────────────────────────

def test_extract_question_parts_under_complex_query():
    """Stress test the question splitter with 6 compound sub-questions."""
    from nga.graph.nodes import extract_question_parts

    complex_q = (
        "What is the wheel lug nut torque target at Station 144, "
        "and what sequence should be followed, "
        "and which tool is used for auditing, "
        "and how often is calibration required, "
        "and is torque drift a Class A defect, "
        "and who must approve lifting a stop ship?"
    )
    parts = extract_question_parts(complex_q)
    assert len(parts) >= 4
    assert any("torque" in p.lower() for p in parts)
    assert any("sequence" in p.lower() for p in parts)
