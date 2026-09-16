"""Tests for NGA Manufacturing Assistant Web Interface endpoints."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nga.evaluation.eval_runner import compare_evaluation_runs, load_eval_questions
from nga.memory.decision_log import (
    get_decision_by_id,
    get_decisions,
    init_decision_log,
    insert_recommendation,
    update_decision,
)
from nga.ui.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_get_roles(client):
    response = client.get("/api/roles")
    assert response.status_code == 200
    data = response.json()
    assert "roles" in data
    role_ids = [r["id"] for r in data["roles"]]
    assert "operator" in role_ids
    assert "technician" in role_ids
    assert "engineer" in role_ids
    assert "manager" in role_ids


def test_get_status(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "environment" in data
    assert "database" in data


def test_decision_log_and_hil_flow():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "app_state.db")
        init_decision_log(db_path)

        # Insert a standard recommendation
        d1_id = insert_recommendation(
            db_path,
            question="What is the torque spec on Station 144?",
            recommendation="Quarantine batch AU25-0015 per SOP-OPR-158.",
            category="TORQUE",
            user_role="technician",
            class_a_alert=False,
            escalation_level="L2",
        )
        assert d1_id > 0

        # Insert a Class A recommendation
        d2_id = insert_recommendation(
            db_path,
            question="Brake fluid moisture is 2.1%.",
            recommendation="Immediate stop-ship and Level 4 escalation.",
            category="BRAKE_SAFETY",
            user_role="engineer",
            class_a_alert=True,
            escalation_level="L4",
        )
        assert d2_id > d1_id

        # Query all
        all_decs = get_decisions(db_path)
        assert len(all_decs) == 2

        # Query single
        d2 = get_decision_by_id(db_path, d2_id)
        assert d2 is not None
        assert d2["class_a_alert"] == 1
        assert d2["status"] == "pending"

        # Update decision
        update_decision(db_path, d2_id, status="approved", approver="Eng-101 (Elena Rostova)")
        d2_updated = get_decision_by_id(db_path, d2_id)
        assert d2_updated["status"] == "approved"
        assert "Elena Rostova" in d2_updated["approver"]


def test_hil_action_api(client, monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "app_state.db")
        init_decision_log(db_path)

        dec_id = insert_recommendation(
            db_path,
            question="Engine mount bolt torque drift",
            recommendation="Halt line station 127 and re-audit 10 units",
            category="ASSEMBLY",
            user_role="operator",
            class_a_alert=True,
            escalation_level="L4",
        )

        import dataclasses

        from nga.ui.server import ctx
        ctx.setup()
        new_settings = dataclasses.replace(ctx.settings, app_state_db_path=db_path)
        monkeypatch.setattr(ctx, "settings", new_settings)

        # 1. Reject without justification on Class A should fail (HTTP 400)
        res_fail = client.post(
            f"/api/decisions/{dec_id}/action",
            json={"action": "rejected", "approver": "Op-42", "reason": ""},
        )
        assert res_fail.status_code == 400

        # 2. Reject without approver on Class A should fail (HTTP 400)
        res_fail2 = client.post(
            f"/api/decisions/{dec_id}/action",
            json={"action": "approved", "approver": "", "reason": "Looks good"},
        )
        assert res_fail2.status_code == 400

        # 3. Valid authorization
        res_ok = client.post(
            f"/api/decisions/{dec_id}/action",
            json={"action": "approved", "approver": "MGR-01 (Marcus Vance)", "reason": "Containment verified"},
        )
        assert res_ok.status_code == 200
        data = res_ok.json()
        assert data["success"] is True
        assert data["decision"]["status"] == "approved"


def test_eval_questions_loading():
    all_q = load_eval_questions()
    assert len(all_q) >= 70

    retrieval_q = load_eval_questions(category="retrieval")
    assert len(retrieval_q) > 0
    assert all(q["category"] == "retrieval" for q in retrieval_q)

    smoke_q = load_eval_questions(smoke_test=True)
    assert len(smoke_q) == 5
    # Distinct categories in smoke test
    categories = {q["category"] for q in smoke_q}
    assert len(categories) == 5


def test_eval_compare_logic():
    report_a = {
        "run_label": "baseline_v1",
        "summary": {
            "total": 10,
            "passed": 8,
            "pass_rate": 0.8,
            "avg_score": 0.85,
            "avg_latency_s": 2.5,
            "by_category": [
                {"category": "retrieval", "total": 5, "passed": 4, "pass_rate": 0.8, "avg_score": 0.85, "avg_latency_s": 2.0},
                {"category": "sql", "total": 5, "passed": 4, "pass_rate": 0.8, "avg_score": 0.85, "avg_latency_s": 3.0},
            ],
        },
        "results": [
            {"id": "R1", "category": "retrieval", "passed": True, "overall_score": 0.9, "latency_s": 2.0, "tools_called": ["search_sop_documents"], "checks": {}},
            {"id": "R2", "category": "retrieval", "passed": True, "overall_score": 0.9, "latency_s": 2.0, "tools_called": ["search_sop_documents"], "checks": {}},
            {"id": "R3", "category": "retrieval", "passed": False, "overall_score": 0.4, "latency_s": 2.0, "tools_called": [], "checks": {}},
            {"id": "SQL1", "category": "sql", "passed": True, "overall_score": 1.0, "latency_s": 3.0, "tools_called": ["query_nga_database"], "checks": {}},
        ],
    }

    report_b = {
        "run_label": "candidate_v2",
        "summary": {
            "total": 10,
            "passed": 9,
            "pass_rate": 0.9,
            "avg_score": 0.92,
            "avg_latency_s": 1.8,
            "by_category": [
                {"category": "retrieval", "total": 5, "passed": 5, "pass_rate": 1.0, "avg_score": 0.95, "avg_latency_s": 1.5},
                {"category": "sql", "total": 5, "passed": 4, "pass_rate": 0.8, "avg_score": 0.88, "avg_latency_s": 2.1},
            ],
        },
        "results": [
            {"id": "R1", "category": "retrieval", "passed": True, "overall_score": 1.0, "latency_s": 1.5, "tools_called": ["search_sop_documents"], "checks": {}},
            {"id": "R2", "category": "retrieval", "passed": False, "overall_score": 0.5, "latency_s": 1.5, "tools_called": [], "checks": {}},  # Regression
            {"id": "R3", "category": "retrieval", "passed": True, "overall_score": 0.95, "latency_s": 1.5, "tools_called": ["search_sop_documents"], "checks": {}},  # Improvement
            {"id": "SQL1", "category": "sql", "passed": True, "overall_score": 1.0, "latency_s": 2.1, "tools_called": ["query_nga_database"], "checks": {}},  # Maintained Pass
        ],
    }

    comparison = compare_evaluation_runs(report_a, report_b)
    assert comparison["summary_delta"]["pass_rate"]["delta"] == 0.1
    assert comparison["counts"]["regressions"] == 1
    assert comparison["counts"]["improvements"] == 1
    assert comparison["counts"]["maintained_pass"] == 2
    assert comparison["regressions"][0]["id"] == "R2"
    assert comparison["improvements"][0]["id"] == "R3"


def test_logs_endpoint(client):
    import logging
    nga_logger = logging.getLogger("nga.test")
    nga_logger.info("Test log message for UI stream verification")

    res = client.get("/api/logs")
    assert res.status_code == 200
    data = res.json()
    assert "logs" in data
    assert any("Test log message" in log["message"] for log in data["logs"])


def test_eval_reports_and_compare_api(client, monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        reports_dir = Path(tmpdir)
        report_a = {
            "run_label": "run_a",
            "summary": {"total": 2, "passed": 1, "pass_rate": 0.5, "avg_score": 0.6, "avg_latency_s": 1.2, "by_category": []},
            "results": [{"id": "R1", "category": "retrieval", "passed": True, "overall_score": 0.9, "latency_s": 1.2}],
        }
        report_b = {
            "run_label": "run_b",
            "summary": {"total": 2, "passed": 2, "pass_rate": 1.0, "avg_score": 0.95, "avg_latency_s": 1.0, "by_category": []},
            "results": [{"id": "R1", "category": "retrieval", "passed": True, "overall_score": 0.95, "latency_s": 1.0}],
        }

        (reports_dir / "run_a.json").write_text(json.dumps(report_a), encoding="utf-8")
        (reports_dir / "run_b.json").write_text(json.dumps(report_b), encoding="utf-8")

        # Monkeypatch reports_dir in server functions
        import nga.ui.server as server_mod
        from nga.evaluation import eval_runner
        monkeypatch.setattr(server_mod, "list_evaluation_reports", lambda: eval_runner.list_evaluation_reports(str(reports_dir)))
        monkeypatch.setattr(server_mod, "get_evaluation_report", lambda label: eval_runner.get_evaluation_report(label, str(reports_dir)))

        # 1. Test get reports
        res_list = client.get("/api/eval/reports")
        assert res_list.status_code == 200
        reports = res_list.json()["reports"]
        labels = [r["run_label"] for r in reports]
        assert "run_a" in labels
        assert "run_b" in labels

        # 2. Test compare API
        res_cmp = client.post("/api/eval/compare", json={"run_a": "run_a", "run_b": "run_b"})
        assert res_cmp.status_code == 200
        cmp_data = res_cmp.json()
        assert cmp_data["summary_delta"]["pass_rate"]["delta"] == 0.5


def test_chat_endpoint_mocked(client, monkeypatch):
    from nga.ui.server import ctx
    mock_graph = MagicMock()
    mock_graph.stream.return_value = [
        {"prepare": {"question_parts": ["What is the torque spec on Station 144?"]}},
        {"synthesis": {
            "final_answer": {
                "direct_answer": "The target torque is 105 Nm ± 5%.",
                "findings": ["Station 144 uses TorqMaster TF-6000."],
                "evidence": [{"source_type": "document", "citation": "SOP-OPR-101", "supports": []}],
                "class_a_alert": False,
                "escalation_level": None,
                "recall_criteria_met": [],
                "recommendation": None,
            }
        }},
    ]
    monkeypatch.setattr(ctx, "get_graph", lambda role: mock_graph)

    res = client.post("/api/chat", json={"message": "What is torque at Station 144?", "role": "operator"})
    assert res.status_code == 200
    data = res.json()
    assert data["role"] == "operator"
    assert "105 Nm" in data["rendered_answer"]
    assert len(data["trace_steps"]) == 2


def test_graph_quality_api(client):
    res = client.get("/api/eval/graph-quality")
    assert res.status_code == 200
    data = res.json()
    assert "graph_quality_index" in data
    assert "entity_extraction" in data
    assert "relation_extraction" in data
    assert "entity_alignment" in data
    assert "knowledge_coverage" in data


def test_stepped_suite_api(client):
    res = client.get("/api/eval/stepped-suite")
    assert res.status_code == 200
    data = res.json()
    assert "tiers" in data
    assert len(data["tiers"]) == 4
    tier_names = [t["tier"] for t in data["tiers"]]
    assert tier_names == ["L1", "L2", "L3", "L4"]


def test_models_and_multi_model_api(client):
    # 1. Models endpoint
    res_models = client.get("/api/eval/models")
    assert res_models.status_code == 200
    models_data = res_models.json()
    assert "models" in models_data
    assert "anthropic/claude-3.5-sonnet" in models_data["models"]

    # 2. Multi-model benchmark endpoint
    res_mm = client.post("/api/eval/multi-model", json={
        "models": ["deepseek/deepseek-chat", "google/gemini-1.5-flash"],
        "questions_limit": 5,
    })
    assert res_mm.status_code == 200
    mm_data = res_mm.json()
    assert "models" in mm_data
    assert len(mm_data["models"]) == 2
    assert "pareto_efficient_models" in mm_data

