"""Unit tests for the enterprise_agent core framework components."""

from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

from enterprise_agent.compression import SessionRawContentStore
from enterprise_agent.evaluation import (
    PINNED_JEV_MODEL,
    GroundingResult,
    evaluate_grounding,
    make_domain_jev_eval_metric,
    to_deep_eval_test_case,
)
from enterprise_agent.memory import (
    EpisodicMemoryStore,
    get_decisions,
    init_decision_log,
)
from enterprise_agent.ui.server import app


@pytest.fixture(autouse=True)
def ensure_typesafe_api_key(monkeypatch):
    """Ensure TYPESAFE_API_KEY is present so DeepEval metrics can be initialized in unit tests without live credentials."""
    if not os.getenv("TYPESAFE_API_KEY", "").strip():
        monkeypatch.setenv("TYPESAFE_API_KEY", "test-typesafe-api-key")


def test_enterprise_evaluation_components():
    """Verify enterprise_agent.evaluation components operate cleanly."""
    # 1. Test case conversion
    tc = to_deep_eval_test_case(
        question_data={
            "id": "EA-001",
            "category": "compliance",
            "tier": "L1",
            "question": "What is the compliance requirement?",
            "expected_answer": "Policy Section 4.1",
        },
        actual_output="According to Section 4.1, compliance is required.",
    )
    assert tc.name == "EA-001"
    assert tc.input == "What is the compliance requirement?"

    # 2. Domain JevEval metric creation
    metric = make_domain_jev_eval_metric(system_one_model=PINNED_JEV_MODEL, name="Generic Domain Quality")
    assert metric.name == "Generic Domain Quality"

    # 3. Grounding evaluation
    manifest = {"SOP-OPR-101", "POL-101"}
    evidence = [{"source_type": "document", "citation": "POL-101"}]
    tool_outputs = [json.dumps({"references": [{"doc_id": "POL-101", "chunk_id": "POL-101:c0:can", "corpus_source": "canonical"}]})]
    res = evaluate_grounding(
        answer_text="Per [POL-101], check compliance.",
        evidence=evidence,
        tool_outputs=tool_outputs,
        corpus_manifest=manifest,
        question_id="EA-001",
    )
    assert isinstance(res, GroundingResult)
    assert res.grounded is True
    assert res.citation_validity_score == 1.0


def test_enterprise_memory_and_cache(tmp_path):
    """Verify enterprise_agent cache and memory stores."""
    # 1. Decision log
    db_file = str(tmp_path / "test_decisions.db")
    init_decision_log(db_file)
    decs = get_decisions(db_file)
    assert decs == []

    # 2. Compression raw store
    raw_store = SessionRawContentStore()
    h = raw_store.put("s1", "Full uncompressed evidence text")
    assert raw_store.get("s1", h) == "Full uncompressed evidence text"
    assert raw_store.get("s1", "missing") is None

    # 3. Episodic memory
    mem_db = str(tmp_path / "test_mem.db")
    with EpisodicMemoryStore(db_path=mem_db) as mem:
        mem.record_approved_decision(
            question="What is the torque fault?",
            recommendation="Replaced sensor and recalibrated",
            category="TORQUE_FAULT",
            class_a_alert=True,
            approver="QD",
            user_id="test_user",
        )
        recalls = mem.recall_similar_incidents("TORQUE_FAULT", user_id="test_user")
        assert len(recalls) >= 1
        assert "recalibrated" in recalls[0]["content"]





def test_enterprise_ui_domain_bootstrap():
    """Verify enterprise_agent.ui bootstrap endpoint."""
    client = TestClient(app)
    resp = client.get("/api/domain/bootstrap")
    assert resp.status_code == 200
    data = resp.json()
    assert data["active_framework"] == "enterprise_agent"
    assert data["eval_engine"] == "deepeval"
    assert len(data["metrics"]) >= 5
    assert "roles" in data
