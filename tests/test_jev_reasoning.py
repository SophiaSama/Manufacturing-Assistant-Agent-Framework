"""Unit tests for Jev reasoning controller and token telemetry."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from nga.graph.orchestrator import _route_after_agent
from nga.graph.state import AgentState
from nga.rag_agent.jev_reasoning import (
    calculate_reasoning_token_telemetry,
    evaluate_evidence_sufficiency,
    get_typesafe_api_key,
)


def test_typesafe_api_key_resolution(monkeypatch):
    """Test environment resolution for TYPESAFE_API_KEY."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert get_typesafe_api_key() is None

    monkeypatch.setenv("TYPESAFE_API_KEY", "ts_live_key_999")
    assert get_typesafe_api_key() == "ts_live_key_999"

    monkeypatch.setenv("TYPESAFE_API_KEY", "your-key-here")
    assert get_typesafe_api_key() is None


def test_sufficiency_fallback_when_unconfigured(monkeypatch):
    """When API key is not configured, sufficiency check should gracefully fall back."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    res = evaluate_evidence_sufficiency(
        query="What is the torque for station 3?",
        question_parts=["What is the torque for station 3?"],
        evidence="Some evidence",
    )
    assert res.is_complete is False
    assert res.source == "fallback"


def test_sufficiency_empty_evidence():
    """Empty evidence should return incomplete without making API calls."""
    res = evaluate_evidence_sufficiency(
        query="What is the torque?",
        question_parts=["What is the torque?"],
        evidence="",
        api_key="ts_mock_key",
    )
    assert res.is_complete is False
    assert res.source == "fallback"


def test_sufficiency_with_mocked_jev_complete():
    """Test Jev sufficiency check when evidence is complete (early exit)."""
    mock_client = MagicMock()
    mock_resp = MagicMock()

    # Mock Noul, Score, Choice answers
    mock_noul = MagicMock()
    mock_noul.probability = 0.92

    mock_score = MagicMock()
    mock_score.score = 4

    mock_choice = MagicMock()
    mock_choice.choice = "proceed_to_synthesis"
    mock_choice.confidence = 0.90

    mock_resp.answers = {
        "is_sufficient": mock_noul,
        "evidence_completeness": mock_score,
        "next_action": mock_choice,
    }
    mock_client.system_one.return_value = mock_resp

    res = evaluate_evidence_sufficiency(
        query="What is the torque specification?",
        question_parts=["What is the torque specification?"],
        evidence="SOP-OPR-101: Torque is 105 Nm ± 5%.",
        client=mock_client,
    )

    assert res.is_complete is True
    assert res.completeness_score == 4
    assert res.is_sufficient_prob == 0.92
    assert res.next_action == "proceed_to_synthesis"
    assert res.source == "typesafe_jev"


def test_sufficiency_with_mocked_jev_incomplete():
    """Test Jev sufficiency check when evidence is incomplete."""
    mock_client = MagicMock()
    mock_resp = MagicMock()

    mock_noul = MagicMock()
    mock_noul.probability = 0.35

    mock_score = MagicMock()
    mock_score.score = 2

    mock_choice = MagicMock()
    mock_choice.choice = "search_documents"
    mock_choice.confidence = 0.85

    mock_resp.answers = {
        "is_sufficient": mock_noul,
        "evidence_completeness": mock_score,
        "next_action": mock_choice,
    }
    mock_client.system_one.return_value = mock_resp

    res = evaluate_evidence_sufficiency(
        query="What is the torque specification?",
        question_parts=["What is the torque specification?"],
        evidence="General maintenance overview with no numbers.",
        client=mock_client,
    )

    assert res.is_complete is False
    assert res.completeness_score == 2
    assert res.next_action == "search_documents"


def test_token_telemetry_calculation():
    """Test Token Consumption & Cost Telemetry KPIs (TCER, zero output tokens, cost savings)."""
    telemetry = calculate_reasoning_token_telemetry(
        llm_prompt_tokens=2400,
        llm_completion_tokens=400,
        jev_calls_count=2,
        jev_input_tokens=800,
        tool_rounds_executed=2,
        early_exit_triggered=True,
    )

    # 1. Output tokens for Jev must be strictly 0
    assert telemetry["system_one_jev"]["output_tokens"] == 0
    assert telemetry["kpis"]["output_token_elimination_rate_pct"] == 100.0

    # 2. TCER must be favorable (< 0.50) when early exit saved 6 rounds
    assert telemetry["kpis"]["tcer"] < 0.50

    # 3. Cost savings % must be positive
    assert telemetry["kpis"]["cost_savings_pct"] > 50.0
    assert telemetry["kpis"]["early_exit_triggered"] is True


def test_route_after_agent_layer1_circuit_breaker():
    """Layer 1 mechanical circuit breaker must trip at MAX_TOOL_CALL_ROUNDS."""
    tool_call_msg = AIMessage(
        content="I will call a tool",
        tool_calls=[{"name": "search_sop_documents", "args": {"query": "torque"}, "id": "1"}],
    )
    # Simulate 8 rounds of tool calling
    messages = []
    for _ in range(8):
        messages.append(tool_call_msg)
        messages.append(ToolMessage(content="Some result", tool_call_id="1"))
    messages.append(tool_call_msg)

    state: AgentState = {
        "messages": messages,
        "question_parts": ["q"],
        "answered_parts": [],
        "unanswered_parts": [],
        "sql_results": [],
        "retrieved_docs": [],
        "final_answer": None,
        "pending_recommendation": None,
        "user_role": "operator",
        "user_level": 1,
        "model_route": None,
        "evidence_sufficiency": None,
        "token_usage": None,
    }

    # Should force route to synthesis unconditionally
    route = _route_after_agent(state)
    assert route == "synthesis"


def test_route_after_agent_layer2_semantic_early_exit():
    """Layer 2 semantic sufficiency gate triggers early exit before round 8."""
    human = HumanMessage(content="What is the torque for Station 3?")
    tool_call = AIMessage(
        content="Calling search",
        tool_calls=[{"name": "search_sop_documents", "args": {"query": "torque"}, "id": "tc1"}],
    )
    tool_resp = ToolMessage(content="SOP-101: 105 Nm", tool_call_id="tc1")
    agent_msg = AIMessage(
        content="Calling search again",
        tool_calls=[{"name": "search_sop_documents", "args": {"query": "tolerance"}, "id": "tc2"}],
    )

    state: AgentState = {
        "messages": [human, tool_call, tool_resp, agent_msg],
        "question_parts": ["What is the torque for Station 3?"],
        "answered_parts": [],
        "unanswered_parts": [],
        "sql_results": [],
        "retrieved_docs": [],
        "final_answer": None,
        "pending_recommendation": None,
        "user_role": "operator",
        "user_level": 1,
        "model_route": None,
        "evidence_sufficiency": None,
        "token_usage": None,
    }

    # Mock Jev evaluating evidence as complete
    with patch("nga.graph.orchestrator.evaluate_evidence_sufficiency") as mock_eval:
        from nga.rag_agent.jev_reasoning import SufficiencyResult
        mock_eval.return_value = SufficiencyResult(
            is_complete=True,
            completeness_score=4,
            is_sufficient_prob=0.95,
            next_action="proceed_to_synthesis",
            confidence=0.92,
            source="typesafe_jev",
        )

        route = _route_after_agent(state)
        # Should early exit to synthesis even though only 1 tool round ran
        assert route == "synthesis"
