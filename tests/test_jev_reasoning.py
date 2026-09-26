"""Unit tests for Jev reasoning controller and token telemetry."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from nga.graph.orchestrator import _route_after_agent
from nga.graph.state import AgentState
from nga.rag_agent.jev_reasoning import (
    calculate_reasoning_token_telemetry,
    detect_cross_source_contradiction,
    evaluate_evidence_sufficiency,
    evaluate_fact_groundedness,
    get_typesafe_api_key,
    plan_speculative_fanout,
    screen_evidence_contradictions,
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


def test_groundedness_fallback_when_unconfigured(monkeypatch):
    """When API key is not configured, groundedness check should gracefully fall back."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    res = evaluate_fact_groundedness(
        query="What is the torque for station 3?",
        evidence="SOP-101 states 105 Nm.",
        generated_answer="The torque is 105 Nm per SOP-101.",
    )
    assert res.is_grounded is True
    assert res.groundedness_score == 4
    assert res.source == "fallback"


def test_groundedness_empty_evidence():
    """Empty evidence should fail grounding check immediately."""
    res = evaluate_fact_groundedness(
        query="What is the torque?",
        evidence="",
        generated_answer="The torque is 999 Nm.",
        api_key="ts_mock_key",
    )
    assert res.is_grounded is False
    assert res.groundedness_score == 1
    assert res.source == "fallback"


def test_groundedness_with_mocked_jev_pass():
    """Test Jev grounding check when answer is fully supported."""
    mock_client = MagicMock()
    mock_resp = MagicMock()

    mock_noul = MagicMock()
    mock_noul.probability = 0.95

    mock_score = MagicMock()
    mock_score.score = 5

    mock_choice = MagicMock()
    mock_choice.choice = "none"
    mock_choice.confidence = 0.92

    mock_resp.answers = {
        "is_faithful": mock_noul,
        "groundedness": mock_score,
        "unsupported_claim_type": mock_choice,
    }
    mock_client.system_one.return_value = mock_resp

    res = evaluate_fact_groundedness(
        query="What is the wheel torque specification?",
        evidence="SOP-CHASSIS-04: Torque is 105 Nm ± 5%.",
        generated_answer="The wheel torque specification is 105 Nm ± 5% per SOP-CHASSIS-04.",
        client=mock_client,
    )

    assert res.is_grounded is True
    assert res.groundedness_score == 5
    assert res.is_faithful_prob == 0.95
    assert res.unsupported_claim_type == "none"
    assert res.source == "typesafe_jev"


def test_groundedness_with_mocked_jev_hallucination():
    """Test Jev grounding check catching hallucinated specs."""
    mock_client = MagicMock()
    mock_resp = MagicMock()

    mock_noul = MagicMock()
    mock_noul.probability = 0.08

    mock_score = MagicMock()
    mock_score.score = 1

    mock_choice = MagicMock()
    mock_choice.choice = "invented_numeric_spec"
    mock_choice.confidence = 0.91

    mock_resp.answers = {
        "is_faithful": mock_noul,
        "groundedness": mock_score,
        "unsupported_claim_type": mock_choice,
    }
    mock_client.system_one.return_value = mock_resp

    res = evaluate_fact_groundedness(
        query="What is the torque specification?",
        evidence="SOP-CHASSIS-04: Torque is 105 Nm.",
        generated_answer="The torque is 250 Nm and must be checked every 10 minutes.",
        client=mock_client,
    )

    assert res.is_grounded is False
    assert res.groundedness_score == 1
    assert res.is_faithful_prob == 0.08
    assert res.unsupported_claim_type == "invented_numeric_spec"
    assert res.source == "typesafe_jev"


def test_synthesis_node_with_jev_grounding_quarantine():
    """Synthesis node quarantines response if Jev detects ungrounded claims."""
    from nga.config import Settings
    from nga.graph.orchestrator import _make_synthesis_node
    from nga.rag_agent.jev_reasoning import FactGroundednessResult

    settings = Settings.from_env()
    synthesis_fn = _make_synthesis_node(settings)

    human = HumanMessage(content="What is the torque?")
    tool_call = AIMessage(
        content="Calling search",
        tool_calls=[{"name": "search_sop_documents", "args": {"query": "torque"}, "id": "tc1"}],
    )
    tool_resp = ToolMessage(content="SOP-101: 105 Nm", tool_call_id="tc1")
    agent_msg = AIMessage(content="DIRECT ANSWER: The torque is 500 Nm.\nFINDINGS:\n- Fake spec\nEVIDENCE:\n- doc_id: SOP-101")

    state: AgentState = {
        "messages": [human, tool_call, tool_resp, agent_msg],
        "question_parts": ["What is the torque?"],
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
        "fact_groundedness": None,
    }

    with patch("nga.graph.orchestrator.evaluate_fact_groundedness") as mock_ground:
        mock_ground.return_value = FactGroundednessResult(
            is_faithful=False,
            is_faithful_prob=0.10,
            groundedness_score=1,
            unsupported_claim_type="invented_numeric_spec",
            unsupported_claim_conf=0.92,
            is_grounded=False,
            source="typesafe_jev",
        )

        result = synthesis_fn(state)
        final_answer = result["final_answer"]
        assert "Grounding Verification Alert" in final_answer["direct_answer"]
        assert "Quarantined ungrounded assertion" in final_answer["findings"][0]
        assert result["fact_groundedness"]["is_grounded"] is False
        assert result["token_usage"]["system_one_jev"]["eval_calls"] == 2


def test_plan_speculative_fanout_fallback(monkeypatch):
    """Fallback keyword heuristic identifies target domains without API key."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    plan = plan_speculative_fanout(
        query="What is the torque specification for the caliper and how many units were assembled today at Station 3?",
    )
    assert plan.needs_sql_db is True
    assert plan.needs_sop_procedures is True
    assert "sql_database" in plan.recommended_sources
    assert "sop_procedures" in plan.recommended_sources
    assert plan.cross_source_depth == "dual_source"
    assert plan.source == "fallback"


def test_plan_speculative_fanout_with_mocked_jev():
    """Mocked Jev predicts multi-source fan-out requirements."""
    mock_client = MagicMock()
    mock_resp = MagicMock()

    noul_sql = MagicMock(probability=0.91)
    noul_sop = MagicMock(probability=0.88)
    noul_qual = MagicMock(probability=0.78)
    noul_maint = MagicMock(probability=0.15)
    choice_depth = MagicMock(choice="triangulation")

    mock_resp.answers = {
        "needs_sql_db": noul_sql,
        "needs_sop_procedures": noul_sop,
        "needs_supplier_quality": noul_qual,
        "needs_maintenance_logs": noul_maint,
        "cross_source_depth": choice_depth,
    }
    mock_client.system_one.return_value = mock_resp

    plan = plan_speculative_fanout(
        query="Investigate root cause of caliper bolt failures including SCAR reports and build yields.",
        client=mock_client,
    )

    assert plan.needs_sql_db is True
    assert plan.needs_sop_procedures is True
    assert plan.needs_supplier_quality is True
    assert plan.needs_maintenance_logs is False
    assert plan.cross_source_depth == "triangulation"
    assert "supplier_quality" in plan.recommended_sources
    assert plan.source == "typesafe_jev"


def test_detect_cross_source_contradiction_precedence_scar_over_sop():
    """SCAR containment notice takes precedence over standard SOP specification."""
    mock_client = MagicMock()
    mock_resp = MagicMock()

    noul_contra = MagicMock(probability=0.94)
    choice_nature = MagicMock(choice="numeric_tolerance")

    mock_resp.answers = {
        "has_contradiction": noul_contra,
        "conflict_nature": choice_nature,
    }
    mock_client.system_one.return_value = mock_resp

    source_sop = {
        "type": "SOP",
        "citation": "SOP-CHASSIS-04",
        "content": "Wheel lug nut nominal torque: 105 Nm ± 5%.",
    }
    source_scar = {
        "type": "SCAR",
        "citation": "SCAR-2025-007",
        "content": "Interim containment: tighten lug nuts to 108 Nm pending supplier investigation.",
    }

    res = detect_cross_source_contradiction(source_sop, source_scar, client=mock_client)
    assert res.has_contradiction is True
    assert res.conflict_nature == "numeric_tolerance"
    # SCAR (Rank 1) must override SOP (Rank 3)
    assert "SCAR-2025-007 (Rank 1) overrides SOP-CHASSIS-04 (Rank 3)" in res.precedence_rule
    assert "[DETECTED SPECIFICATION CONFLICT]" in res.resolution_guidance


def test_detect_cross_source_contradiction_no_conflict():
    """Compatible sources evaluate to no contradiction."""
    mock_client = MagicMock()
    mock_resp = MagicMock()

    noul_contra = MagicMock(probability=0.08)
    choice_nature = MagicMock(choice="none")

    mock_resp.answers = {
        "has_contradiction": noul_contra,
        "conflict_nature": choice_nature,
    }
    mock_client.system_one.return_value = mock_resp

    source_a = {
        "type": "SOP",
        "citation": "SOP-101",
        "content": "Torque: 105 Nm.",
    }
    source_b = {
        "type": "database",
        "citation": "torque_logs",
        "content": "Recorded torque 104.8 Nm on station 3.",
    }

    res = detect_cross_source_contradiction(source_a, source_b, client=mock_client)
    assert res.has_contradiction is False
    assert res.conflict_nature == "none"
    assert res.resolution_guidance == ""


def test_prepare_node_with_speculative_fanout():
    """Prepare node initializes fanout_plan in state."""
    from nga.config import Settings
    from nga.graph.orchestrator import _make_prepare_node

    settings = Settings.from_env()
    prep_fn = _make_prepare_node(settings)

    state: AgentState = {
        "messages": [HumanMessage(content="What is the torque specification and defect rate at Station 3?")],
        "question_parts": [],
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
        "fact_groundedness": None,
        "fanout_plan": None,
        "contradiction_resolution": None,
    }

    result = prep_fn(state)
    assert "fanout_plan" in result
    assert result["fanout_plan"]["cross_source_depth"] in ("single_source", "dual_source", "triangulation")
    assert "recommended_sources" in result["fanout_plan"]
