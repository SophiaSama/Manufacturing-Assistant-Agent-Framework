"""Tests for report generation — covers the generate_markdown_report path
that previously crashed with an undefined `err` variable."""

from nga.evaluation.report import build_summary, generate_markdown_report
from nga.evaluation.scoring import ScoreResult


def _make_result(**overrides) -> ScoreResult:
    defaults = dict(
        question_id="TST-001",
        category="STR1",
        checks={"has_answer": True},
        deterministic_score=0.85,
        judge_score=5,
        overall_score=0.85,
        passed=True,
        tools_called=["search_documents"],
        latency_s=1.2,
        error=None,
        tier="L1",
    )
    defaults.update(overrides)
    return ScoreResult(**defaults)


def test_generate_markdown_report_no_crash():
    """Regression test: generating markdown must not raise NameError on `err`."""
    results = [
        _make_result(question_id="Q1", passed=True, error=None, tier="L1"),
        _make_result(question_id="Q2", passed=False, error="LLM timeout", tier="L3",
                     overall_score=0.3),
    ]
    md = generate_markdown_report(results, run_label="regression-test")
    assert "Q1" in md
    assert "Q2" in md
    # The error column should render the actual error string, not blow up
    assert "LLM timeout" in md
    # A passing result with no error should render a dash
    assert "—" in md


def test_build_summary_includes_tiers():
    results = [
        _make_result(question_id=f"Q{i}", tier=tier)
        for i, tier in enumerate(["L1", "L1", "L2", "L3", "L4"])
    ]
    summary = build_summary(results)
    assert summary["total"] == 5
    tier_labels = {t["tier"] for t in summary.get("by_tier", [])}
    assert "L1" in tier_labels


# ── Router & Classifier Unit Tests ──────────────────────────────────────────

def test_score_complexity_heuristics():
    from nga.rag_agent.classifier import score_complexity

    # Direct simple lookup should score low
    lookup_query = "What is the target torque for wheel lug nuts?"
    assert score_complexity(lookup_query) <= 3

    # Complex multi-hop root-cause / recall query should score high
    complex_query = (
        "Why did Station 144 experience recurring Class A defects and what is "
        "the root cause according to 8D investigation, QCR-501 C1 criteria, and stop-ship scope?"
    )
    assert score_complexity(complex_query) >= 7


def test_classify_model_route_fallback_warning(monkeypatch, caplog):
    import logging
    from unittest.mock import MagicMock

    from nga.rag_agent.classifier import classify_model_route

    # Ensure TYPESAFE_API_KEY is unset
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    mock_settings = MagicMock()
    mock_settings.rag_tier1_model = "anthropic/claude-haiku-4-5"
    mock_settings.rag_tier2_model = "anthropic/claude-sonnet-4-5"
    mock_settings.rag_tier3_model = "anthropic/claude-opus-4-5"
    mock_settings.rag_tier1_context_window = 8000
    mock_settings.rag_tier2_context_window = 32000
    mock_settings.rag_tier3_context_window = 200000
    mock_settings.rag_tier1_max_hops = 1
    mock_settings.rag_tier2_max_hops = 3
    mock_settings.rag_tier3_max_hops = 8

    # 1. Simple lookup
    with caplog.at_level(logging.WARNING):
        route = classify_model_route(
            "What is the torque for Station 144?",
            mock_settings,
        )

    assert route["choice"] == "fast"
    assert route["model_name"] == "anthropic/claude-haiku-4-5"
    assert route["source"] == "heuristic_fallback"
    assert "TYPESAFE_API_KEY is not set; falling back to local heuristic model routing." in caplog.text

    # 2. Complex query
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        complex_route = classify_model_route(
            "Investigate root cause of Class A defect under QCR-501 and recommend stop-ship containment.",
            mock_settings,
        )

    assert complex_route["choice"] == "powerful"
    assert complex_route["model_name"] == "anthropic/claude-opus-4-5"
    assert complex_route["source"] == "heuristic_fallback"


def test_classify_model_route_typesafe_success():
    from unittest.mock import MagicMock

    from nga.rag_agent.classifier import classify_model_route

    mock_settings = MagicMock()
    mock_settings.rag_tier1_model = "anthropic/claude-haiku-4-5"
    mock_settings.rag_tier2_model = "anthropic/claude-sonnet-4-5"
    mock_settings.rag_tier3_model = "anthropic/claude-opus-4-5"
    mock_settings.rag_tier1_context_window = 8000
    mock_settings.rag_tier2_context_window = 32000
    mock_settings.rag_tier3_context_window = 200000
    mock_settings.rag_tier1_max_hops = 1
    mock_settings.rag_tier2_max_hops = 3
    mock_settings.rag_tier3_max_hops = 8

    mock_client = MagicMock()
    mock_choice_answer = MagicMock()
    mock_choice_answer.choice = "powerful"
    mock_choice_answer.confidence = 0.95
    mock_choice_answer.probabilities = {"fast": 0.05, "balanced": 0.10, "powerful": 0.85}

    mock_response = MagicMock()
    mock_response.answers = {"model_route": mock_choice_answer}
    mock_client.system_one.return_value = mock_response

    route = classify_model_route(
        "Evaluate Class A recall criteria QCR-501 C1",
        mock_settings,
        typesafe_client=mock_client,
    )

    assert route["choice"] == "powerful"
    assert route["model_name"] == "anthropic/claude-opus-4-5"
    assert route["source"] == "typesafe"
    assert route["confidence"] == 0.95
    assert route["probabilities"]["powerful"] == 0.85


def test_classify_model_route_exception_fallback(monkeypatch, caplog):
    import logging
    from unittest.mock import MagicMock

    from nga.rag_agent.classifier import classify_model_route

    monkeypatch.setenv("TYPESAFE_API_KEY", "dummy-key")

    mock_settings = MagicMock()
    mock_settings.rag_tier1_model = "anthropic/claude-haiku-4-5"
    mock_settings.rag_tier2_model = "anthropic/claude-sonnet-4-5"
    mock_settings.rag_tier3_model = "anthropic/claude-opus-4-5"
    mock_settings.rag_tier1_context_window = 8000
    mock_settings.rag_tier2_context_window = 32000
    mock_settings.rag_tier3_context_window = 200000
    mock_settings.rag_tier1_max_hops = 1
    mock_settings.rag_tier2_max_hops = 3
    mock_settings.rag_tier3_max_hops = 8

    mock_client = MagicMock()
    mock_client.system_one.side_effect = TimeoutError("TypeSafe API timeout")

    with caplog.at_level(logging.WARNING):
        route = classify_model_route(
            "What is the torque for lug nuts?",
            mock_settings,
            typesafe_client=mock_client,
        )

    assert route["choice"] == "fast"
    assert route["source"] == "heuristic_fallback"
    assert "TypeSafe model routing failed" in caplog.text
    assert "falling back to local heuristic routing" in caplog.text


def test_orchestrator_prepare_node_populates_model_route(monkeypatch):
    from unittest.mock import MagicMock

    from nga.graph.orchestrator import _make_prepare_node

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    mock_settings = MagicMock()
    mock_settings.rag_tier1_model = "anthropic/claude-haiku-4-5"
    mock_settings.rag_tier2_model = "anthropic/claude-sonnet-4-5"
    mock_settings.rag_tier3_model = "anthropic/claude-opus-4-5"
    mock_settings.rag_tier1_context_window = 8000
    mock_settings.rag_tier2_context_window = 32000
    mock_settings.rag_tier3_context_window = 200000
    mock_settings.rag_tier1_max_hops = 1
    mock_settings.rag_tier2_max_hops = 3
    mock_settings.rag_tier3_max_hops = 8

    prepare_node = _make_prepare_node(mock_settings)

    mock_msg = MagicMock()
    mock_msg.content = "What is the torque for lug nuts?"
    state = {
        "messages": [mock_msg],
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
    }

    result = prepare_node(state)
    assert "model_route" in result
    assert result["model_route"]["choice"] == "fast"
    assert result["model_route"]["model_name"] == "anthropic/claude-haiku-4-5"


# ── Token Consumption KPIs & Model Reporting Unit Tests ───────────────────────

def test_build_summary_with_token_telemetry_and_model_name():
    from nga.rag_agent.jev_reasoning import calculate_reasoning_token_telemetry

    tok1 = calculate_reasoning_token_telemetry(
        llm_prompt_tokens=1500,
        llm_completion_tokens=300,
        jev_calls_count=2,
        jev_input_tokens=600,
        tool_rounds_executed=2,
        early_exit_triggered=True,
    )
    tok2 = calculate_reasoning_token_telemetry(
        llm_prompt_tokens=2000,
        llm_completion_tokens=400,
        jev_calls_count=1,
        jev_input_tokens=400,
        tool_rounds_executed=3,
        early_exit_triggered=False,
    )

    r1 = _make_result(
        question_id="Q1",
        model_name="openai/gpt-4o",
        token_usage=tok1,
    )
    r2 = _make_result(
        question_id="Q2",
        model_name="openai/gpt-4o",
        token_usage=tok2,
    )

    summary = build_summary([r1, r2])
    assert summary["model_name"] == "openai/gpt-4o"
    assert "token_summary" in summary
    ts = summary["token_summary"]
    assert ts["total_tokens"] > 0
    assert ts["avg_tcer"] > 0.0
    assert ts["output_token_elimination_rate_pct"] == 100.0
    assert ts["early_exit_rate_pct"] == 50.0  # 1 out of 2
    assert ts["cost_per_1k_queries_usd"] > 0.0
    assert "avg_prompt_tokens" in summary
    assert "avg_completion_tokens" in summary


def test_generate_markdown_report_includes_token_kpis_and_model():
    from nga.rag_agent.jev_reasoning import calculate_reasoning_token_telemetry

    tok = calculate_reasoning_token_telemetry(
        llm_prompt_tokens=1200,
        llm_completion_tokens=250,
        jev_calls_count=2,
        jev_input_tokens=500,
        tool_rounds_executed=2,
        early_exit_triggered=True,
    )
    r = _make_result(
        question_id="Q10",
        model_name="google/gemini-1.5-pro",
        token_usage=tok,
    )

    md = generate_markdown_report([r], run_label="test-tok-run", model_name="google/gemini-1.5-pro")
    assert "google/gemini-1.5-pro" in md
    assert "Token Consumption & Cost Efficiency KPIs (Jev vs. Fallback)" in md
    assert "Token Consumption Efficiency Ratio (TCER)" in md
    assert "Zero Jev output tokens" in md
    assert "Tokens" in md
    assert "TCER" in md


def test_save_report_persists_model_and_token_summary(tmp_path):
    import json

    from nga.evaluation.report import save_report
    from nga.rag_agent.jev_reasoning import calculate_reasoning_token_telemetry

    tok = calculate_reasoning_token_telemetry(
        llm_prompt_tokens=1200,
        llm_completion_tokens=250,
        jev_calls_count=2,
        jev_input_tokens=500,
        tool_rounds_executed=2,
        early_exit_triggered=True,
    )
    r = _make_result(
        question_id="Q20",
        model_name="anthropic/claude-3.5-sonnet",
        token_usage=tok,
    )

    md_path, json_path = save_report(
        [r],
        reports_dir=str(tmp_path),
        run_label="save-test",
        model_name="anthropic/claude-3.5-sonnet",
    )

    assert md_path.exists()
    assert json_path.exists()

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["model_name"] == "anthropic/claude-3.5-sonnet"
    assert data["token_summary"] is not None
    assert data["token_summary"]["total_tokens"] > 0
    assert data["results"][0]["token_usage"] is not None
    assert data["results"][0]["model_name"] == "anthropic/claude-3.5-sonnet"
