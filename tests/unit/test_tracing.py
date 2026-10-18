"""Unit tests for OpenTelemetry tracing configuration, decorators, and token telemetry tracking."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from opentelemetry.trace import StatusCode

from nga.tracing import (
    attach_token_telemetry,
    attach_trace_metadata,
    attach_trace_tags,
    clear_in_memory_spans,
    configure_tracing,
    get_in_memory_spans,
    is_tracing_enabled,
    traceable_if_enabled,
)


def test_is_tracing_enabled(monkeypatch):
    monkeypatch.delenv("OTEL_TRACING_ENABLED", raising=False)
    monkeypatch.delenv("TRACING_ENABLED", raising=False)
    assert not is_tracing_enabled()

    monkeypatch.setenv("OTEL_TRACING_ENABLED", "true")
    assert is_tracing_enabled()

    monkeypatch.setenv("OTEL_TRACING_ENABLED", "false")
    monkeypatch.setenv("TRACING_ENABLED", "true")
    assert is_tracing_enabled()


def test_configure_tracing_disabled(monkeypatch):
    monkeypatch.setenv("OTEL_TRACING_ENABLED", "false")
    mock_settings = MagicMock()
    mock_settings.otel_tracing_enabled = False
    result = configure_tracing(mock_settings)
    assert result is False
    assert os.getenv("OTEL_TRACING_ENABLED") == "false"
    assert os.getenv("LANGCHAIN_TRACING_V2") == "false"
    assert os.getenv("LANGSMITH_TRACING") == "false"


def test_configure_tracing_enabled(monkeypatch):
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "memory")
    mock_settings = MagicMock()
    mock_settings.otel_tracing_enabled = True
    mock_settings.otel_service_name = "nga-test-service"
    mock_settings.otel_exporter_otlp_endpoint = ""

    result = configure_tracing(mock_settings)
    assert result is True
    assert os.getenv("OTEL_TRACING_ENABLED") == "true"
    assert os.getenv("OTEL_SERVICE_NAME") == "nga-test-service"
    assert os.getenv("LANGCHAIN_TRACING_V2") == "false"
    assert os.getenv("LANGSMITH_TRACING") == "false"


def test_traceable_if_enabled_when_disabled(monkeypatch):
    monkeypatch.delenv("OTEL_TRACING_ENABLED", raising=False)
    monkeypatch.delenv("TRACING_ENABLED", raising=False)
    clear_in_memory_spans()

    @traceable_if_enabled(run_type="chain", name="test_func_disabled")
    def sample_func(x: int, y: int = 1) -> int:
        return x + y

    assert sample_func(5, y=3) == 8
    assert len(get_in_memory_spans()) == 0


def test_traceable_if_enabled_when_enabled(monkeypatch):
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "memory")
    mock_settings = MagicMock()
    mock_settings.otel_tracing_enabled = True
    mock_settings.otel_service_name = "nga-test-service"
    mock_settings.otel_exporter_otlp_endpoint = ""
    configure_tracing(mock_settings)
    clear_in_memory_spans()

    @traceable_if_enabled(
        run_type="tool",
        name="test_func_enabled",
        metadata={"component": "unit_test"},
        tags=["initial-tag"],
    )
    def sample_func(x: int) -> int:
        return x * 2

    assert sample_func(4) == 8

    spans = get_in_memory_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "test_func_enabled"
    assert span.attributes.get("nga.run_type") == "tool"
    assert span.attributes.get("openinference.span.kind") == "TOOL"
    assert span.attributes.get("component") == "unit_test"
    assert tuple(span.attributes.get("nga.tags", ())) == ("initial-tag",)


def test_traceable_if_enabled_records_exception(monkeypatch):
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "memory")
    mock_settings = MagicMock()
    mock_settings.otel_tracing_enabled = True
    mock_settings.otel_service_name = "nga-test-service"
    mock_settings.otel_exporter_otlp_endpoint = ""
    configure_tracing(mock_settings)
    clear_in_memory_spans()

    @traceable_if_enabled(run_type="chain", name="failing_func")
    def failing_func() -> None:
        raise ValueError("simulated failure")

    with pytest.raises(ValueError, match="simulated failure"):
        failing_func()

    spans = get_in_memory_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR


def test_attach_token_telemetry_safe_execution(monkeypatch):
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "memory")
    mock_settings = MagicMock()
    mock_settings.otel_tracing_enabled = True
    mock_settings.otel_service_name = "nga-test-service"
    mock_settings.otel_exporter_otlp_endpoint = ""
    configure_tracing(mock_settings)
    clear_in_memory_spans()

    sample_telemetry = {
        "system_two_llm": {
            "prompt_tokens": 1200,
            "completion_tokens": 300,
            "cost_usd": 0.0081,
        },
        "system_one_jev": {
            "eval_calls": 2,
            "input_tokens": 1500,
            "output_tokens": 0,
            "cost_usd": 0.000063,
        },
        "totals": {
            "total_tokens": 3000,
            "total_cost_usd": 0.008163,
        },
        "kpis": {
            "tcer": 0.375,
            "output_token_elimination_rate_pct": 100.0,
            "early_exit_triggered": True,
            "tool_rounds_executed": 3,
            "token_savings_pct": 62.5,
            "cost_savings_pct": 62.3,
            "est_fallback_cost_usd": 0.0216,
        },
    }

    @traceable_if_enabled(run_type="chain", name="orchestrator.synthesis")
    def synthesis_step():
        attach_token_telemetry(sample_telemetry)
        attach_trace_metadata(sample_key="sample_val")
        attach_trace_tags("nga-test")

    synthesis_step()

    spans = get_in_memory_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert attrs.get("tokens_total") == 3000
    assert attrs.get("gen_ai.usage.input_tokens") == 1200
    assert attrs.get("gen_ai.usage.output_tokens") == 300
    assert attrs.get("cost_total_usd") == 0.008163
    assert attrs.get("tcer") == 0.375
    assert attrs.get("early_exit_triggered") is True
    assert attrs.get("sample_key") == "sample_val"
    tags = attrs.get("nga.tags", ())
    assert "early-exit" in tags
    assert "tcer:0.375" in tags
    assert "nga-test" in tags
