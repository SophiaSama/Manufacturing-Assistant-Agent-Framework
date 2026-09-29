"""Unit tests for LangSmith tracing configuration, decorators, and token telemetry tracking."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from nga.tracing import (
    attach_token_telemetry,
    attach_trace_metadata,
    attach_trace_tags,
    configure_tracing,
    is_tracing_enabled,
    traceable_if_enabled,
)


def test_is_tracing_enabled(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    assert not is_tracing_enabled()

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    assert is_tracing_enabled()

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    assert is_tracing_enabled()


def test_configure_tracing_disabled(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    mock_settings = MagicMock()
    mock_settings.langsmith_tracing_enabled = False
    result = configure_tracing(mock_settings)
    assert result is False


def test_configure_tracing_enabled(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_pt_test_key_12345")
    mock_settings = MagicMock()
    mock_settings.langsmith_tracing_enabled = True
    mock_settings.langsmith_project = "nga-test-project"

    result = configure_tracing(mock_settings)
    assert result is True
    assert os.getenv("LANGCHAIN_TRACING_V2") == "true"
    assert os.getenv("LANGSMITH_TRACING") == "true"
    assert os.getenv("LANGCHAIN_API_KEY") == "lsv2_pt_test_key_12345"
    assert os.getenv("LANGCHAIN_PROJECT") == "nga-test-project"


def test_traceable_if_enabled_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

    @traceable_if_enabled(run_type="chain", name="test_func")
    def sample_func(x: int, y: int = 1) -> int:
        return x + y

    assert sample_func(5, y=3) == 8


def test_traceable_if_enabled_when_enabled(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    @traceable_if_enabled(run_type="chain", name="test_func")
    def sample_func(x: int) -> int:
        return x * 2

    assert sample_func(4) == 8


def test_attach_token_telemetry_safe_execution(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    mock_run = MagicMock()
    mock_run.extra = {}

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

    with patch("nga.tracing.get_current_run", return_value=mock_run):
        attach_token_telemetry(sample_telemetry)
        attach_trace_metadata(sample_key="sample_val")
        attach_trace_tags("nga-test")

    metadata = mock_run.extra.get("metadata", {})
    assert metadata.get("tokens_total") == 3000
    assert metadata.get("cost_total_usd") == 0.008163
    assert metadata.get("tcer") == 0.375
    assert metadata.get("early_exit_triggered") is True
    assert metadata.get("sample_key") == "sample_val"
    mock_run.add_tags.assert_any_call(["nga-test"])
