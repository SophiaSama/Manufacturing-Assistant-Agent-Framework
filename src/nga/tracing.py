"""Backward-compatibility re-export for OpenTelemetry tracing."""

from __future__ import annotations

from enterprise_agent.tracing import (
    attach_token_telemetry,
    attach_trace_metadata,
    attach_trace_tags,
    clear_in_memory_spans,
    configure_tracing,
    get_current_run,
    get_current_span,
    get_in_memory_spans,
    get_tracer,
    is_tracing_enabled,
    traceable_if_enabled,
)

__all__ = [
    "attach_token_telemetry",
    "attach_trace_metadata",
    "attach_trace_tags",
    "clear_in_memory_spans",
    "configure_tracing",
    "get_current_run",
    "get_current_span",
    "get_in_memory_spans",
    "get_tracer",
    "is_tracing_enabled",
    "traceable_if_enabled",
]
