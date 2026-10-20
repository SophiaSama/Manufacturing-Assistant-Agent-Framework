"""Northgate Assembly Plant (NGA) Manufacturing Assistant Agent Framework."""

import nga.config  # noqa: F401 - loads .env and disables telemetry before third-party imports
from nga.tracing import (
    attach_token_telemetry,
    attach_trace_metadata,
    attach_trace_tags,
    clear_in_memory_spans,
    configure_tracing,
    get_current_span,
    get_in_memory_spans,
    is_tracing_enabled,
    traceable_if_enabled,
)

__all__ = [
    "attach_token_telemetry",
    "attach_trace_metadata",
    "attach_trace_tags",
    "clear_in_memory_spans",
    "configure_tracing",
    "get_current_span",
    "get_in_memory_spans",
    "is_tracing_enabled",
    "traceable_if_enabled",
]
