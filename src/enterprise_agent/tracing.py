"""OpenTelemetry tracing configuration, conditional decorators, and token telemetry integration for Enterprise Agent."""

from __future__ import annotations

import functools
import inspect
import json
import logging
import os
from typing import Any, Callable, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Span, Status, StatusCode, Tracer

logger = logging.getLogger("enterprise_agent.tracing")

F = TypeVar("F", bound=Callable[..., Any])

_TRACER_PROVIDER: TracerProvider | None = None
_TRACER: Tracer | None = None
_MEMORY_EXPORTER: InMemorySpanExporter | None = None

_RUN_TYPE_TO_SPAN_KIND = {
    "chain": "CHAIN",
    "tool": "TOOL",
    "retriever": "RETRIEVER",
    "llm": "LLM",
    "embedding": "EMBEDDING",
}


def _sanitize_otel_value(value: Any) -> bool | int | float | str | list[str]:
    """Convert arbitrary Python metadata values into OpenTelemetry-compatible attribute types."""
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, (bool, int, float, str)) for item in value):
            return [str(item) if not isinstance(item, (bool, int, float, str)) else item for item in value]
        return json.dumps(value, default=str)
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    return str(value)


def is_tracing_enabled() -> bool:
    """Return True if OpenTelemetry tracing is enabled via environment variables."""
    return (
        os.getenv("OTEL_TRACING_ENABLED", "false").lower() in ("1", "true", "yes")
        or os.getenv("TRACING_ENABLED", "false").lower() in ("1", "true", "yes")
    )


def get_tracer() -> Tracer:
    """Return the active OpenTelemetry tracer."""
    if _TRACER is not None:
        return _TRACER
    return trace.get_tracer("enterprise_agent.tracing")


def get_in_memory_spans() -> list[Any]:
    """Return finished spans recorded by the in-memory exporter (when configured)."""
    if _MEMORY_EXPORTER is None:
        return []
    return list(_MEMORY_EXPORTER.get_finished_spans())


def clear_in_memory_spans() -> None:
    """Clear any spans stored in the in-memory exporter."""
    if _MEMORY_EXPORTER is not None:
        _MEMORY_EXPORTER.clear()


def configure_tracing(settings: Any | None = None) -> bool:
    """Configure OpenTelemetry TracerProvider and span exporters.

    Supports OTLP gRPC exporter (`OTEL_TRACES_EXPORTER=otlp`), console exporter
    (`OTEL_TRACES_EXPORTER=console`), and in-memory exporter (`OTEL_TRACES_EXPORTER=memory`).
    Explicitly disables legacy LangSmith/LangChain tracing environment variables.
    """
    global _TRACER_PROVIDER, _TRACER, _MEMORY_EXPORTER

    # Ensure LangSmith / LangChain V2 built-in tracing is decoupled and disabled
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"

    tracing_enabled = False
    if settings is not None and hasattr(settings, "otel_tracing_enabled"):
        tracing_enabled = bool(settings.otel_tracing_enabled)
    else:
        tracing_enabled = is_tracing_enabled()

    if not tracing_enabled:
        os.environ["OTEL_TRACING_ENABLED"] = "false"
        logger.debug("OpenTelemetry tracing is disabled.")
        return False

    service_name = ""
    if settings is not None:
        candidate = getattr(settings, "otel_service_name", None)
        if isinstance(candidate, str) and candidate.strip():
            service_name = candidate.strip()
    if not service_name:
        service_name = (
            os.getenv("OTEL_SERVICE_NAME") or "enterprise-agent"
        ).strip()

    endpoint = ""
    if settings is not None:
        candidate_ep = getattr(settings, "otel_exporter_otlp_endpoint", None)
        if isinstance(candidate_ep, str) and candidate_ep.strip():
            endpoint = candidate_ep.strip()
    if not endpoint:
        endpoint = (os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()

    exporter_type = (
        os.getenv("OTEL_TRACES_EXPORTER")
        or ("otlp" if endpoint else "memory")
    ).strip().lower()

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    _MEMORY_EXPORTER = None

    if exporter_type == "otlp" and endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        except Exception as exc:
            logger.warning(
                "Failed to initialize OTLPSpanExporter for endpoint=%s (%s); falling back to in-memory span processor.",
                endpoint,
                exc,
            )
            _MEMORY_EXPORTER = InMemorySpanExporter()
            provider.add_span_processor(SimpleSpanProcessor(_MEMORY_EXPORTER))
    elif exporter_type == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif exporter_type == "none":
        pass
    else:
        _MEMORY_EXPORTER = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(_MEMORY_EXPORTER))

    if _TRACER_PROVIDER is None:
        trace.set_tracer_provider(provider)
    _TRACER_PROVIDER = provider
    _TRACER = provider.get_tracer("enterprise_agent.tracing")

    os.environ["OTEL_TRACING_ENABLED"] = "true"
    os.environ["OTEL_SERVICE_NAME"] = service_name
    if endpoint:
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint

    logger.info(
        "OpenTelemetry tracing initialized: service=%s exporter=%s endpoint=%s",
        service_name,
        exporter_type,
        endpoint or "local",
    )
    return True


def _apply_initial_span_attributes(
    span: Span,
    run_type: str,
    metadata: dict[str, Any] | None,
    tags: list[str] | None,
) -> None:
    if not span.is_recording():
        return
    span.set_attribute("agent.run_type", run_type)
    span.set_attribute("nga.run_type", run_type)
    span.set_attribute(
        "openinference.span.kind",
        _RUN_TYPE_TO_SPAN_KIND.get(run_type.lower(), run_type.upper()),
    )
    if metadata:
        for k, v in metadata.items():
            if v is not None:
                span.set_attribute(str(k), _sanitize_otel_value(v))
    if tags:
        tag_list = [str(t) for t in tags]
        span.set_attribute("agent.tags", tag_list)
        span.set_attribute("nga.tags", tag_list)


def traceable_if_enabled(
    _func: Callable | None = None,
    *,
    run_type: str = "chain",
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    **traceable_kwargs: Any,
) -> Callable:
    """Conditional decorator: wraps function in an OpenTelemetry span when tracing is active.

    When tracing is disabled, calls the underlying function directly with minimal overhead.
    """
    def decorator(fn: Callable) -> Callable:
        span_name = name or getattr(fn, "__name__", "agent.operation")

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not is_tracing_enabled():
                    return await fn(*args, **kwargs)
                tracer = get_tracer()
                with tracer.start_as_current_span(span_name) as span:
                    _apply_initial_span_attributes(span, run_type, metadata, tags)
                    try:
                        return await fn(*args, **kwargs)
                    except Exception as exc:
                        if span.is_recording():
                            span.record_exception(exc)
                            span.set_status(Status(StatusCode.ERROR, str(exc)))
                        raise

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not is_tracing_enabled():
                return fn(*args, **kwargs)
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                _apply_initial_span_attributes(span, run_type, metadata, tags)
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    if span.is_recording():
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise

        return sync_wrapper

    if _func is not None and callable(_func):
        return decorator(_func)
    return decorator


def get_current_span() -> Span | None:
    """Retrieve the active recording OpenTelemetry Span if available."""
    try:
        span = trace.get_current_span()
        if span is not None and span.is_recording():
            return span
        return None
    except Exception:
        return None


# Alias for backward compatibility
get_current_run = get_current_span


def attach_trace_metadata(**metadata: Any) -> None:
    """Attach arbitrary key-value attributes to the active OpenTelemetry span."""
    try:
        span = get_current_span()
        if span is None:
            return
        for key, value in metadata.items():
            if value is not None:
                span.set_attribute(str(key), _sanitize_otel_value(value))
    except Exception as exc:
        logger.debug("Failed to attach trace metadata: %s", exc)


def attach_trace_tags(*tags: str) -> None:
    """Attach tags to the active OpenTelemetry span."""
    try:
        if not tags:
            return
        span = get_current_span()
        if span is None:
            return
        existing: list[str] = []
        raw_attrs = getattr(span, "attributes", None) or getattr(span, "_attributes", None)
        if raw_attrs is not None:
            for key in ("agent.tags", "nga.tags"):
                if key in raw_attrs:
                    prev = raw_attrs[key]
                    if isinstance(prev, (list, tuple)):
                        existing.extend(str(t) for t in prev)
        merged = list(dict.fromkeys([*existing, *(str(t) for t in tags)]))
        span.set_attribute("agent.tags", merged)
        span.set_attribute("nga.tags", merged)
    except Exception as exc:
        logger.debug("Failed to attach trace tags: %s", exc)


def attach_token_telemetry(telemetry: dict[str, Any]) -> None:
    """Attach detailed token consumption & cost telemetry to active OpenTelemetry span."""
    if not telemetry or not is_tracing_enabled():
        return

    try:
        span = get_current_span()
        if span is None:
            return

        kpis = telemetry.get("kpis", {})
        s2 = telemetry.get("system_two_llm", {})
        s1 = telemetry.get("system_one_jev", {})
        totals = telemetry.get("totals", {})

        meta_update = {
            "token_telemetry": telemetry,
            "gen_ai.usage.input_tokens": s2.get("prompt_tokens"),
            "gen_ai.usage.output_tokens": s2.get("completion_tokens"),
            "tokens_total": totals.get("total_tokens"),
            "tokens_llm_prompt": s2.get("prompt_tokens"),
            "tokens_llm_completion": s2.get("completion_tokens"),
            "tokens_jev_input": s1.get("input_tokens"),
            "cost_total_usd": totals.get("total_cost_usd"),
            "cost_llm_usd": s2.get("cost_usd"),
            "cost_jev_usd": s1.get("cost_usd"),
            "tcer": kpis.get("tcer"),
            "cost_savings_pct": kpis.get("cost_savings_pct"),
            "token_savings_pct": kpis.get("token_savings_pct"),
            "early_exit_triggered": kpis.get("early_exit_triggered"),
            "tool_rounds_executed": kpis.get("tool_rounds_executed"),
        }

        for key, value in meta_update.items():
            if value is not None:
                span.set_attribute(key, _sanitize_otel_value(value))

        tags: list[str] = []
        if kpis.get("early_exit_triggered"):
            tags.append("early-exit")
        tcer = kpis.get("tcer")
        if tcer is not None:
            tags.append(f"tcer:{tcer}")
        if tags:
            attach_trace_tags(*tags)

    except Exception as exc:
        logger.debug("Failed to attach token telemetry to OpenTelemetry span: %s", exc)
