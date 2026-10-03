"""LangSmith tracing configuration, conditional decorators, and token telemetry integration."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, TypeVar

logger = logging.getLogger("nga.tracing")

F = TypeVar("F", bound=Callable[..., Any])


def is_tracing_enabled() -> bool:
    """Return True if LangSmith / LangChain tracing is enabled via environment variables."""
    return (
        os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        or os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    )


def configure_tracing(settings: Any | None = None) -> bool:
    """Configure environment variables for LangSmith / LangChain V2 tracing.

    Sets both LANGCHAIN_* and LANGSMITH_* environment variables so all LangChain
    components, LangGraph runs, and custom @traceable spans report to LangSmith.
    """
    tracing_enabled = False
    if settings is not None and hasattr(settings, "langsmith_tracing_enabled"):
        tracing_enabled = bool(settings.langsmith_tracing_enabled)
    else:
        tracing_enabled = is_tracing_enabled()

    if not tracing_enabled:
        logger.debug("LangSmith tracing is disabled.")
        return False

    api_key = (
        os.getenv("LANGSMITH_API_KEY")
        or os.getenv("LANGCHAIN_API_KEY")
        or ""
    ).strip()

    if not api_key or api_key.lower().startswith("your-"):
        logger.warning(
            "LangSmith tracing enabled but LANGSMITH_API_KEY / LANGCHAIN_API_KEY is missing or invalid. Tracing disabled."
        )
        return False

    endpoint = (
        os.getenv("LANGSMITH_ENDPOINT")
        or os.getenv("LANGCHAIN_ENDPOINT")
        or "https://api.smith.langchain.com"
    ).strip()

    project = ""
    if settings is not None and getattr(settings, "langsmith_project", None):
        project = settings.langsmith_project
    if not project:
        project = (
            os.getenv("LANGSMITH_PROJECT")
            or os.getenv("LANGCHAIN_PROJECT")
            or "nga-manufacturing-assistant"
        ).strip()

    # Synchronize standard LangChain and LangSmith environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGCHAIN_ENDPOINT"] = endpoint
    os.environ["LANGSMITH_ENDPOINT"] = endpoint
    os.environ["LANGCHAIN_PROJECT"] = project
    os.environ["LANGSMITH_PROJECT"] = project

    logger.info("LangSmith tracing initialized: project=%s endpoint=%s", project, endpoint)
    return True


def traceable_if_enabled(
    _func: Callable | None = None,
    *,
    run_type: str = "chain",
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    **traceable_kwargs: Any,
) -> Callable:
    """Conditional decorator: wraps function with langsmith.traceable if tracing is active.

    When tracing is disabled or langsmith cannot be imported, returns the original
    function directly with zero overhead.
    """
    def decorator(fn: Callable) -> Callable:
        if not is_tracing_enabled():
            return fn

        try:
            from langsmith import traceable

            span_name = name or fn.__name__
            return traceable(
                run_type=run_type,
                name=span_name,
                metadata=metadata,
                tags=tags,
                **traceable_kwargs,
            )(fn)
        except Exception as exc:
            logger.debug("Failed to apply langsmith.traceable to %s: %s", getattr(fn, "__name__", fn), exc)
            return fn

    if _func is not None and callable(_func):
        return decorator(_func)
    return decorator


def get_current_run():
    """Retrieve active LangSmith RunTree context if available."""
    try:
        from langsmith.run_trees import get_current_run_tree
        return get_current_run_tree()
    except Exception:
        return None


def attach_trace_metadata(**metadata: Any) -> None:
    """Attach arbitrary key-value metadata to the active LangSmith span."""
    try:
        run = get_current_run()
        if run is not None and hasattr(run, "extra"):
            run.extra.setdefault("metadata", {}).update(metadata)
    except Exception as exc:
        logger.debug("Failed to attach trace metadata: %s", exc)


def attach_trace_tags(*tags: str) -> None:
    """Attach tags to the active LangSmith span."""
    try:
        run = get_current_run()
        if run is not None and hasattr(run, "add_tags"):
            run.add_tags(list(tags))
    except Exception as exc:
        logger.debug("Failed to attach trace tags: %s", exc)


def attach_token_telemetry(telemetry: dict[str, Any]) -> None:
    """Attach detailed token consumption & cost telemetry to active LangSmith trace.

    Records:
    - System 2 Generative LLM prompt & completion tokens and USD cost
    - System 1 TypeSafe Jev input tokens and USD cost
    - Total tokens and total cost
    - Key performance indicators: TCER, savings percentages, early exit status
    """
    if not telemetry or not is_tracing_enabled():
        return

    try:
        run = get_current_run()
        if run is None:
            return

        kpis = telemetry.get("kpis", {})
        s2 = telemetry.get("system_two_llm", {})
        s1 = telemetry.get("system_one_jev", {})
        totals = telemetry.get("totals", {})

        meta_update = {
            "token_telemetry": telemetry,
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

        if hasattr(run, "extra"):
            run.extra.setdefault("metadata", {}).update(meta_update)

        if hasattr(run, "add_tags"):
            tags = []
            if kpis.get("early_exit_triggered"):
                tags.append("early-exit")
            tcer = kpis.get("tcer")
            if tcer is not None:
                tags.append(f"tcer:{tcer}")
            if tags:
                run.add_tags(tags)

    except Exception as exc:
        logger.debug("Failed to attach token telemetry to LangSmith trace: %s", exc)
