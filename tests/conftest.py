"""Pytest configuration for NGA integration tests."""

from __future__ import annotations

import os

import pytest

# Mock HITL approval to auto-approve for all integration tests
import nga.hitl.approval as approval_module

approval_module.request_approval = lambda *a, **kw: ("approved", None)


def pytest_addoption(parser):
    parser.addoption(
        "--judge",
        action="store_true",
        default=False,
        help="Enable LLM-as-judge scoring (slower, requires API key)",
    )
    parser.addoption(
        "--judge-backend",
        action="store",
        default="auto",
        choices=["auto", "jev", "llm"],
        help="Judge backend: 'jev' (TypeSafe Jev), 'llm' (generative chat model), or 'auto'",
    )
    parser.addoption(
        "--probe-capability",
        action="store_true",
        default=False,
        help="Run a runtime model probe to confirm/upgrade the capability tier",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "required_tier(tier): skip test unless the active model meets tier "
        "(basic|standard|strong) — docs/model-capability-gating-design.md",
    )


def _active_model_id() -> tuple[str | None, str]:
    """Return (model_id, provider) from the environment (Settings)."""
    from nga.config import Settings

    try:
        settings = Settings.from_env()
    except Exception:
        return None, "local"
    if settings.provider == "ollama":
        return settings.ollama_chat_model or None, "ollama"
    return settings.openrouter_model, "openrouter"


def _resolve_active_tier(probe: bool) -> str:
    """Resolve the active capability tier once per session."""
    from nga.evaluation.capability import (
        resolve_tier,
    )

    model_id, provider = _active_model_id()
    override = os.getenv("MODEL_CAPABILITY_OVERRIDE") or None
    tier = resolve_tier(model_id, provider, override=override)

    if probe and tier == "basic" and provider == "ollama":
        # Runtime probe: try a trivial structured call; upgrade to standard on success.
        if _probe_model_works():
            tier = "standard"
    return tier


def _probe_model_works() -> bool:
    """Cheap probe: ask the chat model for strict JSON output."""
    from nga.config import Settings
    from nga.providers.factory import make_chat_model

    try:
        settings = Settings.from_env()
        llm = make_chat_model(settings)
        response = llm.invoke('Reply with ONLY this JSON: {"ok": true}')
        text = str(response.content) if hasattr(response, "content") else str(response)
        return '"ok"' in text and "true" in text
    except Exception as exc:
        print(f"[capability probe] failed: {exc}")
        return False


@pytest.fixture(scope="session")
def use_judge(request):
    return request.config.getoption("--judge")


@pytest.fixture(scope="session")
def judge_backend(request):
    return request.config.getoption("--judge-backend")


@pytest.fixture(scope="session")
def active_capability(request):
    """Resolved capability tier for the configured model (basic|standard|strong)."""
    return _resolve_active_tier(bool(request.config.getoption("--probe-capability")))


def pytest_collection_modifyitems(config, items):
    """Skip tests whose required_tier exceeds the active model's tier."""
    from nga.evaluation.capability import describe_reason, tier_meets

    marked = [i for i in items if i.get_closest_marker("required_tier")]
    if not marked:
        return

    active = _resolve_active_tier(bool(config.getoption("--probe-capability")))
    skipped = 0
    for item in marked:
        marker = item.get_closest_marker("required_tier")
        required = marker.args[0] if marker.args else None
        if required and not tier_meets(active, required):
            model_id, _ = _active_model_id()
            reason = describe_reason(active, required, model_id)
            item.add_marker(pytest.mark.skip(reason=reason))
            skipped += 1
    if skipped:
        print(
            f"[capability] active tier={active}; skipped {skipped} "
            f"integration test(s) below requirement"
        )



# ── Shared integration test fixtures ───────────────────────────────────────────

@pytest.fixture(scope="session")
def settings():
    """Load Settings from environment (uses .env or .env.example defaults)."""
    from nga.config import Settings
    return Settings.from_env()


@pytest.fixture(scope="session")
def agent_graph(settings, tmp_path_factory):
    """Build a full agent graph for integration tests (engineer-level access)."""

    return _build_graph(settings, tmp_path_factory, profile="main")


@pytest.fixture(scope="session")
def conflict_agent_graph(settings, tmp_path_factory):
    """Build an agent graph over the CONFLICT corpus profile.

    Reuses a pre-built conflict store (data/conflict_vector_store) or builds
    one on first use. Skips when the store cannot be built/loaded (no
    embedding provider).
    """
    from pathlib import Path

    from nga.ingestion.build_vector_store import build_vector_store

    store_dir = Path(settings.conflict_vector_store_dir)
    store_ready = (store_dir / "chroma.sqlite3").exists()
    if not store_ready:
        base_dir = Path(settings.nga_db_path).parent.parent
        try:
            build_vector_store(settings, base_dir=base_dir, profile="conflict")
        except Exception as exc:
            pytest.skip(f"conflict vector store unavailable: {exc}")
    return _build_graph(settings, tmp_path_factory, profile="conflict")


def _build_graph(settings, tmp_path_factory, *, profile: str):
    """Build a full agent graph for integration tests (engineer-level access)."""


    from nga.graph.orchestrator import build_orchestrator
    from nga.ingestion.build_graph import load_graph
    from nga.ingestion.build_vector_store import open_vector_store
    from nga.memory.checkpointer import build_checkpointer
    from nga.memory.decision_log import init_decision_log
    from nga.providers.factory import make_embeddings
    from nga.tools.tool_factory import make_retrieval_tool, make_sql_tool

    # Use a temp directory for app state to avoid conflicts
    tmp = tmp_path_factory.mktemp("app_state")
    db_path = str(tmp / "app_state.db")
    init_decision_log(db_path)

    embeddings = make_embeddings(settings)
    store = open_vector_store(settings, embeddings=embeddings, profile=profile)
    graph_data = load_graph(settings.graph_store_dir)
    sql_tool = make_sql_tool(settings.nga_db_path)
    # Use engineer-level (3) for eval — full access
    retrieval_tool = make_retrieval_tool(
        store, graph=graph_data, embeddings=embeddings, user_level=3
    )
    checkpointer = build_checkpointer(db_path)

    return build_orchestrator(settings, checkpointer, sql_tool, retrieval_tool)
