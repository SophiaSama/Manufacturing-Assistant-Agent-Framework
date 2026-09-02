"""Pytest configuration for NGA integration tests."""

from __future__ import annotations

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


@pytest.fixture(scope="session")
def use_judge(request):
    return request.config.getoption("--judge")


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

    Ingests canonical + the 8 differing variant files so conflict-detection
    tests surface planted inconsistencies (docs/variant-corpus-ingestion-design.md).
    Skips when the conflict store cannot be built (no embedding provider).
    """
    from pathlib import Path

    from nga.ingestion.build_vector_store import build_vector_store

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
