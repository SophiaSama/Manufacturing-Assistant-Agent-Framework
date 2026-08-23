"""Persistent SQLite checkpointer for LangGraph conversation state."""

from __future__ import annotations

from pathlib import Path


def build_checkpointer(app_state_db_path: str):
    """Build a SQLite-backed LangGraph checkpointer."""
    from langgraph.checkpoint.sqlite import SqliteSaver
    Path(app_state_db_path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver.from_conn_string(app_state_db_path)
