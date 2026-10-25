"""Persistent SQLite checkpointer for LangGraph conversation state."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def build_checkpointer(app_state_db_path: str):
    """Build a SQLite-backed LangGraph checkpointer."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    Path(app_state_db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(app_state_db_path, check_same_thread=False)
    return SqliteSaver(conn)
