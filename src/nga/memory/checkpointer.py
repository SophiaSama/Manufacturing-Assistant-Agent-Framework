"""Persistent SQLite checkpointer for LangGraph conversation state."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def build_checkpointer(app_state_db_path: str):
    """Build a SQLite-backed LangGraph checkpointer.

    ``SqliteSaver.from_conn_string`` is a context manager — calling it without
    ``with`` returns a *generator* object, not the saver itself, which causes
    ``'_GeneratorContextManager' object has no attribute 'get_next_version'``.
    Instead we open the connection directly and pass it to ``SqliteSaver``.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    Path(app_state_db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(app_state_db_path, check_same_thread=False)
    return SqliteSaver(conn)
