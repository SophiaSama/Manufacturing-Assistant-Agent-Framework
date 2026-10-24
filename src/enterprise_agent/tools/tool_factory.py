"""Generic LangChain tool factory for SQL, hybrid retrieval, and Headroom compression."""

from __future__ import annotations

import json
import sqlite3
import sys
from typing import Any

from langchain_core.tools import tool

from enterprise_agent.database import engine as _db_engine
from enterprise_agent.database.engine import DatabaseAdapter, SqlValidationError
from enterprise_agent.rag.retrieval import (
    build_or_open_vector_store,
    make_generic_retrieval_tool,
)

# Backwards-compatibility alias for callers importing from enterprise_agent.database.sqlite_adapter
sys.modules.setdefault("enterprise_agent.database.sqlite_adapter", _db_engine)

# Ensure in-memory sqlite connections are supported without raising OperationalError
if not getattr(_db_engine.SqliteAdapter, "_supports_memory", False):
    _orig_connect = _db_engine.SqliteAdapter._connect

    def _safe_connect(self: _db_engine.SqliteAdapter) -> sqlite3.Connection:
        if ":memory:" in getattr(self, "db_path", ""):
            return sqlite3.connect(":memory:")
        return _orig_connect(self)

    _db_engine.SqliteAdapter._connect = _safe_connect  # type: ignore[method-assign]
    _db_engine.SqliteAdapter._supports_memory = True

__all__ = [
    "build_or_open_vector_store",
    "make_generic_retrieval_tool",
    "make_generic_sql_tool",
    "make_headroom_retrieve_tool",
]


def make_generic_sql_tool(
    db_adapter: DatabaseAdapter,
    tool_name: str = "query_database",
    tool_description: str | None = None,
    cache: Any | None = None,
    headroom_mgr: Any | None = None,
    session_id: str = "default",
):
    """Create a LangChain read-only SQL tool using a generic DatabaseAdapter."""
    try:
        schema_desc = db_adapter.describe_schema()
    except Exception:
        schema_desc = ""
    description = tool_description or (
        f"Run a read-only SELECT query against the relational database.\n"
        f"Available tables and columns:\n{schema_desc}\n\n"
        f"Argument `sql` must be a single SELECT statement."
    )

    @tool(tool_name, description=description)
    def sql_tool_func(sql: str) -> str:
        """Execute a SQL query against the domain database."""
        def _run() -> str:
            try:
                result = db_adapter.run_query(sql)
            except SqlValidationError as e:
                return f"Query not allowed: {e}"
            except Exception as e:
                return f"Database error: {e}"
            return json.dumps(result, default=str)

        if cache is not None:
            from enterprise_agent.cache.layers import cached_run_query
            raw = cached_run_query(
                _run,
                sql=sql,
                db_path=getattr(db_adapter, "db_path", "db"),
                cache=cache,
            )
        else:
            raw = _run()

        if headroom_mgr is not None:
            return headroom_mgr.compress_tool_output(tool_name, raw, session_id=session_id)
        return raw

    sql_tool_func.description = description
    return sql_tool_func


def make_headroom_retrieve_tool(
    headroom_mgr: Any,
    session_id: str = "default",
):
    """Create tool allowing the agent to fetch verbatim uncompressed evidence by hash."""

    @tool
    def retrieve_uncompressed_evidence(content_hash: str) -> str:
        """Retrieve original verbatim uncompressed text using a Headroom content hash.

        Use this tool when you need exact numerical tolerances, strict safety compliance clauses,
        or full procedure steps for critical decisions that may be referenced with [hash=...].
        """
        raw = headroom_mgr.store.get(session_id, content_hash)
        if raw is not None:
            return raw
        return f"Error: No uncompressed content found for hash '{content_hash}' in session '{session_id}'."

    return retrieve_uncompressed_evidence
