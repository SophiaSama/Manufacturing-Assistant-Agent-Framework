"""Read-only, validated text-to-SQL tool over nga.db.

Delegates to the enterprise_agent.database.engine generic SQLite adapter.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

from enterprise_agent.database.engine import (
    SqlValidationError,
    SqliteAdapter,
    validate_schema_references as _validate_schema_references_generic,
    validate_select_only,
)


def _connect_read_only(db_path: str) -> sqlite3.Connection:
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _load_schema(db_path: str) -> dict[str, set[str]]:
    return SqliteAdapter(db_path).load_schema()


def _validate_schema_references(db_path: str, statement: exp.Expression) -> None:
    schema = _load_schema(db_path)
    _validate_schema_references_generic(statement, schema)


def describe_schema(db_path: str) -> str:
    """Return a compact human-readable description of all NGA tables and columns."""
    return SqliteAdapter(db_path).describe_schema()


def run_query(db_path: str, sql: str) -> dict[str, Any]:
    """Validate and execute a read-only SELECT against nga.db.

    Returns: {query, rows (list[dict]), row_count, tables (list[str])}
    """
    return SqliteAdapter(db_path).run_query(sql)
