"""Read-only, validated text-to-SQL tool over nga.db."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp


class SqlValidationError(Exception):
    """Raised when a generated SQL statement is not a safe, single SELECT."""


def _connect_read_only(db_path: str) -> sqlite3.Connection:
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _load_schema(db_path: str) -> dict[str, set[str]]:
    with _connect_read_only(db_path) as con:
        cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [row[0] for row in cursor.fetchall()]
        schema: dict[str, set[str]] = {}
        for table_name in table_names:
            pragma_cursor = con.execute(f"PRAGMA table_info({table_name})")
            schema[table_name] = {row[1] for row in pragma_cursor.fetchall()}
    return schema


def _validate_schema_references(db_path: str, statement: exp.Expression) -> None:
    schema = _load_schema(db_path)
    referenced_tables = [table for table in statement.find_all(exp.Table)]
    alias_map: dict[str, str] = {}
    real_tables: list[str] = []

    for table in referenced_tables:
        table_name = table.name
        if table_name not in schema:
            raise SqlValidationError(
                f"Unknown table `{table_name}`. "
                f"Available tables: {', '.join(sorted(schema))}"
            )
        real_tables.append(table_name)
        alias_map[table_name] = table_name
        if table.alias:
            alias_map[table.alias] = table_name

    for column in statement.find_all(exp.Column):
        column_name = column.name
        table_name = column.table
        if table_name:
            resolved_table = alias_map.get(table_name)
            if resolved_table is None:
                continue
            if column_name not in schema[resolved_table]:
                available = ", ".join(sorted(schema[resolved_table]))
                raise SqlValidationError(
                    f"Unknown column `{column_name}` on table `{resolved_table}`. "
                    f"Available columns: {available}"
                )
            continue
        candidate_tables = [
            t for t in real_tables if column_name in schema[t]
        ]
        if not candidate_tables:
            table_details = "; ".join(
                f"{t}({', '.join(sorted(schema[t]))})" for t in real_tables
            )
            raise SqlValidationError(
                f"Unknown column `{column_name}` for referenced tables: "
                f"{', '.join(real_tables)}. Available: {table_details}"
            )


def validate_select_only(sql: str) -> None:
    """Raise SqlValidationError unless `sql` is exactly one read-only SELECT."""
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except sqlglot.errors.ParseError as e:
        raise SqlValidationError(f"Could not parse SQL: {e}") from e

    non_empty = [s for s in statements if s is not None]
    if len(non_empty) != 1:
        raise SqlValidationError("Only a single SQL statement is allowed")

    statement = non_empty[0]
    if not isinstance(statement, exp.Select):
        raise SqlValidationError(
            f"Only SELECT statements are allowed, got: {type(statement).__name__}"
        )

    forbidden_types = (
        exp.Insert, exp.Update, exp.Delete, exp.Drop,
        exp.Create, exp.Alter, exp.Attach, exp.Command,
    )
    for node in statement.walk():
        if isinstance(node, forbidden_types):
            raise SqlValidationError(
                f"Disallowed SQL construct found: {type(node).__name__}"
            )


def describe_schema(db_path: str) -> str:
    """Return a compact human-readable description of all NGA tables and columns."""
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [row[0] for row in cursor.fetchall()]
        lines = []
        for table in sorted(table_names):
            cursor = con.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            lines.append(f"{table}({', '.join(columns)})")
    finally:
        con.close()
    return "\n".join(lines)


def run_query(db_path: str, sql: str) -> dict[str, Any]:
    """Validate and execute a read-only SELECT against nga.db.

    Returns: {query, rows (list[dict]), row_count, tables (list[str])}
    """
    validate_select_only(sql)
    parsed = sqlglot.parse_one(sql, read="sqlite")
    _validate_schema_references(db_path, parsed)

    con = _connect_read_only(db_path)
    try:
        con.row_factory = sqlite3.Row
        cursor = con.execute(sql)
        rows = [dict(row) for row in cursor.fetchall()]
    finally:
        con.close()

    tables = sorted({t.name for t in parsed.find_all(exp.Table)})
    return {
        "query": sql,
        "rows": rows,
        "row_count": len(rows),
        "tables": tables,
    }
