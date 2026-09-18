"""Generic SQL database adapter and AST security validator.

Supports SQLite, PostgreSQL, MySQL, DuckDB, etc., via SQLGlot AST parsing,
strict read-only SELECT enforcement, and dynamic schema introspection.
"""

from __future__ import annotations

import abc
import sqlite3
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp


class SqlValidationError(Exception):
    """Raised when a generated SQL query violates security or schema constraints."""


def validate_select_only(sql: str, dialect: str = "sqlite") -> exp.Expression:
    """Ensure `sql` is strictly a single, read-only SELECT statement."""
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except sqlglot.errors.ParseError as e:
        raise SqlValidationError(f"Could not parse SQL: {e}") from e

    non_empty = [s for s in statements if s is not None]
    if len(non_empty) != 1:
        raise SqlValidationError("Only a single SQL statement is allowed")

    statement = non_empty[0]
    if not isinstance(statement, (exp.Select, exp.Union)):
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

    return statement


def validate_schema_references(
    statement: exp.Expression,
    schema: dict[str, set[str]],
) -> None:
    """Verify that all tables and columns referenced in `statement` exist in `schema`."""
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


class DatabaseAdapter(abc.ABC):
    """Abstract interface for read-only database connections."""

    @property
    @abc.abstractmethod
    def dialect(self) -> str:
        """SQLGlot dialect name (e.g. 'sqlite', 'postgres', 'mysql')."""

    @abc.abstractmethod
    def load_schema(self) -> dict[str, set[str]]:
        """Return mapping of table_name -> set of column_names."""

    @abc.abstractmethod
    def describe_schema(self) -> str:
        """Return human-readable schema summary string for LLM prompting."""

    @abc.abstractmethod
    def run_query(self, sql: str) -> dict[str, Any]:
        """Validate and execute a read-only query.

        Returns: {query, rows (list[dict]), row_count, tables (list[str])}
        """


class SqliteAdapter(DatabaseAdapter):
    """Read-only SQLite database adapter."""

    def __init__(self, db_path: str | Path, allowed_tables: list[str] | None = None):
        self.db_path = str(Path(db_path).resolve())
        self.allowed_tables = set(allowed_tables) if allowed_tables else None

    @property
    def dialect(self) -> str:
        return "sqlite"

    def _connect(self) -> sqlite3.Connection:
        uri = f"{Path(self.db_path).resolve().as_uri()}?mode=ro"
        return sqlite3.connect(uri, uri=True)

    def load_schema(self) -> dict[str, set[str]]:
        with self._connect() as con:
            cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
            table_names = [row[0] for row in cursor.fetchall()]
            if self.allowed_tables:
                table_names = [t for t in table_names if t in self.allowed_tables]
            schema: dict[str, set[str]] = {}
            for table in table_names:
                pragma = con.execute(f"PRAGMA table_info({table})")
                schema[table] = {row[1] for row in pragma.fetchall()}
        return schema

    def describe_schema(self) -> str:
        schema = self.load_schema()
        lines = []
        for table in sorted(schema):
            cols = sorted(schema[table])
            lines.append(f"{table}({', '.join(cols)})")
        return "\n".join(lines)

    def run_query(self, sql: str) -> dict[str, Any]:
        statement = validate_select_only(sql, dialect=self.dialect)
        schema = self.load_schema()
        validate_schema_references(statement, schema)

        with self._connect() as con:
            con.row_factory = sqlite3.Row
            cursor = con.execute(sql)
            rows = [dict(row) for row in cursor.fetchall()]

        tables = sorted({t.name for t in statement.find_all(exp.Table)})
        return {
            "query": sql,
            "rows": rows,
            "row_count": len(rows),
            "tables": tables,
        }


class GenericSqlAdapter(DatabaseAdapter):
    """Generic SQL adapter using ANSI information_schema."""

    def __init__(
        self,
        connection_factory: Any,
        dialect: str = "postgres",
        allowed_tables: list[str] | None = None,
    ):
        self._connection_factory = connection_factory
        self._dialect = dialect
        self.allowed_tables = set(allowed_tables) if allowed_tables else None

    @property
    def dialect(self) -> str:
        return self._dialect

    def load_schema(self) -> dict[str, set[str]]:
        conn = self._connection_factory()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' ORDER BY table_name, ordinal_position"
            )
            schema: dict[str, set[str]] = {}
            for table, col in cursor.fetchall():
                if self.allowed_tables and table not in self.allowed_tables:
                    continue
                schema.setdefault(table, set()).add(col)
            return schema
        finally:
            conn.close()

    def describe_schema(self) -> str:
        schema = self.load_schema()
        lines = []
        for table in sorted(schema):
            lines.append(f"{table}({', '.join(sorted(schema[table]))})")
        return "\n".join(lines)

    def run_query(self, sql: str) -> dict[str, Any]:
        statement = validate_select_only(sql, dialect=self.dialect)
        schema = self.load_schema()
        validate_schema_references(statement, schema)

        conn = self._connection_factory()
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            conn.close()

        tables = sorted({t.name for t in statement.find_all(exp.Table)})
        return {
            "query": sql,
            "rows": rows,
            "row_count": len(rows),
            "tables": tables,
        }
