"""Unit tests verifying independent session memory, turn isolation, and SQL column alias support."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest
import sqlglot
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from enterprise_agent.database.engine import (
    SqlValidationError,
    validate_schema_references,
)
from nga.graph.orchestrator import (
    _count_tool_rounds,
    _get_current_turn_messages,
    _tool_loop_detected,
)
from nga.memory.decision_log import (
    get_decision_by_id,
    get_decisions,
    init_decision_log,
    insert_recommendation,
)

# ── 1. SQL Column Alias Validation Unit Tests ─────────────────────────────────

def test_sql_alias_in_order_by_accepted():
    """Verify that column aliases defined in SELECT are accepted in ORDER BY."""
    schema = {
        "defects": {"defect_class", "defect_code", "description", "status", "vin"},
        "quality_checks": {"check_type", "result", "measured_value", "vin"},
    }

    # Query with alias 'cnt' in ORDER BY
    q1 = "SELECT defect_class, COUNT(*) AS cnt, status FROM defects GROUP BY defect_class, status ORDER BY cnt DESC;"
    stmt1 = sqlglot.parse_one(q1)
    # Should not raise SqlValidationError
    validate_schema_references(stmt1, schema)

    # Query with alias 'num_checks' in ORDER BY
    q2 = "SELECT check_type, result, COUNT(*) AS num_checks FROM quality_checks GROUP BY check_type, result ORDER BY num_checks DESC;"
    stmt2 = sqlglot.parse_one(q2)
    validate_schema_references(stmt2, schema)


def test_sql_alias_in_having_accepted():
    """Verify that column aliases defined in SELECT are accepted in HAVING clauses."""
    schema = {
        "defects": {"defect_class", "defect_code", "status", "vin"},
    }
    q = "SELECT defect_class, COUNT(*) AS defect_count FROM defects GROUP BY defect_class HAVING defect_count > 2;"
    stmt = sqlglot.parse_one(q)
    validate_schema_references(stmt, schema)


def test_sql_unknown_column_still_rejected():
    """Verify that non-existent columns that are NOT aliases are still strictly rejected."""
    schema = {
        "defects": {"defect_class", "defect_code", "status"},
    }
    # 'non_existent_col' is neither a schema column nor a defined alias
    q = "SELECT defect_class, non_existent_col FROM defects;"
    stmt = sqlglot.parse_one(q)
    with pytest.raises(SqlValidationError) as exc_info:
        validate_schema_references(stmt, schema)
    assert "Unknown column `non_existent_col`" in str(exc_info.value)


# ── 2. Decision Log & Migration Unit Tests ────────────────────────────────────

def test_decision_log_migration_and_thread_isolation():
    """Verify thread_id column creation, auto-migration on existing databases, and filtering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test_app_state.db")

        # Create table WITHOUT thread_id to simulate a pre-existing legacy database
        con = sqlite3.connect(db_path)
        con.execute(
            """
            CREATE TABLE decisions_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                category TEXT NOT NULL,
                user_role TEXT NOT NULL DEFAULT 'operator',
                class_a_alert INTEGER NOT NULL DEFAULT 0,
                escalation_level TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                approver TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        con.commit()
        con.close()

        # Run init_decision_log which should migrate and add thread_id column
        init_decision_log(db_path)

        # Verify column now exists
        con = sqlite3.connect(db_path)
        cursor = con.execute("PRAGMA table_info(decisions_log)")
        cols = {row[1] for row in cursor.fetchall()}
        con.close()
        assert "thread_id" in cols

        # Insert records across two different threads
        id_1 = insert_recommendation(
            db_path,
            question="Question for Session 1",
            recommendation="Recommendation 1",
            category="SOP",
            thread_id="session-aaa-111",
        )
        id_2 = insert_recommendation(
            db_path,
            question="Question for Session 2",
            recommendation="Recommendation 2",
            category="SOP",
            thread_id="session-bbb-222",
        )

        # Query by thread_id
        session_1_records = get_decisions(db_path, thread_id="session-aaa-111")
        assert len(session_1_records) == 1
        assert session_1_records[0]["id"] == id_1
        assert session_1_records[0]["thread_id"] == "session-aaa-111"
        assert session_1_records[0]["question"] == "Question for Session 1"

        session_2_records = get_decisions(db_path, thread_id="session-bbb-222")
        assert len(session_2_records) == 1
        assert session_2_records[0]["id"] == id_2
        assert session_2_records[0]["thread_id"] == "session-bbb-222"

        # Query single by ID
        rec = get_decision_by_id(db_path, id_1)
        assert rec is not None
        assert rec["thread_id"] == "session-aaa-111"


# ── 3. Orchestrator Turn-Isolation Unit Tests ─────────────────────────────────

def test_get_current_turn_messages():
    """Verify that _get_current_turn_messages extracts only messages from the latest turn."""
    # Multi-turn conversation history
    history = [
        HumanMessage(content="Turn 1: hello"),
        AIMessage(content="Turn 1 response"),
        HumanMessage(content="Turn 2: check shift A"),
        AIMessage(content="Turn 2 looking up data", tool_calls=[{"name": "query_db", "args": {}, "id": "c1"}]),
        ToolMessage(content='{"result": "ok"}', tool_call_id="c1"),
    ]

    current_turn = _get_current_turn_messages(history)
    assert len(current_turn) == 3
    assert isinstance(current_turn[0], HumanMessage)
    assert current_turn[0].content == "Turn 2: check shift A"


def test_count_tool_rounds_turn_scoped():
    """Verify tool rounds from earlier turns do not contaminate the current turn count."""
    history = [
        # Turn 1 had 2 tool rounds
        HumanMessage(content="Turn 1 query"),
        AIMessage(content="", tool_calls=[{"name": "t1", "args": {}, "id": "c1"}]),
        ToolMessage(content="r1", tool_call_id="c1"),
        AIMessage(content="", tool_calls=[{"name": "t2", "args": {}, "id": "c2"}]),
        ToolMessage(content="r2", tool_call_id="c2"),
        AIMessage(content="Turn 1 final answer"),
        # Turn 2 starts fresh
        HumanMessage(content="Turn 2 query"),
        AIMessage(content="", tool_calls=[{"name": "t3", "args": {}, "id": "c3"}]),
        ToolMessage(content="r3", tool_call_id="c3"),
    ]

    # Overall history has 3 tool rounds, but current turn only has 1
    current_rounds = _count_tool_rounds(history)
    assert current_rounds == 1


def test_tool_loop_detected_turn_scoped():
    """Verify tool loop detection only evaluates repeating tool calls within the current turn."""
    # Turn 1 had repeating calls that were resolved
    history = [
        HumanMessage(content="Turn 1"),
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"key": "same"}, "id": "c1"}]),
        ToolMessage(content="err", tool_call_id="c1"),
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"key": "same"}, "id": "c2"}]),
        ToolMessage(content="err", tool_call_id="c2"),
        AIMessage(content="Turn 1 finished"),
        # Turn 2 has a single call with the same signature
        HumanMessage(content="Turn 2"),
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"key": "same"}, "id": "c3"}]),
        ToolMessage(content="ok", tool_call_id="c3"),
    ]

    # Since turn 2 only called it once in this turn, loop detection should be False
    is_loop = _tool_loop_detected(history)
    assert is_loop is False
