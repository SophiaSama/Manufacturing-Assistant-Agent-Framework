"""Persistent long-term memory: decisions_log table in app_state.db."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_decision_log(db_path: str) -> None:
    con = _connect(db_path)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions_log (
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
    finally:
        con.close()


def insert_recommendation(
    db_path: str,
    question: str,
    recommendation: str,
    category: str,
    user_role: str = "operator",
    class_a_alert: bool = False,
    escalation_level: str | None = None,
) -> int:
    now = datetime.now(timezone.utc).isoformat()
    con = _connect(db_path)
    try:
        cursor = con.execute(
            """
            INSERT INTO decisions_log
                (question, recommendation, category, user_role,
                 class_a_alert, escalation_level, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
            """,
            (
                question, recommendation, category, user_role,
                int(class_a_alert), escalation_level, now, now,
            ),
        )
        con.commit()
        return cursor.lastrowid
    finally:
        con.close()


def update_decision(
    db_path: str,
    decision_id: int,
    status: str,
    approver: str | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    con = _connect(db_path)
    try:
        con.execute(
            """
            UPDATE decisions_log
            SET status = ?, approver = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, approver, now, decision_id),
        )
        con.commit()
    finally:
        con.close()


def get_decisions(
    db_path: str,
    status: str | None = None,
    limit: int = 50,
) -> list[dict[str, object]]:
    """Retrieve decision records from decisions_log table."""
    con = _connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        if status:
            cursor = con.execute(
                """
                SELECT id, question, recommendation, category, user_role,
                       class_a_alert, escalation_level, status, approver,
                       created_at, updated_at
                FROM decisions_log
                WHERE status = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (status, limit),
            )
        else:
            cursor = con.execute(
                """
                SELECT id, question, recommendation, category, user_role,
                       class_a_alert, escalation_level, status, approver,
                       created_at, updated_at
                FROM decisions_log
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        con.close()


def get_decision_by_id(db_path: str, decision_id: int) -> dict[str, object] | None:
    """Retrieve a single decision record by ID."""
    con = _connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        cursor = con.execute(
            """
            SELECT id, question, recommendation, category, user_role,
                   class_a_alert, escalation_level, status, approver,
                   created_at, updated_at
            FROM decisions_log
            WHERE id = ?
            """,
            (decision_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        con.close()

