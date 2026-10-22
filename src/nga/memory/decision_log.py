"""Backward-compatibility re-export for decision log."""

from enterprise_agent.memory.decision_log import (
    get_decision_by_id,
    get_decisions,
    init_decision_log,
    insert_recommendation,
    update_decision,
)

__all__ = [
    "init_decision_log",
    "insert_recommendation",
    "update_decision",
    "get_decisions",
    "get_decision_by_id",
]
