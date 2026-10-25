"""Memory management module for Enterprise Agent.

Provides:
- Episodic semantic memory (Headroom Memory + SQLite fallback)
- Persistent SQLite conversation checkpointer
- Centralized decisions and recommendations audit log
"""

from __future__ import annotations

from enterprise_agent.memory.checkpointer import build_checkpointer
from enterprise_agent.memory.decision_log import (
    get_decision_by_id,
    get_decisions,
    init_decision_log,
    insert_recommendation,
    update_decision,
)
from enterprise_agent.memory.episodic import (
    EpisodicMemoryStore,
    NgaEpisodicMemory,
)

__all__ = [
    "EpisodicMemoryStore",
    "NgaEpisodicMemory",
    "build_checkpointer",
    "init_decision_log",
    "insert_recommendation",
    "update_decision",
    "get_decisions",
    "get_decision_by_id",
]
