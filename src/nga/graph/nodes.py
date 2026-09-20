"""Node helper functions for the NGA Manufacturing Assistant orchestrator.

Delegates core algorithms to enterprise_agent.
"""

from __future__ import annotations

from enterprise_agent.graph.nodes import (
    extract_question_parts as _extract_question_parts_generic,
)
from enterprise_agent.hitl.policy_gate import (
    extract_recommendation as _extract_recommendation_generic,
)

# High-consequence action verbs that trigger HITL approval
_HITL_ACTION_VERBS = [
    "stop-ship", "stop ship", "quarantine", "escalate", "recall",
    "rework", "scrap", "halt", "hold line", "notify nrsa", "field action",
    "level 4", "level 3", "open nc", "8d", "stop station",
]


def extract_question_parts(question: str) -> list[str]:
    """Split a compound user question into explicit sub-questions."""
    return _extract_question_parts_generic(question)


def extract_recommendation(answer_text: str) -> str | None:
    """Return the first sentence containing a high-consequence actionable recommendation."""
    return _extract_recommendation_generic(answer_text, action_verbs=_HITL_ACTION_VERBS)


def is_class_a_defect(text: str) -> bool:
    """Return True if the text indicates a Class A (safety-critical) defect."""
    class_a_systems = [
        "brake", "steering", "airbag", "seat belt", "fuel",
        "wheel retention", "engine mount", "windshield retention",
    ]
    lowered = text.lower()
    return "class a" in lowered or any(s in lowered for s in class_a_systems)
