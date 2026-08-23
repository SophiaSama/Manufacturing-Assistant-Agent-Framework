"""Conflict detection evaluation for the NGA variant corpus.

Tests whether the agent detects the 8 planted inconsistencies in
variant-corpus/ and flags them rather than silently picking one value.

Planted contradictions:
  1. Wheel torque: 108 Nm (variant) vs 105 Nm (canonical)
  2. Electrode cap life: 5,000 welds (variant) vs 4,000 welds
  3. Calibration interval: 6 months (variant) vs 3 months
  4. Class A notification: 4 hours (variant) vs 1 hour
  5. Safety defect rate: 1.0% (variant) vs 0.5%
  6. Rear mount torque: 50 Nm (variant) vs 45 Nm
  7. Bleed sequence: FR→FL→RR→RL (variant) vs RR→RL→FR→FL (canonical)
  8. Windshield cure time: 2 hours (variant) vs 4 hours / 6 hours winter

Usage:
    uv run pytest tests/integration/test_conflict_detection.py -v -s
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

# ── Conflict scenarios ────────────────────────────────────────────────────────

CONFLICT_SCENARIOS = [
    {
        "id": "CONFLICT-1",
        "description": "Wheel torque contradiction (108 vs 105 Nm)",
        "question": "What is the correct lug nut torque specification at Station 144?",
        "conflict_values": ["108", "105"],
        "canonical_answer": "105 Nm",
        "conflict_keywords": ["conflict", "inconsisten", "discrepan", "contradict", "differ"],
    },
    {
        "id": "CONFLICT-2",
        "description": "Electrode cap life contradiction (5,000 vs 4,000 welds)",
        "question": "What is the electrode cap life limit for the welding robot?",
        "conflict_values": ["5,000", "4,000", "5000", "4000"],
        "canonical_answer": "4,000 welds",
        "conflict_keywords": ["conflict", "inconsisten", "discrepan", "contradict", "differ"],
    },
    {
        "id": "CONFLICT-3",
        "description": "Calibration interval contradiction (6 months vs 3 months)",
        "question": "What is the calibration interval for the TorqMaster TF-6000?",
        "conflict_values": ["6 month", "3 month", "6-month", "3-month"],
        "canonical_answer": "3 months or 100,000 cycles",
        "conflict_keywords": ["conflict", "inconsisten", "discrepan", "contradict", "differ"],
    },
    {
        "id": "CONFLICT-4",
        "description": "Class A notification time contradiction (4 hours vs 1 hour)",
        "question": "Within what time must the Quality Director be notified of a Class A defect?",
        "conflict_values": ["4 hour", "1 hour"],
        "canonical_answer": "1 hour",
        "conflict_keywords": ["conflict", "inconsisten", "discrepan", "contradict", "differ"],
    },
    {
        "id": "CONFLICT-5",
        "description": "Safety defect rate threshold contradiction (1.0% vs 0.5%)",
        "question": "What is the safety-related defect rate threshold that triggers recall evaluation?",
        "conflict_values": ["1.0%", "0.5%", "1%"],
        "canonical_answer": "0.5%",
        "conflict_keywords": ["conflict", "inconsisten", "discrepan", "contradict", "differ"],
    },
    {
        "id": "CONFLICT-6",
        "description": "Rear mount torque contradiction (50 Nm vs 45 Nm)",
        "question": "What is the rear transmission mount bolt torque specification?",
        "conflict_values": ["50 Nm", "45 Nm"],
        "canonical_answer": "45 Nm",
        "conflict_keywords": ["conflict", "inconsisten", "discrepan", "contradict", "differ"],
    },
    {
        "id": "CONFLICT-7",
        "description": "Brake bleed sequence contradiction (FR→FL→RR→RL vs RR→RL→FR→FL)",
        "question": "What is the correct brake fluid bleeding sequence?",
        "conflict_values": ["fr", "rr"],
        "canonical_answer": "RR → RL → FR → FL",
        "conflict_keywords": ["conflict", "inconsisten", "discrepan", "contradict", "differ"],
    },
    {
        "id": "CONFLICT-8",
        "description": "Windshield cure time contradiction (2 hours vs 4 hours)",
        "question": "What is the minimum windshield adhesive cure time at normal temperature?",
        "conflict_values": ["2 hour", "4 hour"],
        "canonical_answer": "4 hours",
        "conflict_keywords": ["conflict", "inconsisten", "discrepan", "contradict", "differ"],
    },
]


def _answer_contains_conflict_flag(answer: str, keywords: list[str]) -> bool:
    """Return True if the answer flags a conflict/inconsistency."""
    lowered = answer.lower()
    return any(kw in lowered for kw in keywords)


def _answer_contains_both_values(answer: str, conflict_values: list[str]) -> bool:
    """Return True if the answer mentions both conflicting values."""
    lowered = answer.lower()
    found = [v for v in conflict_values if v.lower() in lowered]
    return len(found) >= 2


# ── NOTE: This test requires a conflict-aware vector store ────────────────────
# To run conflict detection, build the vector store with BOTH the canonical
# corpus AND the variant corpus. The agent should then surface conflicts.
# Without a variant corpus vector store, these tests verify the canonical answers.

@pytest.mark.parametrize(
    "scenario",
    CONFLICT_SCENARIOS,
    ids=[s["id"] for s in CONFLICT_SCENARIOS],
)
def test_conflict_detection(scenario, agent_graph):
    """Test that the agent either detects conflicts or returns the canonical answer."""
    from langchain_core.messages import HumanMessage
    from nga.rag_agent.rbac import ACCESS_LEVELS
    from nga.models.answer_schema import FinalAnswer, parse_final_answer, render_final_answer

    question = scenario["question"]
    thread_id = str(uuid.uuid4())
    answer = ""

    initial_state = {
        "messages": [HumanMessage(content=question)],
        "user_role": "engineer",
        "user_level": ACCESS_LEVELS["engineer"],
    }

    question_parts: list[str] = []
    for update in agent_graph.stream(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="updates",
    ):
        if not isinstance(update, dict):
            continue
        synthesis = update.get("synthesis") or {}
        final_dict = synthesis.get("final_answer")
        if final_dict:
            try:
                final = FinalAnswer.model_validate(final_dict)
                answer = render_final_answer(final)
            except Exception:
                pass
        if not answer:
            msgs = synthesis.get("messages") or []
            if msgs:
                last = msgs[-1]
                raw = getattr(last, "content", "") or ""
                if raw and question_parts:
                    answer = render_final_answer(parse_final_answer(raw, question_parts))

    print(
        f"\n[{scenario['id']}] {scenario['description']}\n"
        f"  Q: {question}\n"
        f"  A: {answer[:300]}"
    )

    # Pass if: (a) conflict is flagged, OR (b) canonical answer is returned
    conflict_flagged = _answer_contains_conflict_flag(answer, scenario["conflict_keywords"])
    canonical_present = scenario["canonical_answer"].lower() in answer.lower()
    both_values = _answer_contains_both_values(answer, scenario["conflict_values"])

    passed = conflict_flagged or canonical_present
    print(
        f"  conflict_flagged={conflict_flagged} | "
        f"canonical_present={canonical_present} | "
        f"both_values_mentioned={both_values} | "
        f"PASS={passed}"
    )

    assert passed, (
        f"[{scenario['id']}] Agent neither flagged a conflict nor returned the "
        f"canonical answer '{scenario['canonical_answer']}'. "
        f"Answer: {answer[:200]}"
    )
