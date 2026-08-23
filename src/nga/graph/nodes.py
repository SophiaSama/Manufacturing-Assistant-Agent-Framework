"""Node helper functions for the NGA Manufacturing Assistant orchestrator."""

from __future__ import annotations

import re

# High-consequence action verbs that trigger HITL approval
_HITL_ACTION_VERBS = [
    "stop-ship", "stop ship", "quarantine", "escalate", "recall",
    "rework", "scrap", "halt", "hold line", "notify nrsa", "field action",
    "level 4", "level 3", "open nc", "8d", "stop station",
]

_QUESTION_STARTERS = (
    "what", "which", "whether", "do", "does", "is", "are", "should",
    "can", "will", "how", "why", "when", "where", "who",
)
_IMPERATIVE_STARTERS = (
    "identify", "list", "confirm", "provide", "determine",
    "explain", "describe", "walk through", "evaluate", "compute",
)


def extract_question_parts(question: str) -> list[str]:
    """Split a compound user question into explicit sub-questions."""
    normalized = " ".join(question.split())
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    question_parts: list[str] = []
    starters_pattern = "|".join(_QUESTION_STARTERS + _IMPERATIVE_STARTERS)

    for sentence in sentences:
        stripped = sentence.strip().rstrip(".?!")
        if not stripped:
            continue
        if not stripped.lower().startswith(_QUESTION_STARTERS + _IMPERATIVE_STARTERS):
            match = re.search(
                rf"\b({starters_pattern})\b", stripped, flags=re.IGNORECASE
            )
            if match:
                stripped = stripped[match.start():]
        subparts = re.split(
            rf",\s+(?=(?:and\s+)?(?:{starters_pattern}|the)\b)",
            stripped,
            flags=re.IGNORECASE,
        )
        for part in subparts:
            candidate = re.sub(
                r"^and\s+", "", part.strip(), flags=re.IGNORECASE
            ).rstrip(".?!")
            lowered = candidate.lower()
            if lowered.startswith("whether "):
                question_parts.append(f"Do {candidate[len('whether '):].strip()}?")
                continue
            if lowered.startswith("the "):
                question_parts.append(f"What are {candidate}?")
                continue
            if lowered.startswith(_QUESTION_STARTERS + _IMPERATIVE_STARTERS):
                question_parts.append(f"{candidate[0].upper()}{candidate[1:]}?")

    if question_parts:
        return question_parts
    fallback = normalized.rstrip(".?!")
    return [f"{fallback}?"] if fallback else []


def extract_recommendation(answer_text: str) -> str | None:
    """Return the first sentence containing a high-consequence actionable recommendation."""
    sentences = re.split(r"(?<=[.!?])\s+", answer_text)
    for sentence in sentences:
        lowered = sentence.lower()
        if any(verb in lowered for verb in _HITL_ACTION_VERBS):
            return sentence.strip()
    return None


def is_class_a_defect(text: str) -> bool:
    """Return True if the text indicates a Class A (safety-critical) defect."""
    class_a_systems = [
        "brake", "steering", "airbag", "seat belt", "fuel",
        "wheel retention", "engine mount", "windshield retention",
    ]
    lowered = text.lower()
    return "class a" in lowered or any(s in lowered for s in class_a_systems)
