"""Heuristic query complexity classifier for NGA Manufacturing Assistant.

Runs synchronously in <5ms — no LLM call. Produces an integer score
0–10 that the router maps to a model tier (Haiku / Sonnet / Opus).
"""

from __future__ import annotations

import re

CONFIG_SIGNALS: dict[str, dict] = {
    "chained_keywords": {
        "words": [
            "why", "compare", "root cause", "trace", "analyze", "diagnose",
            "explain how", "step by step", "investigate", "impact of", "assess",
            "correlate", "trend", "8d", "recall", "escalation", "containment",
            "walk through", "which criteria", "scope", "stop-ship",
        ],
        "weight": 3,
        "max_matches": 2,
    },
    "lookup_keywords": {
        "words": [
            "what is", "define", "status of", "when was", "who is",
            "list", "show me", "what are", "how many", "what torque",
            "what error", "what code",
        ],
        "weight": -2,
        "max_matches": 2,
    },
    "query_length": {
        "threshold": 15,
        "weight": 2,
    },
    "entity_count": {
        "threshold": 3,
        "weight": 2,
    },
    "question_count": {
        "threshold": 2,
        "weight": 3,
    },
}

_NEUTRAL_BASELINE = 5
_NAMED_ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b")

# NGA-specific named entities that strongly increase complexity
_NGA_COMPLEX_ENTITIES = re.compile(
    r"\b(NC-\d+|WO-\d+|QCR-501|ESC-402|FAP-401|SOP-[A-Z]+-\d+|TEC-\d+|"
    r"NGA-[A-Z0-9-]+|FA-\d+|SCAR-\d+|AU-\d+|SO-\d+|E-\d{4}|P-\d{3}|"
    r"Class A|Class B|NRSA|C1|C2|C3|C4|C5)\b",
    re.IGNORECASE,
)


def score_complexity(query: str) -> int:
    """Score query complexity 0–10.

    Higher scores route to more capable (and slower) model tiers.
    """
    score = _NEUTRAL_BASELINE
    q_lower = query.lower()
    words = q_lower.split()

    # Complexity-increasing keywords
    chained = CONFIG_SIGNALS["chained_keywords"]
    matches = sum(1 for kw in chained["words"] if kw in q_lower)
    score += min(matches, chained["max_matches"]) * chained["weight"]

    # Lookup (complexity-reducing) keywords
    lookup = CONFIG_SIGNALS["lookup_keywords"]
    matches = sum(1 for kw in lookup["words"] if kw in q_lower)
    score += min(matches, lookup["max_matches"]) * lookup["weight"]

    # Query length
    length_cfg = CONFIG_SIGNALS["query_length"]
    if len(words) > length_cfg["threshold"]:
        score += length_cfg["weight"]

    # Named entity count
    entity_cfg = CONFIG_SIGNALS["entity_count"]
    entities = _NAMED_ENTITY_PATTERN.findall(query)
    if len(entities) >= entity_cfg["threshold"]:
        score += entity_cfg["weight"]

    # NGA-specific complex identifiers
    nga_entities = _NGA_COMPLEX_ENTITIES.findall(query)
    if len(nga_entities) >= 2:
        score += 2

    # Multiple question marks → compound question
    q_cfg = CONFIG_SIGNALS["question_count"]
    if query.count("?") >= q_cfg["threshold"]:
        score += q_cfg["weight"]

    return max(0, min(10, score))
