"""Model tier routing and query complexity classification for NGA Manufacturing Assistant.

Supports:
1. TypeSafe AI System One Choice classification (sub-50ms, calibrated semantic routing).
2. Fallback to local heuristic query scoring with a warning log when TypeSafe is unavailable or unconfigured.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from nga.config import Settings

logger = logging.getLogger(__name__)

# ── Heuristic Query Complexity Baseline ───────────────────────────────────────

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

_NGA_COMPLEX_ENTITIES = re.compile(
    r"\b(NC-\d+|WO-\d+|QCR-501|ESC-402|FAP-401|SOP-[A-Z]+-\d+|TEC-\d+|"
    r"NGA-[A-Z0-9-]+|FA-\d+|SCAR-\d+|AU-\d+|SO-\d+|E-\d{4}|P-\d{3}|"
    r"Class A|Class B|NRSA|C1|C2|C3|C4|C5)\b",
    re.IGNORECASE,
)


def score_complexity(query: str) -> int:
    """Score query complexity 0–10 using local heuristics.

    Higher scores route to more capable (and slower) model tiers.
    Used as deterministic fallback when TypeSafe API is unavailable.
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


# ── TypeSafe AI Model Router Configuration ────────────────────────────────────

ROUTER_INSTRUCTIONS = (
    "Choose the least costly model tier that can complete the manufacturing task safely, "
    "accurately, and without hallucination."
)

ROUTER_CHOICES_CRITERIA = {
    "fast": (
        "Direct lookups, single-document SOP procedures, torque specs, table row counts, "
        "and simple status inquiries with explicit targets."
    ),
    "balanced": (
        "Multi-document comparisons, standard maintenance troubleshooting, cross-checking "
        "specifications, and process verification."
    ),
    "powerful": (
        "Root-cause analysis (8D / 5-Why / Ishikawa), Class A safety defect assessments, "
        "novel root-cause reasoning, recall criteria evaluation (QCR-501 C1–C5), and stop-ship recommendations."
    ),
}


def _tier_choice_to_model_info(choice: str, settings: Settings) -> tuple[str, int, int]:
    """Map choice ('fast' | 'balanced' | 'powerful') to (model_slug, context_window, max_hops)."""
    if choice == "fast":
        return (
            settings.rag_tier1_model,
            settings.rag_tier1_context_window,
            settings.rag_tier1_max_hops,
        )
    elif choice == "powerful":
        return (
            settings.rag_tier3_model,
            settings.rag_tier3_context_window,
            settings.rag_tier3_max_hops,
        )
    # Default to balanced (tier 2)
    return (
        settings.rag_tier2_model,
        settings.rag_tier2_context_window,
        settings.rag_tier2_max_hops,
    )


def _heuristic_fallback_route(query: str, settings: Settings) -> dict[str, Any]:
    """Route query using score_complexity() when TypeSafe is unconfigured or fails."""
    score = score_complexity(query)
    if score <= 3:
        choice = "fast"
    elif score <= 6:
        choice = "balanced"
    else:
        choice = "powerful"

    model_slug, context_window, max_hops = _tier_choice_to_model_info(choice, settings)
    return {
        "choice": choice,
        "model_name": model_slug,
        "context_window": context_window,
        "max_hops": max_hops,
        "source": "heuristic_fallback",
        "heuristic_score": score,
        "confidence": None,
        "probabilities": None,
    }


def classify_model_route(
    query: str,
    settings: Settings,
    typesafe_client: Any | None = None,
) -> dict[str, Any]:
    """Classify the incoming query and select the optimal model tier.

    Uses TypeSafe System One Choice classification when TYPESAFE_API_KEY is configured.
    Falls back to score_complexity() with a warning log if TypeSafe is not available.
    """
    api_key = os.getenv("TYPESAFE_API_KEY")
    if not api_key and typesafe_client is None:
        logger.warning(
            "TYPESAFE_API_KEY is not set; falling back to local heuristic model routing."
        )
        return _heuristic_fallback_route(query, settings)

    try:
        if typesafe_client is None:
            from typesafe_sdk import Choice, TypeSafeClient
            client = TypeSafeClient(api_key=api_key)
        else:
            from typesafe_sdk import Choice
            client = typesafe_client

        questions = {
            "model_route": Choice(
                instructions=ROUTER_INSTRUCTIONS,
                criteria=ROUTER_CHOICES_CRITERIA,
            )
        }

        response = client.system_one(state=query, questions=questions)
        choice_ans = response.answers["model_route"]
        selected_choice = choice_ans.choice

        # Validate that the selected choice is recognized
        if selected_choice not in ROUTER_CHOICES_CRITERIA:
            selected_choice = "balanced"

        model_slug, context_window, max_hops = _tier_choice_to_model_info(
            selected_choice, settings
        )

        return {
            "choice": selected_choice,
            "model_name": model_slug,
            "context_window": context_window,
            "max_hops": max_hops,
            "source": "typesafe",
            "confidence": getattr(choice_ans, "confidence", None),
            "probabilities": getattr(choice_ans, "probabilities", {}),
        }

    except Exception as exc:
        logger.warning(
            "TypeSafe model routing failed (%s); falling back to local heuristic routing.",
            exc,
        )
        return _heuristic_fallback_route(query, settings)

