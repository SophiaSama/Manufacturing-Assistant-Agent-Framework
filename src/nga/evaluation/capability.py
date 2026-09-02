"""Model capability tiers for integration-test gating.

Docs: docs/model-capability-gating-design.md

Integration suites declare a required tier; tests are skipped when the
configured model cannot meet it (e.g., llama3.2:3b is 'basic').

Tiers (monotonic):
    basic     — single-hop lookup, no strict structured output
    standard  — multi-hop retrieval + tool use + structured FinalAnswer JSON
    strong    — + cross-document contradiction reasoning, schema-aware SQL
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("nga.evaluation.capability")

TIERS = ("basic", "standard", "strong")
TIER_RANK: dict[str, int] = {tier: i for i, tier in enumerate(TIERS)}

DEFAULT_CLOUD_TIER = "standard"   # fail-open for unknown cloud providers
DEFAULT_LOCAL_TIER = "basic"      # fail-closed for unknown local models

# Known cloud model families → tier (matched by lowercase substring)
CLOUD_ALLOWLIST: dict[str, str] = {
    "claude-haiku": "standard",
    "claude-sonnet": "strong",
    "claude-opus": "strong",
    "gpt-4o": "strong",
    "gpt-4.1": "strong",
    "gpt-5": "strong",
    "gemini-1.5-pro": "strong",
    "gemini-2.5-pro": "strong",
    "gemini-2.5-flash": "standard",
    "deepseek-v4": "strong",
    "deepseek-v3": "standard",
    "llama-3.3": "strong",
    "qwen3": "strong",
    "qwen2.5-coder": "strong",
}

# Local model parameter-size thresholds (billions) → tier.
# Evidence (TEST_REVIEW_SUMMARY): llama3.2:3b (3.21B) fails integration, so
# <4B → basic; 4B ≤ size < 7B → standard; ≥7B → strong.
LOCAL_PARAMS_RANK = (
    (4.0, "basic"),      # < 4B
    (7.0, "standard"),   # 4B ≤ size < 7B
)
LOCAL_STRONG_THRESHOLD = 7.0  # ≥ 7B → strong

_PARAMS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]$", re.ASCII)


def parse_local_params(model_id: str) -> float | None:
    """Extract parameter count in billions from a local model id.

    Handles 'llama3.2:3b', 'qwen2.5:7b-instruct', 'llama3.1:70b' style names.
    The size token must appear at the end (after the last separator).
    """
    tokens = re.split(r"[:/-]", model_id.strip())
    for token in reversed(tokens):
        m = _PARAMS_RE.search(token)
        if m:
            return float(m.group(1))
    return None


def cloud_allowlist_tier(model_id: str) -> str | None:
    lowered = model_id.lower()
    if "embedding" in lowered:
        return None  # embedding models are never chat-capable tiers
    for family, tier in CLOUD_ALLOWLIST.items():
        if family in lowered:
            return tier
    return None


def resolve_tier(
    model_id: str | None,
    provider: str,
    override: str | None = None,
) -> str:
    """Resolve the capability tier for a model.

    Resolution order:
      1. explicit override (env MODEL_CAPABILITY_OVERRIDE)
      2. known cloud allowlist (substring match)
      3. local: parameter size from name
      4. unknown cloud → 'standard' (fail-open)
         unknown local  → 'basic'   (fail-closed)
    """
    if override:
        norm = override.strip().lower()
        if norm in TIER_RANK:
            return norm
        logger.warning("Ignoring unknown capability override %r", override)

    model = (model_id or "").strip()
    if not model:
        return DEFAULT_LOCAL_TIER

    if provider and provider.lower() == "ollama":
        params = parse_local_params(model)
        if params is None:
            return DEFAULT_LOCAL_TIER
        if params >= LOCAL_STRONG_THRESHOLD:
            return "strong"
        for threshold, tier in LOCAL_PARAMS_RANK:
            if params < threshold:
                return tier
        return "standard"

    # cloud / openrouter / unknown provider
    allowlist = cloud_allowlist_tier(model)
    if allowlist is not None:
        return allowlist
    # fall back to local-style size parsing (some cloud ids embed sizes)
    params = parse_local_params(model)
    if params is not None:
        if params >= LOCAL_STRONG_THRESHOLD:
            return "strong"
        if params < 4.0:
            return "basic"
    return DEFAULT_CLOUD_TIER


def tier_meets(active: str, required: str) -> bool:
    """True when active tier is >= required tier."""
    return TIER_RANK.get(active, 0) >= TIER_RANK.get(required, 99)


def require_tier(required: str):
    """Return a pytest marker that skips when the active model is below `required`.

    Reads the active tier from the `active_capability` fixture via a marker
    evaluated lazily inside `pytest_collection_modifyitems`, or — when used
    directly as `pytestmark` — via skipif against a module-level value set by
    conftest's `active_capability` fixture is NOT possible at import time, so
    we use a dynamic marker consumed in conftest hook.
    """
    import pytest

    if required not in TIER_RANK:
        raise ValueError(f"Unknown required tier: {required!r}")
    return pytest.mark.required_tier(required)


def describe_reason(active: str, required: str, model_id: str | None) -> str:
    return (
        f"model {model_id or '(unknown)'!r} resolves to tier '{active}'; "
        f"test requires '{required}'"
    )
