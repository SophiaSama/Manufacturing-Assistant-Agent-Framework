"""Generic Human-In-The-Loop (HITL) and Governance Policy Gate.

Intercepts high-consequence actions, validates safety/compliance triggers,
and manages approval workflows across arbitrary domains.
"""

from __future__ import annotations

import re
from typing import Any

from enterprise_agent.config.domain_config import GovernanceConfig
from enterprise_agent.models.answer_schema import GovernanceAlert


def extract_recommendation(
    answer_text: str,
    action_verbs: list[str] | None = None,
) -> str | None:
    """Return the first sentence containing a high-consequence actionable recommendation."""
    verbs = action_verbs or []
    sentences = re.split(r"(?<=[.!?])\s+", answer_text)
    for sentence in sentences:
        lowered = sentence.lower()
        if any(verb.lower() in lowered for verb in verbs):
            return sentence.strip()
    return None


def evaluate_governance(
    text: str,
    recommendation: str | None,
    config: GovernanceConfig,
) -> GovernanceAlert:
    """Evaluate text and recommendation against domain governance rules."""
    lowered_text = text.lower()
    lowered_rec = (recommendation or "").lower()
    full_text = f"{lowered_text}\n{lowered_rec}"

    # Critical triggers detection
    triggered_terms = [
        term for term in config.critical_terms if term.lower() in full_text
    ]
    is_critical = bool(triggered_terms)

    # HITL required action detection
    triggered_verb = None
    for verb in config.hitl_action_verbs:
        if verb.lower() in lowered_rec or verb.lower() in lowered_text:
            triggered_verb = verb
            break

    hitl_required = bool(triggered_verb) or is_critical

    # Detect escalation level if present (e.g. L1-L4 or domain-configured)
    escalation_tier = None
    for level in config.escalation_levels:
        pat = rf"\b{re.escape(level.lower())}\b"
        if re.search(pat, full_text):
            escalation_tier = level
            break

    return GovernanceAlert(
        is_critical=is_critical,
        critical_triggers=triggered_terms,
        escalation_tier=escalation_tier,
        hitl_required=hitl_required,
        action_verb=triggered_verb,
    )


def request_approval(
    recommendation_summary: str,
    *,
    governance: GovernanceAlert | None = None,
    config: GovernanceConfig | None = None,
    interactive_input: Any = input,
) -> tuple[str, str | None]:
    """Prompt an authorized user to approve or reject a recommendation.

    Returns: (status: 'approved' | 'rejected' | 'pending', note: str | None)
    """
    gov = governance or GovernanceAlert()
    cfg = config or GovernanceConfig()

    banner = ""
    if gov.is_critical:
        banner = "⚠️  CRITICAL ACTION / HIGH-CONSEQUENCE RECOMMENDATION — Requires Approval\n"
    if gov.escalation_tier:
        banner += f"Escalation Level: {gov.escalation_tier}\n"
    if gov.critical_triggers:
        banner += f"Triggers: {', '.join(gov.critical_triggers)}\n"

    answer = (
        interactive_input(
            f"\n{banner}"
            f"Recommendation: {recommendation_summary}\n"
            "Approve this action? [y/n, empty=skip]: "
        )
        .strip()
        .lower()
    )

    if answer == "y":
        note = None
        if gov.is_critical or cfg.justification_required:
            note = interactive_input("Approver ID / Authorization code: ").strip() or None
        return "approved", note

    if answer == "n":
        prompt = (
            "Mandatory rejection reason (critical action requires justification): "
            if gov.is_critical
            else "Optional rejection note: "
        )
        note = interactive_input(prompt).strip() or None
        return "rejected", note

    return "pending", None
