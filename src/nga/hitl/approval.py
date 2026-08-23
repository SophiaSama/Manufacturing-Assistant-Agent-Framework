"""CLI-based human-in-the-loop approval for NGA safety-critical recommendations."""

from __future__ import annotations


def request_approval(
    recommendation_summary: str,
    *,
    class_a_alert: bool = False,
    escalation_level: str | None = None,
) -> tuple[str, str | None]:
    """Prompt an authorized person to approve or reject a recommendation.

    For Class A defects, a mandatory justification note is required on rejection.
    Returns (status, note) where status is 'approved', 'rejected', or 'pending'.
    """
    banner = ""
    if class_a_alert:
        banner = "⚠️  CLASS A SAFETY-CRITICAL ACTION — Requires authorized approval\n"
    if escalation_level:
        banner += f"Escalation Level: {escalation_level}\n"

    answer = (
        input(
            f"\n{banner}"
            f"Recommendation: {recommendation_summary}\n"
            "Approve this action? [y/n, empty=skip]: "
        )
        .strip()
        .lower()
    )

    if answer == "y":
        note = None
        if class_a_alert:
            note = input("Approver ID / name (required for Class A): ").strip() or None
        return "approved", note

    if answer == "n":
        prompt = (
            "Mandatory rejection reason (Class A requires justification): "
            if class_a_alert
            else "Optional rejection note: "
        )
        note = input(prompt).strip() or None
        return "rejected", note

    return "pending", None
