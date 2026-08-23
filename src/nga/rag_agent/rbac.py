"""RBAC definitions for the NGA Manufacturing Assistant.

4-tier role hierarchy: operator → technician → engineer → manager.
Access levels derived server-side from document folder paths.
"""

from __future__ import annotations

# ── Access level hierarchy ────────────────────────────────────────────────────
ACCESS_LEVELS: dict[str, int] = {
    "operator": 1,    # Assembly-line operators — SOPs only
    "technician": 2,  # Maintenance technicians — troubleshooting + calibration
    "engineer": 3,    # Process / quality engineers — RCA, 8D, supplier quality
    "manager": 4,     # Plant manager / quality director — recall, stop-ship, regulatory
}

LEVEL_NAMES: dict[int, str] = {v: k for k, v in ACCESS_LEVELS.items()}

ROLE_PERMISSIONS: dict[str, list[str]] = {
    "manager":    ["operator", "technician", "engineer", "manager"],
    "engineer":   ["operator", "technician", "engineer"],
    "technician": ["operator", "technician"],
    "operator":   ["operator"],
}

# ── Role-specific system prompts ─────────────────────────────────────────────
SYSTEM_PROMPTS: dict[str, str] = {
    "manager": (
        "You are the Plant Manager / Quality Director decision-support assistant for "
        "Apex Automotive — Northgate Assembly Plant (NGA). "
        "You have full access to all documentation, production data, and regulatory guidance. "
        "You can evaluate recall trigger criteria (QCR-501 C1–C5), authorize stop-ship decisions, "
        "and review field actions. NRSA regulatory timelines (5 business days for safety defects) "
        "must always be highlighted. "
        "Always cite the specific document ID and section for every decision. "
        "For Class A defects, always state whether Level 4 escalation and joint PM/QD approval are required."
    ),
    "engineer": (
        "You are a Process / Quality Engineer assistant for Northgate Assembly Plant. "
        "You have access to all SOPs, machine specs, failure analysis procedures (FAP-401), "
        "supplier quality data, and training records. "
        "You can perform root-cause analysis (8D / 5-Why / Ishikawa), evaluate NC records, "
        "and assess defect classification (Class A/B/C). "
        "Cite every claim with document ID, section, and relevant data from the NGA database. "
        "Do not speculate beyond the provided context."
    ),
    "technician": (
        "You are a Maintenance Technician assistant for Northgate Assembly Plant. "
        "Answer only with step-by-step maintenance and troubleshooting procedures. "
        "You have access to technician SOPs (SOP-TEC-*) and machine specifications (TEC-*). "
        "Do not perform root-cause analysis or evaluate recall criteria — escalate to engineering when needed. "
        "Cite document IDs for every procedure step."
    ),
    "operator": (
        "You are an Assembly Operator assistant for Northgate Assembly Plant. "
        "Answer only with step-by-step assembly procedures from operator SOPs (SOP-OPR-*). "
        "Do not discuss machine internals, root-cause analysis, or quality escalation procedures. "
        "If you cannot answer from the provided SOPs, say so and advise escalation to your supervisor. "
        "Always cite the SOP document ID and revision."
    ),
}

# ── Folder-to-access-level mapping ───────────────────────────────────────────
# Derived server-side from file paths — never accepted from caller input.
FOLDER_LEVEL_MAP: dict[str, str] = {
    "operator-sops/":           "operator",
    "technician-sops/":         "technician",
    "machine-details/":         "technician",
    "failure-analysis/":        "engineer",
    "additional-docs/maintenance-work-orders/": "technician",
    "additional-docs/supplier-quality/":        "engineer",
    "additional-docs/training/":                "engineer",
    "recall-quality/":          "manager",
}

_SAFE_DEFAULT_LEVEL = "operator"


def level_from_path(path: str) -> str:
    """Derive the access level from a document file path.

    Checks each prefix in FOLDER_LEVEL_MAP. Falls back to 'operator' (most
    restrictive) if no prefix matches. Always derived server-side.
    """
    for prefix, level in FOLDER_LEVEL_MAP.items():
        if prefix in path or path.endswith(prefix.rstrip("/")):
            return level
    return _SAFE_DEFAULT_LEVEL


def max_level_rank(role: str) -> int:
    """Return the maximum level_rank the given role is allowed to see."""
    return ACCESS_LEVELS.get(role, ACCESS_LEVELS[_SAFE_DEFAULT_LEVEL])


# ── High-consequence action keywords for HITL gate ───────────────────────────
# The orchestrator checks for these verbs to identify recommendations
# that must be logged and approved before execution.
HITL_ACTION_VERBS: list[str] = [
    "stop-ship",
    "stop ship",
    "quarantine",
    "escalate",
    "recall",
    "rework",
    "scrap",
    "halt",
    "hold",
    "notify nrsa",
    "field action",
    "level 4",
    "level 3",
    "8d",
]

# ── Class A safety-critical systems (per QCR-501) ────────────────────────────
CLASS_A_SYSTEMS: list[str] = [
    "brake", "steering", "airbag", "seat belt", "fuel",
    "wheel retention", "engine mount", "windshield retention",
]
