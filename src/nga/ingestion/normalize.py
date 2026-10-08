"""Shared normalization and ontology for the NGA knowledge graph.

Single source of truth used by graph construction, entity alignment and
evaluation so that IDs, entity types and relation rules never drift apart.
"""

from __future__ import annotations

import re
from typing import Any

NGA_ENTITY_TYPES = [
    "Machine", "Station", "SOP", "FaultCode", "DefectCode", "Part",
    "Supplier", "Personnel", "Torque", "Threshold", "Procedure",
    "NCRecord", "WorkOrder", "RecallCriteria", "EscalationLevel",
]

NGA_RELATION_TYPES = [
    "causes", "indicates", "prevents", "monitors", "threshold_for",
    "documented_in", "escalates_to", "requires", "audited_by",
    "owned_by", "certified_for", "triggers_recall",
]

# relation -> (valid source types, valid target types), all lowercase.
NGA_ONTOLOGY_RULES: dict[str, tuple[set[str], set[str]]] = {
    "causes": ({"faultcode", "defectcode", "machine"}, {"defectcode", "ncrecord", "machine"}),
    "indicates": ({"machine", "threshold", "sensor", "station"}, {"faultcode", "defectcode"}),
    "prevents": ({"sop", "procedure"}, {"faultcode", "defectcode", "ncrecord"}),
    "monitors": ({"machine", "station", "personnel"}, {"machine", "threshold", "torque", "station"}),
    "threshold_for": ({"sop", "procedure", "machine", "station", "part"}, {"threshold", "torque"}),
    "documented_in": (
        {"threshold", "torque", "machine", "faultcode", "sop", "station", "procedure", "defectcode", "ncrecord"},
        {"sop", "station", "procedure"},
    ),
    "escalates_to": ({"defectcode", "ncrecord", "faultcode"}, {"escalationlevel", "personnel", "role"}),
    "requires": (
        {"sop", "procedure", "defectcode", "ncrecord", "workorder"},
        {"machine", "procedure", "sop", "personnel", "torque", "part", "station"},
    ),
    "audited_by": ({"machine", "station", "torque", "workorder", "sop", "procedure"}, {"machine", "personnel"}),
    "triggers_recall": ({"defectcode", "ncrecord", "faultcode"}, {"recallcriteria"}),
}

_ENTITY_TYPE_LOOKUP = {t.lower(): t for t in NGA_ENTITY_TYPES}
_RELATION_LOOKUP = {r.lower(): r for r in NGA_RELATION_TYPES}


def ontology_prompt_text() -> str:
    """Human-readable direction rules, one line per relation, for LLM prompts."""
    lines = []
    for rel, (srcs, tgts) in NGA_ONTOLOGY_RULES.items():
        lines.append(f"- {rel}: {' | '.join(sorted(srcs))}  --{rel}-->  {' | '.join(sorted(tgts))}")
    return "\n".join(lines)


def canonical_id(text: Any) -> str:
    """Normalize any identifier / name into a lowercase hyphen slug.

    ``"TQ_6012"``, ``"tq 6012"`` and ``"TQ-6012"`` all become ``"tq-6012"``.
    """
    if text is None:
        return ""
    s = str(text).lower().strip()
    s = re.sub(r"[\s_/]+", "-", s)
    s = re.sub(r"[^a-z0-9.%\-]", "", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-")


def normalize_entity_type(raw: Any) -> str | None:
    """Return the canonical entity type, or None if outside the ontology."""
    key = re.sub(r"[\s_\-]", "", str(raw or "")).lower()
    return _ENTITY_TYPE_LOOKUP.get(key)


def normalize_relation_type(raw: Any) -> str | None:
    """Return the canonical relation type, or None if outside the ontology."""
    key = re.sub(r"[\s\-]+", "_", str(raw or "").strip().lower())
    return _RELATION_LOOKUP.get(key)


def check_relation(relation: str, source_type: str, target_type: str) -> str | None:
    """Validate an edge against the ontology.

    Returns ``None`` when valid, otherwise a short reason string. Relation
    types without an explicit rule are accepted if they belong to the ontology.
    """
    rel = (relation or "").lower()
    st = (source_type or "").lower()
    tt = (target_type or "").lower()
    if rel not in {r.lower() for r in NGA_RELATION_TYPES}:
        return f"unknown relation '{rel}'"
    if not st or not tt:
        return "missing endpoint type"
    rule = NGA_ONTOLOGY_RULES.get(rel)
    if rule is None:
        return None
    valid_src, valid_tgt = rule
    if st not in valid_src:
        return f"source type '{st}' not allowed for '{rel}'"
    if tt not in valid_tgt:
        return f"target type '{tt}' not allowed for '{rel}'"
    return None
