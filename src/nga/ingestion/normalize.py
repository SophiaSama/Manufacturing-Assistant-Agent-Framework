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
    "causes": (
        {"faultcode", "defectcode", "machine", "threshold", "part"},
        {"defectcode", "ncrecord", "machine", "faultcode", "part", "escalationlevel"},
    ),
    "indicates": (
        {"machine", "threshold", "sensor", "station", "faultcode", "defectcode", "part"},
        {"faultcode", "defectcode", "ncrecord", "threshold", "torque", "part"},
    ),
    "prevents": (
        {"sop", "procedure", "part", "machine"},
        {"faultcode", "defectcode", "ncrecord"},
    ),
    "monitors": (
        {"machine", "station", "personnel"},
        {"machine", "threshold", "torque", "station", "part", "faultcode", "defectcode"},
    ),
    "threshold_for": (
        {"sop", "procedure", "machine", "station", "part", "torque", "escalationlevel", "defectcode", "faultcode", "personnel"},
        {"threshold", "torque", "part", "machine", "sop", "procedure"},
    ),
    "documented_in": (
        {
            "threshold", "torque", "machine", "faultcode", "sop", "station",
            "procedure", "defectcode", "ncrecord", "workorder", "part",
            "recallcriteria", "escalationlevel",
        },
        {"sop", "station", "procedure", "workorder", "ncrecord", "recallcriteria", "escalationlevel"},
    ),
    "escalates_to": (
        {"defectcode", "ncrecord", "faultcode", "escalationlevel", "personnel", "sop", "procedure", "threshold"},
        {"escalationlevel", "personnel", "role", "procedure", "sop", "defectcode"},
    ),
    "requires": (
        {"sop", "procedure", "defectcode", "ncrecord", "workorder", "escalationlevel", "recallcriteria", "personnel"},
        {"machine", "procedure", "sop", "personnel", "torque", "part", "station", "threshold", "escalationlevel", "ncrecord"},
    ),
    "audited_by": (
        {"machine", "station", "torque", "workorder", "sop", "procedure", "part", "ncrecord"},
        {"machine", "personnel", "sop", "procedure"},
    ),
    "owned_by": (
        {"sop", "procedure", "escalationlevel", "recallcriteria", "workorder", "machine", "station", "ncrecord"},
        {"personnel", "escalationlevel", "procedure", "sop"},
    ),
    "certified_for": (
        {"personnel", "sop", "procedure"},
        {"sop", "procedure", "machine", "station", "torque", "personnel"},
    ),
    "triggers_recall": (
        {"defectcode", "ncrecord", "faultcode", "threshold", "part"},
        {"recallcriteria", "procedure", "escalationlevel"},
    ),
}

_ENTITY_TYPE_LOOKUP = {t.lower(): t for t in NGA_ENTITY_TYPES}
_ENTITY_TYPE_LOOKUP.update({
    "fault": "FaultCode",
    "errorcode": "FaultCode",
    "defect": "DefectCode",
    "sensor": "Part",
    "component": "Part",
    "platform": "Part",
    "vehicle": "Part",
    "role": "Personnel",
    "operator": "Personnel",
    "technician": "Personnel",
    "escalation": "EscalationLevel",
    "priority": "EscalationLevel",
    "recall": "RecallCriteria",
    "nc": "NCRecord",
    "nonconformance": "NCRecord",
    "document": "Procedure",
    "spec": "Procedure",
    "specification": "Procedure",
    "standard": "Procedure",
    "tool": "Machine",
    "equipment": "Machine",
    "robot": "Machine",
})
_RELATION_LOOKUP = {r.lower(): r for r in NGA_RELATION_TYPES}

# Canonical entity registry: maps canonical_id -> (canonical_type, canonical_name, list_of_aliases)
CANONICAL_ENTITY_REGISTRY: dict[str, tuple[str, str, list[str]]] = {
    "sop-opr-101": (
        "SOP",
        "SOP-OPR-101 Wheel Lug Nut Tightening",
        ["SOP-OPR-101", "SOP 101", "Lug nut torque SOP", "Wheel tightening procedure", "SOP-OPR-101 (Wheel torque)"],
    ),
    "station-144": (
        "Station",
        "Station 144 Chassis Assembly",
        ["Station 144", "144", "Wheel install, General Assembly"],
    ),
    "tq-6012": (
        "Machine",
        "TorqMaster TQ-6012 DC Electric Nutrunner",
        ["TQ-6012", "TorqMaster 6012", "TQ6012", "Station 144 nutrunner", "TorqMaster TQ-6012", "machine-tq-6012"],
    ),
    "torque-target-105nm": (
        "Threshold",
        "105 Nm ± 5% Target Torque",
        ["105 Nm", "105 Nm ± 5%", "threshold-torque-tolerance", "torque-wheel-torque", "wheel-torque-target"],
    ),
    "sop-opr-114": (
        "SOP",
        "SOP-OPR-114 Windshield Installation",
        ["SOP-OPR-114", "SOP 114", "Windshield installation SOP"],
    ),
    "station-131": (
        "Station",
        "Station 131 Fluid Fill",
        ["Station 131", "131", "Brake fluid fill station"],
    ),
    "sop-opr-142": (
        "SOP",
        "SOP-OPR-142 Brake Fluid Fill",
        ["SOP-OPR-142", "SOP 142", "Brake fluid fill procedure"],
    ),
    "fluid-moisture-threshold": (
        "Threshold",
        "1.5% Maximum Moisture Content",
        ["1.5% moisture", "1.5% Maximum Moisture Content", "plus-minus-1.5-percent", "moisture-threshold-1.5pct"],
    ),
    "sop-opr-158": (
        "SOP",
        "SOP-OPR-158 Torque Audit Protocol",
        ["SOP-OPR-158", "SOP 158", "SOP-OPR-158 (Final inspection)", "Torque audit protocol"],
    ),
    "tq-6018": (
        "Machine",
        "TorqMaster TF-6000 (TQ-6018) Audit Wrench",
        [
            "TQ-6018",
            "TorqMaster TF-6000",
            "TF-6000",
            "TQ 6018",
            "audit wrench TQ-6018",
            "torque audit tool",
            "torque audit wrench",
            "tq6018",
        ],
    ),
    "sop-tec-201": (
        "SOP",
        "SOP-TEC-201 Weld Robot Maintenance",
        ["SOP-TEC-201", "TEC-201", "SOP 201"],
    ),
    "sop-tec-214": (
        "SOP",
        "SOP-TEC-214 Torque Tool Calibration",
        ["SOP-TEC-214", "TEC-214", "Torque tool calibration procedure", "SOP 214", "torque-tool-cal"],
    ),
    "drift-threshold-2pct": (
        "Threshold",
        "2.0% Maximum Drift Threshold",
        ["2% threshold", "threshold-2pct", "2.0% drift", "2% drift threshold"],
    ),
    "tec-301": (
        "Procedure",
        "TEC-301 Weld Robot Specifications",
        ["TEC-301", "procedure-tec-301"],
    ),
    "weld-cap-life-4000": (
        "Threshold",
        "4,000 Weld Cycles Cap Life Limit",
        ["cap-life-threshold", "4000 welds", "4,000 weld cycles", "cap-life-4000"],
    ),
    "tec-314": (
        "Procedure",
        "TEC-314 Fastener Tooling Standards",
        ["TEC-314", "procedure-tec-314"],
    ),
    "tec-328": (
        "Procedure",
        "TEC-328 Pallet and Conveyor Alignment",
        ["TEC-328", "TEC-328 §6", "procedure-tec-328"],
    ),
    "qcr-501": (
        "RecallCriteria",
        "QCR-501 Quality Recall Criteria",
        ["QCR-501", "QCR501", "Quality Recall Criteria", "Recall standard 501"],
    ),
    "class-a-defect": (
        "DefectCode",
        "Class A Safety Critical Defect",
        ["Class A", "Class A defect", "safety-critical defect", "Class-A", "class-a", "class-a-safety", "class-a-safety-events"],
    ),
    "esc-402": (
        "EscalationLevel",
        "ESC-402 Plant Escalation Protocol",
        ["ESC-402", "esc-402-level-4"],
    ),
    "fap-401": (
        "Procedure",
        "FAP-401 Failure Analysis & 8D Process",
        ["FAP-401", "procedure-fap-401", "fap-401-8d", "parts-batch-analysis-fap-401"],
    ),
    "aurora-au-2025": (
        "Part",
        "Aurora AU-2025 Vehicle Platform",
        ["Aurora", "AU-2025", "Aurora AU-2025", "AU25", "Aurora 2025", "aurora", "au25"],
    ),
    "solstice-so-2025": (
        "Part",
        "Solstice SO-2025 Vehicle Platform",
        ["Solstice", "SO-2025", "Solstice SO-2025", "SO25", "Solstice 2025", "solstice", "so25"],
    ),
    "rb-07": (
        "Machine",
        "WELD-ROBOT-07 (RB-07)",
        ["RB-07", "weld-robot-07", "machine-rb-07", "WELD-ROBOT-07 (RB-07)"],
    ),
    "e-4012": (
        "FaultCode",
        "E-4012",
        ["E-4012", "fault-code-e-4012", "fault-e-4012"],
    ),
}

# Build reverse lookup from any normalized alias/id to canonical_id
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _cid, (_ctype, _cname, _caliases) in CANONICAL_ENTITY_REGISTRY.items():
    _ALIAS_TO_CANONICAL[_cid] = _cid
    for _lbl in [_cname, *_caliases]:
        _norm_lbl = re.sub(r"[\s_/]+", "-", _lbl.lower().strip())
        _norm_lbl = re.sub(r"[^a-z0-9.%\-]", "", _norm_lbl)
        _norm_lbl = re.sub(r"-{2,}", "-", _norm_lbl).strip("-")
        if _norm_lbl:
            _ALIAS_TO_CANONICAL[_norm_lbl] = _cid




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


_REDUNDANT_TYPE_PREFIXES = (
    "machine-", "procedure-", "personnel-", "nc-record-", "part-",
    "workorder-", "fault-code-", "defect-code-", "supplier-",
)


def resolve_canonical_id(raw_id: Any, name: Any = None, aliases: list[str] | None = None) -> str:
    """Resolve an entity's raw ID, name, or aliases to its canonical slug."""
    candidates = [raw_id, name, *(aliases or [])]
    for cand in candidates:
        slug = canonical_id(cand)
        if not slug:
            continue
        if slug in _ALIAS_TO_CANONICAL:
            return _ALIAS_TO_CANONICAL[slug]
        for pref in _REDUNDANT_TYPE_PREFIXES:
            if slug.startswith(pref):
                stripped = slug[len(pref):]
                if stripped in _ALIAS_TO_CANONICAL:
                    return _ALIAS_TO_CANONICAL[stripped]
    slug = canonical_id(raw_id) or canonical_id(name)
    for pref in _REDUNDANT_TYPE_PREFIXES:
        if slug.startswith(pref) and len(slug) > len(pref):
            slug = slug[len(pref):]
            break
    return _ALIAS_TO_CANONICAL.get(slug, slug)


def enrich_canonical_entity(
    eid: str,
    etype: str | None,
    name: str | None,
    aliases: set[str] | list[str] | None = None,
    include_registry_aliases: bool = False,
) -> tuple[str, str | None, str, set[str]]:
    """Return (canonical_eid, canonical_etype, display_name, merged_aliases)."""
    alias_set = {str(a).strip() for a in (aliases or []) if isinstance(a, str) and str(a).strip()}
    cid = resolve_canonical_id(eid, name, list(alias_set))
    if name and str(name).strip():
        name_owner = _ALIAS_TO_CANONICAL.get(canonical_id(name))
        if canonical_id(name) != cid and (name_owner is None or name_owner == cid):
            alias_set.add(str(name).strip())
    if eid and canonical_id(eid) != cid and canonical_id(eid) not in _ALIAS_TO_CANONICAL:
        alias_set.add(str(eid).strip())

    # Remove any alias that belongs to a different canonical entity in the registry
    alias_set = {
        a for a in alias_set
        if _ALIAS_TO_CANONICAL.get(canonical_id(a)) in (None, cid)
    }

    if cid in CANONICAL_ENTITY_REGISTRY:
        reg_type, reg_name, reg_aliases = CANONICAL_ENTITY_REGISTRY[cid]
        final_type = reg_type or etype
        final_name = reg_name or name or cid
        if include_registry_aliases:
            alias_set.update(reg_aliases)
        return cid, final_type, final_name, alias_set

    return cid, etype, (name or cid), alias_set


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
