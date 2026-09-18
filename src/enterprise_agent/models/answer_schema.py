"""Structured final-answer schema and governance output for Enterprise Agent."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class EvidenceReference(BaseModel):
    """Reference to evidence grounding an answer."""
    source_type: Literal["sql", "document", "graph"]
    citation: str  # Document ID, SQL Table, NC-id, or Graph entity
    supports: list[str] = Field(default_factory=list)


class GovernanceAlert(BaseModel):
    """Generic risk, compliance, and governance container."""
    is_critical: bool = False
    critical_triggers: list[str] = Field(default_factory=list)
    escalation_tier: str | None = None  # e.g., "L1", "L2", "L3", "L4"
    compliance_codes: list[str] = Field(default_factory=list)  # e.g. ["QCR-501", "OFAC-SANCTION"]
    hitl_required: bool = False
    action_verb: str | None = None


class FinalAnswer(BaseModel):
    """Structured response produced by the synthesis node."""
    direct_answer: str
    findings: list[str] = Field(default_factory=list)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    answered_questions: list[str] = Field(default_factory=list)
    unanswered_questions: list[str] = Field(default_factory=list)
    recommendation: str | None = None
    governance: GovernanceAlert = Field(default_factory=GovernanceAlert)

    # Backward compatibility fields (synced with governance)
    class_a_alert: bool = False
    escalation_level: str | None = None
    recall_criteria_met: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def sync_governance_compat(self) -> "FinalAnswer":
        # Synchronize backward compat fields with generic governance alert
        if self.class_a_alert and not self.governance.is_critical:
            self.governance.is_critical = True
        elif self.governance.is_critical:
            self.class_a_alert = True

        if self.escalation_level and not self.governance.escalation_tier:
            self.governance.escalation_tier = self.escalation_level
        elif self.governance.escalation_tier:
            self.escalation_level = self.governance.escalation_tier

        if self.recall_criteria_met and not self.governance.compliance_codes:
            self.governance.compliance_codes = self.recall_criteria_met
        elif self.governance.compliance_codes:
            self.recall_criteria_met = self.governance.compliance_codes

        return self


def _normalize_source_type(value: object) -> str:
    if not isinstance(value, str):
        return "document"
    normalized = value.strip().lower()
    alias_map = {
        "sql": "sql", "database": "sql", "query_database": "sql",
        "nga_database": "sql", "query_nga_database": "sql",
        "document": "document", "sop": "document", "spec": "document",
        "search_documents": "document", "search_sop_documents": "document",
        "reference": "document", "graph": "graph", "knowledge_graph": "graph",
    }
    return alias_map.get(normalized, "document")


def _normalize_evidence_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        normalized["source_type"] = _normalize_source_type(normalized.get("source_type"))
        result.append(normalized)
    return result


def _coerce_str(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _coerce_str_list(value: object) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def _normalize_payload_dict(payload: dict[str, object]) -> dict[str, object]:
    wrapped = payload.get("final_answer")
    if isinstance(wrapped, dict):
        payload = wrapped

    normalized: dict[str, object] = {}
    normalized["direct_answer"] = _coerce_str(payload.get("direct_answer"))
    normalized["findings"] = _coerce_str_list(payload.get("findings"))
    normalized["evidence"] = _normalize_evidence_list(payload.get("evidence"))
    normalized["answered_questions"] = _coerce_str_list(payload.get("answered_questions"))
    normalized["unanswered_questions"] = _coerce_str_list(payload.get("unanswered_questions"))

    rec = payload.get("recommendation")
    normalized["recommendation"] = _coerce_str(rec) if rec else None

    # Handle governance
    gov_raw = payload.get("governance")
    if isinstance(gov_raw, dict):
        normalized["governance"] = gov_raw
    else:
        normalized["governance"] = {}

    # Backward compatibility fields
    class_a = payload.get("class_a_alert")
    if isinstance(class_a, bool):
        normalized["class_a_alert"] = class_a
    elif isinstance(class_a, str):
        normalized["class_a_alert"] = class_a.strip().lower() in ("true", "1", "yes")

    esc = payload.get("escalation_level")
    normalized["escalation_level"] = _coerce_str(esc) if esc else None

    normalized["recall_criteria_met"] = _coerce_str_list(payload.get("recall_criteria_met"))

    return normalized


def parse_final_answer(raw: str) -> FinalAnswer:
    """Parse JSON string into a structured FinalAnswer."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        first_line = lines[0].lower()
        if first_line.startswith("```json") or first_line == "```":
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return FinalAnswer(
            direct_answer=raw,
            findings=[],
            evidence=[],
            answered_questions=[],
            unanswered_questions=[],
        )

    if not isinstance(data, dict):
        return FinalAnswer(
            direct_answer=str(data),
            findings=[],
            evidence=[],
        )

    normalized = _normalize_payload_dict(data)
    return FinalAnswer.model_validate(normalized)


def render_final_answer(answer: FinalAnswer) -> str:
    """Format FinalAnswer as readable Markdown."""
    lines: list[str] = []

    # Critical Banner
    if answer.governance.is_critical or answer.class_a_alert:
        lines.append("🚨 **CRITICAL RISK / POLICY ALERT**")
        if answer.governance.critical_triggers:
            triggers = ", ".join(answer.governance.critical_triggers)
            lines.append(f"> Triggered: {triggers}")
        lines.append("")

    # Direct Answer
    lines.append(answer.direct_answer)
    lines.append("")

    # Findings
    if answer.findings:
        lines.append("### Key Findings")
        for finding in answer.findings:
            lines.append(f"- {finding}")
        lines.append("")

    # Evidence
    if answer.evidence:
        lines.append("### Evidence & Citations")
        for ev in answer.evidence:
            supports = f" (supports: {', '.join(ev.supports)})" if ev.supports else ""
            lines.append(f"- `[{ev.source_type.upper()}]` **{ev.citation}**{supports}")
        lines.append("")

    # Recommendation & Escalation
    if answer.recommendation:
        lines.append("### Recommendation")
        lines.append(answer.recommendation)
        esc = answer.governance.escalation_tier or answer.escalation_level
        if esc:
            lines.append(f"*Escalation Level:* `{esc}`")
        if answer.governance.compliance_codes:
            lines.append(f"*Compliance Standards:* {', '.join(answer.governance.compliance_codes)}")
        lines.append("")

    return "\n".join(lines).strip()
