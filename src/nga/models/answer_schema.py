"""Structured final-answer schema and rendering for the NGA assistant."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, ValidationError


class EvidenceReference(BaseModel):
    source_type: Literal["sql", "document"]
    citation: str          # doc_id / SOP number / SQL table or NC-id
    supports: list[str] = Field(default_factory=list)


class FinalAnswer(BaseModel):
    direct_answer: str
    findings: list[str] = Field(default_factory=list)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    answered_questions: list[str] = Field(default_factory=list)
    unanswered_questions: list[str] = Field(default_factory=list)
    recommendation: str | None = None
    # NGA-specific: safety flags
    class_a_alert: bool = False
    escalation_level: str | None = None   # "L1","L2","L3","L4"
    recall_criteria_met: list[str] = Field(default_factory=list)  # ["C1","C3"]


def _normalize_source_type(value: object) -> str:
    if not isinstance(value, str):
        return "document"
    normalized = value.strip().lower()
    alias_map = {
        "sql": "sql", "nga_database": "sql", "nga database": "sql",
        "query_nga_database": "sql", "database": "sql",
        "document": "document", "sop": "document", "spec": "document",
        "search_sop_documents": "document", "reference": "document",
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

    if "direct_answer" in payload:
        normalized = dict(payload)
        if isinstance(normalized.get("evidence"), list):
            normalized["evidence"] = _normalize_evidence_list(normalized.get("evidence"))
        return normalized

    direct_answer = (
        payload.get("answer") or payload.get("response") or payload.get("output", "")
    )
    if isinstance(direct_answer, dict):
        direct_answer = direct_answer.get("direct_answer", "")

    normalized = {
        "direct_answer": direct_answer.strip() if isinstance(direct_answer, str) else "",
        "findings": _coerce_str_list(payload.get("findings") or payload.get("highlights")),
        "answered_questions": _coerce_str_list(payload.get("answered_questions")),
        "unanswered_questions": _coerce_str_list(payload.get("unanswered_questions")),
        "recommendation": payload.get("recommendation") or payload.get("action"),
        "class_a_alert": bool(payload.get("class_a_alert", False)),
        "escalation_level": payload.get("escalation_level"),
        "recall_criteria_met": _coerce_str_list(payload.get("recall_criteria_met")),
    }
    evidence = payload.get("evidence")
    if isinstance(evidence, list):
        normalized["evidence"] = _normalize_evidence_list(evidence)
    return normalized


def _try_parse_json(raw: str) -> FinalAnswer | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return FinalAnswer.model_validate(_normalize_payload_dict(payload))
    except ValidationError:
        return None


def parse_final_answer(raw_answer: str, question_parts: list[str]) -> FinalAnswer:
    """Parse model output into a validated structured FinalAnswer."""
    normalized_parts = list(
        dict.fromkeys(p.strip() for p in question_parts if p.strip())
    )
    try:
        answer = FinalAnswer.model_validate_json(raw_answer)
    except ValidationError:
        parsed = _try_parse_json(raw_answer)
        answer = parsed or FinalAnswer(
            direct_answer=raw_answer.strip(),
            unanswered_questions=normalized_parts,
        )

    answered = [p for p in answer.answered_questions if p in normalized_parts]
    if not answered and normalized_parts and len(normalized_parts) <= 2:
        answered = list(normalized_parts)

    unanswered = [
        p for p in answer.unanswered_questions
        if p in normalized_parts and p not in answered
    ]
    for p in normalized_parts:
        if p not in answered and p not in unanswered:
            unanswered.append(p)

    return answer.model_copy(
        update={"answered_questions": answered, "unanswered_questions": unanswered}
    )


def render_final_answer(answer: FinalAnswer) -> str:
    """Render a FinalAnswer into user-facing prose with NGA safety flags."""
    lines: list[str] = []

    # Class A alert banner
    if answer.class_a_alert:
        lines.append("⚠️  CLASS A SAFETY-CRITICAL DEFECT — Immediate Level 4 escalation required.")
        lines.append("")

    lines.append(answer.direct_answer.strip())

    if answer.findings:
        lines.append("")
        lines.append("Findings:")
        lines.extend(f"  • {f}" for f in answer.findings)

    if answer.escalation_level:
        lines.append("")
        lines.append(f"Escalation Required: {answer.escalation_level}")

    if answer.recall_criteria_met:
        lines.append("")
        lines.append(
            f"QCR-501 Criteria Met: {', '.join(answer.recall_criteria_met)} — "
            "Field action evaluation required."
        )

    if answer.unanswered_questions:
        lines.append("")
        lines.append("Unanswered questions:")
        lines.extend(f"  - {q}" for q in answer.unanswered_questions)

    if answer.recommendation:
        lines.append("")
        lines.append(f"Recommendation: {answer.recommendation}")

    if answer.evidence:
        citations = " ".join(f"[{e.citation}]" for e in answer.evidence)
        lines.append("")
        lines.append(f"Sources: {citations}")

    return "\n".join(lines).strip()
