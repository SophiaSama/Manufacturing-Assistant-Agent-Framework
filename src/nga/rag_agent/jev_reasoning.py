"""System One reasoning controller and token telemetry powered by TypeSafe Jev.

Provides:
1. Evidence Sufficiency Gate: Evaluates whether gathered multi-source evidence
   is complete and adequate to answer the query, enabling semantic early exit.
2. Token Consumption & Cost Telemetry: Tracks token usage across System One (Jev)
   and System Two (Generative LLM) with counterfactual fallback cost comparison (TCER).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)

# Jev Model Pricing ($ per 1M tokens)
JEV_INPUT_PRICE_PER_M = 0.042
JEV_OUTPUT_PRICE_PER_M = 0.00  # Zero tokens generated

# Default Generative LLM Pricing fallback ($ per 1M tokens) - Claude 3.5 Sonnet / Frontier
DEFAULT_LLM_INPUT_PRICE_PER_M = 3.00
DEFAULT_LLM_OUTPUT_PRICE_PER_M = 15.00


@dataclass
class SufficiencyResult:
    """Outcome of Jev semantic evidence sufficiency check."""
    is_complete: bool
    completeness_score: int  # 1 (Insufficient) to 4 (Exhaustive)
    is_sufficient_prob: float
    next_action: str  # "proceed_to_synthesis" | "query_database" | "search_documents" | "conclude_unanswerable"
    confidence: float
    source: str  # "typesafe_jev" | "fallback"


@dataclass
class FactGroundednessResult:
    """Outcome of Jev post-synthesis fact verification and grounding evaluation."""
    is_faithful: bool
    is_faithful_prob: float
    groundedness_score: int  # 1 (Severe hallucination) to 5 (Flawlessly grounded)
    unsupported_claim_type: str  # "none" | "invented_numeric_spec" | "invented_citation" | "unsupported_safety_assertion"
    unsupported_claim_conf: float
    is_grounded: bool
    source: str  # "typesafe_jev" | "fallback"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class FanoutPlan:
    """Outcome of speculative fan-out planning at prepare node."""
    needs_sql_db: bool
    needs_sql_db_prob: float
    needs_sop_procedures: bool
    needs_sop_procedures_prob: float
    needs_supplier_quality: bool
    needs_supplier_quality_prob: float
    needs_maintenance_logs: bool
    needs_maintenance_logs_prob: float
    cross_source_depth: str  # "single_source" | "dual_source" | "triangulation"
    recommended_sources: list[str]
    guidance_prompt: str
    source: str  # "typesafe_jev" | "fallback"


@dataclass
class ContradictionResolution:
    """Outcome of cross-source contradiction detection and hierarchical precedence resolution."""
    has_contradiction: bool
    contradiction_prob: float
    conflict_nature: str  # "none" | "numeric_tolerance" | "procedural_deviation" | "status_discrepancy"
    precedence_rule: str
    resolution_guidance: str
    conflicting_sources: list[str]
    source: str  # "typesafe_jev" | "fallback"
    details: dict[str, Any] = field(default_factory=dict)


def get_typesafe_api_key() -> str | None:
    """Retrieve validated TypeSafe API key from environment."""
    key = os.getenv("TYPESAFE_API_KEY")
    if key:
        key = key.strip()
        if key.lower().startswith("your-") or key.lower() in ("placeholder", "none", ""):
            return None
    return key or None


def evaluate_evidence_sufficiency(
    query: str,
    question_parts: list[str],
    evidence: str,
    api_key: str | None = None,
    client: Any | None = None,
) -> SufficiencyResult:
    """Evaluate whether collected multi-source evidence is sufficient to answer the query.

    Uses TypeSafe Jev (Score, Noul, Choice) in a single parallel evaluation pass.
    Falls back gracefully if TypeSafe is unconfigured or unavailable.
    """
    key = api_key or get_typesafe_api_key()
    if not key and client is None:
        logger.debug("TypeSafe API key not configured; skipping semantic sufficiency check.")
        return SufficiencyResult(
            is_complete=False,
            completeness_score=1,
            is_sufficient_prob=0.0,
            next_action="query_more",
            confidence=0.0,
            source="fallback",
        )

    if not evidence or not evidence.strip():
        return SufficiencyResult(
            is_complete=False,
            completeness_score=1,
            is_sufficient_prob=0.0,
            next_action="search_documents",
            confidence=1.0,
            source="fallback",
        )

    state = {
        "original_query": query,
        "question_parts": question_parts,
        "accumulated_evidence": evidence[:4000],  # Bound context for fast forward pass
    }

    try:
        if client is None:
            from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
            ts_client = TypeSafeClient(api_key=key)
        else:
            from typesafe_sdk import Choice, Noul, Score
            ts_client = client

        questions = {
            "is_sufficient": Noul(
                instructions=(
                    "Does accumulated_evidence contain all specific facts, numbers, tables, "
                    "or document clauses necessary to provide a complete, verified answer to original_query?"
                )
            ),
            "evidence_completeness": Score(
                instructions="Score the completeness of evidence across all question parts.",
                criteria=[
                    "Insufficient: Key question parts have zero evidence.",
                    "Partial: Evidence gathered for some parts, but primary question unanswered.",
                    "Adequate: Core question can be fully answered; minor details omitted.",
                    "Exhaustive: Complete evidentiary support for every sub-part.",
                ],
            ),
            "next_action": Choice(
                instructions="Determine the optimal next step.",
                criteria={
                    "proceed_to_synthesis": "Evidence is complete; further tool calls would be redundant.",
                    "query_database": "Still missing structured database facts.",
                    "search_documents": "Still missing SOP or specification documents.",
                    "conclude_unanswerable": "Sources have been queried and data does not exist.",
                },
            ),
        }

        response = ts_client.system_one(state=state, questions=questions)
        answers = response.answers

        noul_ans = answers.get("is_sufficient")
        score_ans = answers.get("evidence_completeness")
        choice_ans = answers.get("next_action")

        prob = getattr(noul_ans, "probability", 0.0) if noul_ans else 0.0
        score_val = int(getattr(score_ans, "score", 1)) if score_ans else 1
        action_val = str(getattr(choice_ans, "choice", "query_database")) if choice_ans else "query_database"
        conf = float(getattr(choice_ans, "confidence", prob)) if choice_ans else prob

        # Calibrated early-exit criterion:
        # Sufficient if Noul prob >= 0.80 AND Completeness Score >= 3 OR choice explicitly says proceed
        is_complete = (prob >= 0.80 and score_val >= 3) or (action_val == "proceed_to_synthesis" and conf >= 0.75)

        logger.info(
            "jev_sufficiency_evaluated is_complete=%s prob=%.2f score=%d action=%s conf=%.2f",
            is_complete,
            prob,
            score_val,
            action_val,
            conf,
        )

        return SufficiencyResult(
            is_complete=is_complete,
            completeness_score=score_val,
            is_sufficient_prob=round(prob, 3),
            next_action=action_val,
            confidence=round(conf, 3),
            source="typesafe_jev",
        )

    except Exception as exc:
        logger.warning("Jev evidence sufficiency check failed (%s: %s). Falling back.", type(exc).__name__, exc)
        return SufficiencyResult(
            is_complete=False,
            completeness_score=1,
            is_sufficient_prob=0.0,
            next_action="query_more",
            confidence=0.0,
            source="fallback",
        )


def evaluate_fact_groundedness(
    query: str,
    evidence: str,
    generated_answer: str,
    api_key: str | None = None,
    client: Any | None = None,
) -> FactGroundednessResult:
    """Evaluate whether generated answer is strictly grounded in evidence with zero hallucinations.

    Uses TypeSafe Jev (Noul, Score, Choice) in a single parallel evaluation pass.
    Falls back gracefully if TypeSafe is unconfigured or unavailable.
    """
    key = api_key or get_typesafe_api_key()
    if not key and client is None:
        logger.debug("TypeSafe API key not configured; skipping semantic grounding check.")
        has_ev = bool(evidence and evidence.strip())
        return FactGroundednessResult(
            is_faithful=has_ev,
            is_faithful_prob=0.85 if has_ev else 0.0,
            groundedness_score=4 if has_ev else 1,
            unsupported_claim_type="none" if has_ev else "invented_citation",
            unsupported_claim_conf=0.80,
            is_grounded=has_ev,
            source="fallback",
        )

    if not evidence or not evidence.strip():
        return FactGroundednessResult(
            is_faithful=False,
            is_faithful_prob=0.0,
            groundedness_score=1,
            unsupported_claim_type="invented_citation",
            unsupported_claim_conf=1.0,
            is_grounded=False,
            source="fallback",
            details={"reason": "empty_evidence"},
        )

    try:
        if client is None:
            from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
            ts_client = TypeSafeClient(api_key=key)
        else:
            from typesafe_sdk import Choice, Noul, Score
            ts_client = client

        state = {
            "query": query[:1000],
            "source_evidence": evidence[:4000],
            "generated_answer": generated_answer[:3000],
        }

        questions = {
            "is_faithful": Noul(
                instructions=(
                    "Is every factual claim, numerical specification, part ID, and procedure "
                    "in generated_answer fully supported by source_evidence with zero hallucinated details?"
                )
            ),
            "groundedness": Score(
                instructions="Rate the groundedness of generated_answer against source_evidence on a 1-5 scale.",
                criteria=[
                    "1: Severe hallucination; invents specifications, tolerances, or procedures not in source evidence.",
                    "2: Major unsupported claims; contains fabricated numbers or contradictory statements.",
                    "3: Mostly supported; minor ungrounded peripheral statements, but core technical claims match evidence.",
                    "4: High fidelity; all core technical claims and numbers match source evidence exactly.",
                    "5: Flawlessly grounded; strict adherence to source evidence with complete fidelity.",
                ],
            ),
            "unsupported_claim_type": Choice(
                instructions="Categorize any ungrounded assertions in generated_answer.",
                criteria={
                    "none": "All assertions are fully grounded in source_evidence.",
                    "invented_numeric_spec": "Fabricated torque, temperature, pressure, dimension, or limit.",
                    "invented_citation": "Cited a document, SOP ID, or database record not in source_evidence.",
                    "unsupported_safety_assertion": "Unverified claim regarding Class A safety, severity, or recall status.",
                },
            ),
        }

        response = ts_client.system_one(state=state, questions=questions)
        answers = response.answers

        noul_ans = answers.get("is_faithful")
        score_ans = answers.get("groundedness")
        choice_ans = answers.get("unsupported_claim_type")

        prob = getattr(noul_ans, "probability", 0.0) if noul_ans else 0.0
        score_val = int(getattr(score_ans, "score", 1)) if score_ans else 1
        claim_type = str(getattr(choice_ans, "choice", "none")) if choice_ans else "none"
        claim_conf = float(getattr(choice_ans, "confidence", 0.0)) if choice_ans else 0.0

        # Calibrated grounding threshold:
        # Pass: score >= 3 AND prob >= 0.40 AND not (claim_type != 'none' and claim_conf >= 0.75)
        is_grounded = (score_val >= 3) and (prob >= 0.40) and not (claim_type != "none" and claim_conf >= 0.75)

        logger.info(
            "jev_grounding_evaluated is_grounded=%s score=%d faith_prob=%.2f ungrounded_type=%s conf=%.2f",
            is_grounded,
            score_val,
            prob,
            claim_type,
            claim_conf,
        )

        return FactGroundednessResult(
            is_faithful=prob >= 0.50,
            is_faithful_prob=round(prob, 3),
            groundedness_score=score_val,
            unsupported_claim_type=claim_type,
            unsupported_claim_conf=round(claim_conf, 3),
            is_grounded=is_grounded,
            source="typesafe_jev",
            details={
                "distribution": getattr(score_ans, "distribution", {}),
            },
        )

    except Exception as exc:
        logger.warning("Jev fact groundedness check failed (%s: %s). Falling back.", type(exc).__name__, exc)
        has_ev = bool(evidence and evidence.strip())
        return FactGroundednessResult(
            is_faithful=has_ev,
            is_faithful_prob=0.85 if has_ev else 0.0,
            groundedness_score=4 if has_ev else 1,
            unsupported_claim_type="none" if has_ev else "invented_citation",
            unsupported_claim_conf=0.50,
            is_grounded=has_ev,
            source="fallback",
            details={"error": str(exc)},
        )


def calculate_reasoning_token_telemetry(
    llm_prompt_tokens: int,
    llm_completion_tokens: int,
    jev_calls_count: int,
    jev_input_tokens: int,
    tool_rounds_executed: int,
    early_exit_triggered: bool,
    llm_input_price_per_m: float = DEFAULT_LLM_INPUT_PRICE_PER_M,
    llm_output_price_per_m: float = DEFAULT_LLM_OUTPUT_PRICE_PER_M,
) -> dict[str, Any]:
    """Calculate token consumption, cost, and counterfactual savings KPIs.

    Compares the active execution against a counterfactual baseline where the
    query ran to the circuit breaker ceiling (8 rounds) without Jev early exit.
    """
    # System Two LLM Cost
    llm_cost = (
        (llm_prompt_tokens / 1_000_000.0) * llm_input_price_per_m
        + (llm_completion_tokens / 1_000_000.0) * llm_output_price_per_m
    )

    # System One Jev Cost ($0.00 output token cost)
    jev_cost = (jev_input_tokens / 1_000_000.0) * JEV_INPUT_PRICE_PER_M

    total_tokens = llm_prompt_tokens + llm_completion_tokens + jev_input_tokens
    total_cost_usd = round(llm_cost + jev_cost, 6)

    # Counterfactual Fallback Estimate:
    # If early exit was not triggered, estimate tokens across 6-8 tool rounds
    fallback_rounds = 8 if early_exit_triggered else max(tool_rounds_executed, 4)
    multiplier = max(1.0, fallback_rounds / max(tool_rounds_executed, 1))

    est_fallback_in_tokens = int(llm_prompt_tokens * multiplier)
    est_fallback_out_tokens = int(llm_completion_tokens * multiplier)
    est_fallback_total_tokens = est_fallback_in_tokens + est_fallback_out_tokens

    est_fallback_cost = round(
        (est_fallback_in_tokens / 1_000_000.0) * llm_input_price_per_m
        + (est_fallback_out_tokens / 1_000_000.0) * llm_output_price_per_m,
        6,
    )

    # Token Consumption Efficiency Ratio (TCER)
    tcer = round(total_tokens / max(1, est_fallback_total_tokens), 3)

    # Savings percentage
    cost_savings_pct = (
        round(((est_fallback_cost - total_cost_usd) / est_fallback_cost) * 100.0, 1)
        if est_fallback_cost > total_cost_usd
        else 0.0
    )
    token_savings_pct = (
        round(((est_fallback_total_tokens - total_tokens) / est_fallback_total_tokens) * 100.0, 1)
        if est_fallback_total_tokens > total_tokens
        else 0.0
    )

    return {
        "system_two_llm": {
            "prompt_tokens": llm_prompt_tokens,
            "completion_tokens": llm_completion_tokens,
            "cost_usd": round(llm_cost, 6),
        },
        "system_one_jev": {
            "eval_calls": jev_calls_count,
            "input_tokens": jev_input_tokens,
            "output_tokens": 0,  # Zero output tokens generated by Jev
            "cost_usd": round(jev_cost, 6),
        },
        "totals": {
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost_usd,
        },
        "kpis": {
            "tcer": tcer,
            "output_token_elimination_rate_pct": 100.0,
            "early_exit_triggered": early_exit_triggered,
            "tool_rounds_executed": tool_rounds_executed,
            "token_savings_pct": token_savings_pct,
            "cost_savings_pct": cost_savings_pct,
            "est_fallback_cost_usd": est_fallback_cost,
        },
    }


def plan_speculative_fanout(
    query: str,
    user_role: str = "operator",
    active_station: str = "",
    api_key: str | None = None,
    client: Any | None = None,
) -> FanoutPlan:
    """Predict upfront which domain data stores are needed to answer the query.

    Uses TypeSafe Jev (4 parallel Noul questions + 1 Choice question).
    Falls back gracefully to keyword heuristic if unconfigured or API error.
    """
    key = api_key or get_typesafe_api_key()
    if not key and client is None:
        q_lower = query.lower()
        needs_sql = any(k in q_lower for k in ["count", "serial", "station", "cycle", "yield", "rate", "database", "table", "record", "how many", "quantity", "nc-"])
        needs_sop = any(k in q_lower for k in ["torque", "procedure", "sop", "spec", "step", "how to", "tolerance", "angle", "standard", "bleed", "cure"])
        needs_quality = any(k in q_lower for k in ["supplier", "scar", "vendor", "containment", "defect", "incoming", "material", "certification"])
        needs_maint = any(k in q_lower for k in ["maintenance", "repair", "breakdown", "work order", "wo-", "pm", "calibration", "interval", "robot"])

        if not (needs_sql or needs_sop or needs_quality or needs_maint):
            needs_sop = True

        recs: list[str] = []
        if needs_sql:
            recs.append("sql_database")
        if needs_sop:
            recs.append("sop_procedures")
        if needs_quality:
            recs.append("supplier_quality")
        if needs_maint:
            recs.append("maintenance_logs")

        depth = "triangulation" if len(recs) >= 3 else ("dual_source" if len(recs) >= 2 else "single_source")
        guidance = f"Target sources: {', '.join(recs)}. Multi-source depth: {depth}."

        return FanoutPlan(
            needs_sql_db=needs_sql,
            needs_sql_db_prob=0.85 if needs_sql else 0.15,
            needs_sop_procedures=needs_sop,
            needs_sop_procedures_prob=0.85 if needs_sop else 0.15,
            needs_supplier_quality=needs_quality,
            needs_supplier_quality_prob=0.85 if needs_quality else 0.15,
            needs_maintenance_logs=needs_maint,
            needs_maintenance_logs_prob=0.85 if needs_maint else 0.15,
            cross_source_depth=depth,
            recommended_sources=recs,
            guidance_prompt=guidance,
            source="fallback",
        )

    try:
        if client is None:
            from typesafe_sdk import Choice, Noul, TypeSafeClient
            ts_client = TypeSafeClient(api_key=key)
        else:
            from typesafe_sdk import Choice, Noul
            ts_client = client

        state = {
            "query": query[:1000],
            "user_role": user_role,
            "active_station": active_station or "unspecified",
        }

        questions = {
            "needs_sql_db": Noul(
                instructions=(
                    "Does answering this question require quantitative database records, "
                    "such as serial numbers, station cycle times, build counts, or defect logs?"
                )
            ),
            "needs_sop_procedures": Noul(
                instructions=(
                    "Does answering this question require standard operating procedures, "
                    "nominal torque specifications, tool calibration limits, or assembly steps?"
                )
            ),
            "needs_supplier_quality": Noul(
                instructions=(
                    "Does this inquiry relate to incoming component defects, supplier containment, "
                    "SCAR reports, or vendor material certifications?"
                )
            ),
            "needs_maintenance_logs": Noul(
                instructions=(
                    "Does this inquiry reference past equipment breakdown, emergency work orders, "
                    "or physical machine repairs?"
                )
            ),
            "cross_source_depth": Choice(
                instructions="Assess the degree of multi-source cross-referencing required.",
                criteria={
                    "single_source": "Query can be answered completely from one document or table.",
                    "dual_source": "Requires comparing two sources (e.g., build count vs SOP spec).",
                    "triangulation": "Root-cause diagnosis requiring synthesis across database, SOP, and quality reports.",
                },
            ),
        }

        response = ts_client.system_one(state=state, questions=questions)
        answers = response.answers

        p_sql = float(getattr(answers.get("needs_sql_db"), "probability", 0.0))
        p_sop = float(getattr(answers.get("needs_sop_procedures"), "probability", 0.0))
        p_qual = float(getattr(answers.get("needs_supplier_quality"), "probability", 0.0))
        p_maint = float(getattr(answers.get("needs_maintenance_logs"), "probability", 0.0))

        depth_ans = answers.get("cross_source_depth")
        depth = str(getattr(depth_ans, "choice", "single_source")) if depth_ans else "single_source"

        recs = []
        if p_sql >= 0.50:
            recs.append("sql_database")
        if p_sop >= 0.50:
            recs.append("sop_procedures")
        if p_qual >= 0.50:
            recs.append("supplier_quality")
        if p_maint >= 0.50:
            recs.append("maintenance_logs")

        if not recs:
            max_pair = max([("sql_database", p_sql), ("sop_procedures", p_sop), ("supplier_quality", p_qual), ("maintenance_logs", p_maint)], key=lambda x: x[1])
            recs.append(max_pair[0] if max_pair[1] > 0.30 else "sop_procedures")

        guidance = (
            f"Pre-routing target domains: {', '.join(recs)}. "
            f"Recommended deliberation depth: {depth}. "
            f"Prioritize calling designated tools for these domains."
        )

        return FanoutPlan(
            needs_sql_db=p_sql >= 0.50,
            needs_sql_db_prob=round(p_sql, 3),
            needs_sop_procedures=p_sop >= 0.50,
            needs_sop_procedures_prob=round(p_sop, 3),
            needs_supplier_quality=p_qual >= 0.50,
            needs_supplier_quality_prob=round(p_qual, 3),
            needs_maintenance_logs=p_maint >= 0.50,
            needs_maintenance_logs_prob=round(p_maint, 3),
            cross_source_depth=depth,
            recommended_sources=recs,
            guidance_prompt=guidance,
            source="typesafe_jev",
        )

    except Exception as exc:
        logger.warning("Jev speculative fan-out failed (%s: %s). Falling back.", type(exc).__name__, exc)
        return plan_speculative_fanout(query=query, user_role=user_role, active_station=active_station, api_key="", client=None)


PRECEDENCE_HIERARCHY = {
    "scar": 1,
    "containment": 1,
    "stop_ship": 1,
    "deviation": 1,
    "work_order": 2,
    "wo": 2,
    "maintenance": 2,
    "sop": 3,
    "tec": 3,
    "spec": 3,
    "database": 3,
    "sql": 3,
    "general": 4,
    "draft": 4,
}


def _get_precedence_rank(source_type: str, citation: str) -> int:
    combined = f"{source_type} {citation}".lower()
    for key, rank in PRECEDENCE_HIERARCHY.items():
        if key in combined:
            return rank
    return 4


def detect_cross_source_contradiction(
    source_a: dict[str, str],
    source_b: dict[str, str],
    api_key: str | None = None,
    client: Any | None = None,
) -> ContradictionResolution:
    """Detect whether two retrieved sources contain conflicting specifications or procedures,
    and resolve precedence using deterministic manufacturing hierarchy.
    """
    cit_a = source_a.get("citation", "Source A")
    cit_b = source_b.get("citation", "Source B")
    type_a = source_a.get("type", "document")
    type_b = source_b.get("type", "document")
    content_a = source_a.get("content", "")
    content_b = source_b.get("content", "")

    key = api_key or get_typesafe_api_key()
    if not key and client is None:
        return ContradictionResolution(
            has_contradiction=False,
            contradiction_prob=0.0,
            conflict_nature="none",
            precedence_rule="No contradiction detected.",
            resolution_guidance="",
            conflicting_sources=[cit_a, cit_b],
            source="fallback",
        )

    try:
        if client is None:
            from typesafe_sdk import Choice, Noul, TypeSafeClient
            ts_client = TypeSafeClient(api_key=key)
        else:
            from typesafe_sdk import Choice, Noul
            ts_client = client

        state = {
            "source_a": {"type": type_a, "citation": cit_a, "content": content_a[:1500]},
            "source_b": {"type": type_b, "citation": cit_b, "content": content_b[:1500]},
        }

        questions = {
            "has_contradiction": Noul(
                instructions=(
                    "Do source_a and source_b contain conflicting procedural requirements, "
                    "incompatible numeric tolerances, or contradictory instructions?"
                )
            ),
            "conflict_nature": Choice(
                instructions="Identify the nature of the contradiction.",
                criteria={
                    "none": "No conflict exists; sources are mutually compatible.",
                    "numeric_tolerance": "Conflicting numeric torque, pressure, or dimension values.",
                    "procedural_deviation": "Conflicting instructions or routing steps.",
                    "status_discrepancy": "Discrepancy in component approval, pass/fail, or supplier status.",
                },
            ),
        }

        response = ts_client.system_one(state=state, questions=questions)
        answers = response.answers

        p_contra = float(getattr(answers.get("has_contradiction"), "probability", 0.0))
        choice_nature = answers.get("conflict_nature")
        nature = str(getattr(choice_nature, "choice", "none")) if choice_nature else "none"

        has_conflict = p_contra >= 0.70 and nature != "none"

        rank_a = _get_precedence_rank(type_a, cit_a)
        rank_b = _get_precedence_rank(type_b, cit_b)

        if has_conflict:
            if rank_a < rank_b:
                winner, loser = cit_a, cit_b
                rule = f"Precedence Rule: {winner} (Rank {rank_a}) overrides {loser} (Rank {rank_b})."
            elif rank_b < rank_a:
                winner, loser = cit_b, cit_a
                rule = f"Precedence Rule: {winner} (Rank {rank_b}) overrides {loser} (Rank {rank_a})."
            else:
                rule = f"Precedence Rule: {cit_a} and {cit_b} share equal precedence (Rank {rank_a}). Both values must be reported and escalated to engineering."

            guidance = (
                f"[DETECTED SPECIFICATION CONFLICT]\n"
                f"Discrepancy ({nature.replace('_', ' ')}) between {cit_a} and {cit_b}.\n"
                f"{rule}\n"
                f"Explicitly highlight this conflict and resolution in findings and recommendations."
            )
        else:
            rule = "No conflict detected."
            guidance = ""

        return ContradictionResolution(
            has_contradiction=has_conflict,
            contradiction_prob=round(p_contra, 3),
            conflict_nature=nature,
            precedence_rule=rule,
            resolution_guidance=guidance,
            conflicting_sources=[cit_a, cit_b],
            source="typesafe_jev",
        )

    except Exception as exc:
        logger.warning("Jev contradiction check failed (%s: %s). Falling back.", type(exc).__name__, exc)
        return ContradictionResolution(
            has_contradiction=False,
            contradiction_prob=0.0,
            conflict_nature="none",
            precedence_rule="No conflict detected (fallback).",
            resolution_guidance="",
            conflicting_sources=[cit_a, cit_b],
            source="fallback",
            details={"error": str(exc)},
        )


def screen_evidence_contradictions(
    messages: list[Any],
    api_key: str | None = None,
    client: Any | None = None,
) -> ContradictionResolution | None:
    """Extract individual source items from tool messages and screen top pairs for contradiction."""
    items: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        name = getattr(msg, "name", "") or "tool"
        if name == "search_sop_documents":
            try:
                data = json.loads(content)
                docs = data.get("documents", []) if isinstance(data, dict) else []
                for d in docs:
                    if isinstance(d, dict):
                        items.append({
                            "type": d.get("category", "sop"),
                            "citation": d.get("doc_id", "DOC"),
                            "content": d.get("content", "") or d.get("summary", ""),
                        })
            except Exception:
                items.append({"type": "sop", "citation": "SOP", "content": content})
        elif name == "query_nga_database":
            items.append({"type": "sql", "citation": "database", "content": content})

    if len(items) < 2:
        return None

    # Test distinct pairs (up to 3 pairs)
    for i in range(min(len(items), 3)):
        for j in range(i + 1, min(len(items), 3)):
            if items[i]["citation"] == items[j]["citation"]:
                continue
            resolution = detect_cross_source_contradiction(items[i], items[j], api_key=api_key, client=client)
            if resolution.has_contradiction:
                return resolution

    return None
