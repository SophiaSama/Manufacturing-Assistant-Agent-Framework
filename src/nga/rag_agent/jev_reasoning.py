"""System One reasoning controller and token telemetry powered by TypeSafe Jev.

Provides:
1. Evidence Sufficiency Gate: Evaluates whether gathered multi-source evidence
   is complete and adequate to answer the query, enabling semantic early exit.
2. Token Consumption & Cost Telemetry: Tracks token usage across System One (Jev)
   and System Two (Generative LLM) with counterfactual fallback cost comparison (TCER).
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from typing import Any

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
