"""Multi-Model A/B Evaluation Arena & Pareto Optimization.

Dispatches evaluation questions across multiple frontier LLMs (via OpenRouter):
- Claude 3.5 Sonnet
- GPT-4o
- Gemini 1.5 Pro & Flash
- DeepSeek V3 / R1
- xAI Grok-2

Calculates reasoning accuracy, TTFT / latency, token usage, and cost per 1k queries.
Plots the Pareto Efficiency Frontier (Accuracy vs Cost vs Speed).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("nga.evaluation.multi_model_eval")

# Benchmark model roster with OpenRouter slug and standard token pricing ($/1M tokens)
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "anthropic/claude-3.5-sonnet": {
        "display_name": "Claude 3.5 Sonnet",
        "provider": "Anthropic",
        "input_price_per_m": 3.00,
        "output_price_per_m": 15.00,
        "tier": "frontier",
    },
    "openai/gpt-4o": {
        "display_name": "GPT-4o",
        "provider": "OpenAI",
        "input_price_per_m": 2.50,
        "output_price_per_m": 10.00,
        "tier": "frontier",
    },
    "google/gemini-1.5-pro": {
        "display_name": "Gemini 1.5 Pro",
        "provider": "Google",
        "input_price_per_m": 1.25,
        "output_price_per_m": 5.00,
        "tier": "frontier",
    },
    "google/gemini-1.5-flash": {
        "display_name": "Gemini 1.5 Flash",
        "provider": "Google",
        "input_price_per_m": 0.075,
        "output_price_per_m": 0.30,
        "tier": "fast_economy",
    },
    "deepseek/deepseek-chat": {
        "display_name": "DeepSeek V3",
        "provider": "DeepSeek",
        "input_price_per_m": 0.14,
        "output_price_per_m": 0.28,
        "tier": "economy_frontier",
    },
    "x-ai/grok-2": {
        "display_name": "Grok-2",
        "provider": "xAI",
        "input_price_per_m": 2.00,
        "output_price_per_m": 10.00,
        "tier": "frontier",
    },
}


def calculate_query_cost(
    model_slug: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Calculate estimated cost in USD for a single query."""
    meta = MODEL_REGISTRY.get(model_slug, {
        "input_price_per_m": 1.00,
        "output_price_per_m": 3.00,
    })
    in_cost = (prompt_tokens / 1_000_000.0) * meta["input_price_per_m"]
    out_cost = (completion_tokens / 1_000_000.0) * meta["output_price_per_m"]
    return round(in_cost + out_cost, 6)


def compute_pareto_frontier(model_summaries: list[dict[str, Any]]) -> list[str]:
    """Identify models on the Pareto efficiency frontier (optimizing for Score vs Cost)."""
    frontier = []
    for candidate in model_summaries:
        cand_score = candidate.get("avg_score", 0.0)
        cand_cost = candidate.get("cost_per_1k", 0.0)
        
        # Candidate is dominated if another model has higher score AND lower or equal cost
        is_dominated = False
        for other in model_summaries:
            other_score = other.get("avg_score", 0.0)
            other_cost = other.get("cost_per_1k", 0.0)
            if (other_score > cand_score and other_cost <= cand_cost) or \
               (other_score >= cand_score and other_cost < cand_cost):
                is_dominated = True
                break
        if not is_dominated:
            frontier.append(candidate.get("model_slug", ""))
    return frontier


def generate_multi_model_analysis(
    runs_by_model: dict[str, dict[str, Any]],
    reports_dir: str = "reports/eval",
    label: str = "multi_model_benchmark",
) -> dict[str, Any]:
    """Collate and analyze multi-model runs into a comparative leaderboard and Pareto matrix."""
    model_summaries = []
    question_matrix: dict[str, dict[str, Any]] = {}

    for model_slug, report in runs_by_model.items():
        meta = MODEL_REGISTRY.get(model_slug, {
            "display_name": model_slug.split("/")[-1],
            "provider": "Custom",
            "tier": "unknown",
            "input_price_per_m": 1.0,
            "output_price_per_m": 2.0,
        })

        summary = report.get("summary", {})
        results = report.get("results", [])
        total = len(results) or 1
        passed = sum(1 for r in results if r.get("passed"))
        pr = round(passed / total, 3)
        avg_score = float(summary.get("avg_score", 0.0))
        avg_lat = float(summary.get("avg_latency_s", 0.0))

        # Approximate or aggregate token metrics
        avg_prompt_tokens = int(summary.get("avg_prompt_tokens", 1400))
        avg_completion_tokens = int(summary.get("avg_completion_tokens", 350))
        cost_per_query = calculate_query_cost(model_slug, avg_prompt_tokens, avg_completion_tokens)
        cost_per_1k = round(cost_per_query * 1000, 3)

        model_entry = {
            "model_slug": model_slug,
            "display_name": meta["display_name"],
            "provider": meta["provider"],
            "tier": meta["tier"],
            "total_questions": total,
            "passed": passed,
            "pass_rate": pr,
            "avg_score": avg_score,
            "avg_latency_s": avg_lat,
            "avg_prompt_tokens": avg_prompt_tokens,
            "avg_completion_tokens": avg_completion_tokens,
            "cost_per_query_usd": cost_per_query,
            "cost_per_1k": cost_per_1k,
        }
        model_summaries.append(model_entry)

        # Build question matrix for side-by-side inspector
        for r in results:
            qid = r.get("question_id") or r.get("id")
            if qid not in question_matrix:
                question_matrix[qid] = {
                    "id": qid,
                    "question": r.get("question", ""),
                    "category": r.get("category", ""),
                    "tier": r.get("tier", "L2"),
                    "answers": {},
                }
            question_matrix[qid]["answers"][model_slug] = {
                "answer": r.get("answer", ""),
                "passed": r.get("passed", False),
                "score": r.get("overall_score", 0.0),
                "latency_s": r.get("latency_s", 0.0),
            }

    # Identify Pareto efficient models
    pareto_models = compute_pareto_frontier(model_summaries)
    for m in model_summaries:
        m["is_pareto_efficient"] = m["model_slug"] in pareto_models

    result = {
        "analysis_label": label,
        "models": model_summaries,
        "pareto_efficient_models": pareto_models,
        "question_matrix": list(question_matrix.values()),
    }

    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / f"{label}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Multi-model evaluation analysis saved to %s", report_file)

    return result
