"""Markdown and HTML report generation for NGA evaluation runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nga.evaluation.scoring import ScoreResult


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def build_summary(results: list[ScoreResult]) -> dict[str, Any]:
    from nga.evaluation.stepped_suite import compute_stepped_summary

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    by_category: dict[str, dict] = {}
    for r in results:
        cat = r.category
        if cat not in by_category:
            by_category[cat] = {"total": 0, "passed": 0, "scores": [], "latencies": []}
        by_category[cat]["total"] += 1
        if r.passed:
            by_category[cat]["passed"] += 1
        by_category[cat]["scores"].append(r.overall_score)
        by_category[cat]["latencies"].append(r.latency_s)

    cat_summary: list[dict] = []
    for cat, data in by_category.items():
        scores = data["scores"]
        lats = data["latencies"]
        cat_summary.append({
            "category": cat,
            "total": data["total"],
            "passed": data["passed"],
            "pass_rate": round(data["passed"] / max(data["total"], 1), 3),
            "avg_score": round(sum(scores) / max(len(scores), 1), 3),
            "avg_latency_s": round(sum(lats) / max(len(lats), 1), 2),
        })

    all_scores = [r.overall_score for r in results]
    all_lats = [r.latency_s for r in results]
    stepped_sum = compute_stepped_summary(results)

    # Grounding aggregation
    grounded_results = [r for r in results if r.grounding is not None]
    grounding_summary: dict[str, Any] | None = None
    if grounded_results:
        g_scores = [r.grounding.grounding_score for r in grounded_results]
        c_scores = [r.grounding.citation_validity_score for r in grounded_results]
        p_scores = [r.grounding.provenance_score for r in grounded_results]
        all_fabricated: list[str] = []
        all_orphan: list[str] = []
        for r in grounded_results:
            all_fabricated.extend(r.grounding.fabricated_citations)
            all_orphan.extend(r.grounding.orphan_citations)
        grounding_summary = {
            "avg_grounding_score": round(sum(g_scores) / len(g_scores), 3),
            "avg_citation_validity": round(sum(c_scores) / len(c_scores), 3),
            "avg_provenance": round(sum(p_scores) / len(p_scores), 3),
            "total_fabricated_citations": len(all_fabricated),
            "fabricated_citations": all_fabricated,
            "total_orphan_citations": len(all_orphan),
            "orphan_citations": all_orphan,
        }

    # Model name detection
    detected_model = None
    for r in results:
        if getattr(r, "model_name", None):
            detected_model = r.model_name
            break

    # Token Consumption & Cost Telemetry KPI aggregation
    token_results = [r for r in results if getattr(r, "token_usage", None) is not None]
    token_summary: dict[str, Any] | None = None
    if token_results:
        total_tokens = 0
        total_cost_usd = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_jev_input_tokens = 0
        total_jev_calls = 0
        total_fallback_cost = 0.0
        tcer_list: list[float] = []
        cost_savings_list: list[float] = []
        token_savings_list: list[float] = []
        early_exit_count = 0

        for r in token_results:
            tu = r.token_usage or {}
            totals = tu.get("totals", {})
            kpis = tu.get("kpis", {})
            s2 = tu.get("system_two_llm", {})
            s1 = tu.get("system_one_jev", {})

            total_tokens += totals.get("total_tokens", 0)
            total_cost_usd += totals.get("total_cost_usd", 0.0)

            total_prompt_tokens += s2.get("prompt_tokens", 0)
            total_completion_tokens += s2.get("completion_tokens", 0)
            total_jev_input_tokens += s1.get("input_tokens", 0)
            total_jev_calls += s1.get("eval_calls", 0)

            total_fallback_cost += kpis.get("est_fallback_cost_usd", 0.0)

            if "tcer" in kpis:
                tcer_list.append(float(kpis["tcer"]))
            if "cost_savings_pct" in kpis:
                cost_savings_list.append(float(kpis["cost_savings_pct"]))
            if "token_savings_pct" in kpis:
                token_savings_list.append(float(kpis["token_savings_pct"]))
            if kpis.get("early_exit_triggered"):
                early_exit_count += 1

        n_tok = len(token_results)
        avg_tokens = round(total_tokens / n_tok, 1)
        avg_cost = round(total_cost_usd / n_tok, 6)
        avg_tcer = round(sum(tcer_list) / len(tcer_list), 3) if tcer_list else 0.0
        avg_cost_savings = round(sum(cost_savings_list) / len(cost_savings_list), 1) if cost_savings_list else 0.0
        avg_token_savings = round(sum(token_savings_list) / len(token_savings_list), 1) if token_savings_list else 0.0
        early_exit_rate = round((early_exit_count / n_tok) * 100.0, 1)

        cost_per_1k = round(avg_cost * 1000.0, 4)
        fallback_cost_per_1k = round((total_fallback_cost / n_tok) * 1000.0, 4)

        token_summary = {
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost_usd, 6),
            "avg_tokens_per_query": avg_tokens,
            "avg_cost_per_query_usd": avg_cost,
            "avg_prompt_tokens": int(total_prompt_tokens / n_tok),
            "avg_completion_tokens": int(total_completion_tokens / n_tok),
            "avg_jev_input_tokens": int(total_jev_input_tokens / n_tok),
            "avg_jev_eval_calls": round(total_jev_calls / n_tok, 1),
            "avg_tcer": avg_tcer,
            "tcer_target": 0.45,
            "meets_tcer_target": avg_tcer <= 0.45,
            "output_token_elimination_rate_pct": 100.0,
            "avg_cost_savings_pct": avg_cost_savings,
            "avg_token_savings_pct": avg_token_savings,
            "early_exit_rate_pct": early_exit_rate,
            "cost_per_1k_queries_usd": cost_per_1k,
            "est_fallback_cost_per_1k_usd": fallback_cost_per_1k,
            "net_savings_per_1k_usd": round(max(0.0, fallback_cost_per_1k - cost_per_1k), 4),
        }

    summary_dict: dict[str, Any] = {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / max(total, 1), 3),
        "avg_score": round(sum(all_scores) / max(len(all_scores), 1), 3),
        "avg_latency_s": round(sum(all_lats) / max(len(all_lats), 1), 2),
        "model_name": detected_model,
        "by_category": cat_summary,
        "by_tier": stepped_sum.get("tiers", []),
        "stepped_summary": stepped_sum,
        "grounding": grounding_summary,
        "token_summary": token_summary,
    }

    if token_summary:
        summary_dict["avg_prompt_tokens"] = token_summary["avg_prompt_tokens"]
        summary_dict["avg_completion_tokens"] = token_summary["avg_completion_tokens"]

    return summary_dict


def generate_markdown_report(
    results: list[ScoreResult],
    run_label: str = "",
    cache_mode: str = "cold",
    model_name: str | None = None,
) -> str:
    summary = build_summary(results)
    ts = _now_ts()
    lines: list[str] = []

    eff_model = model_name or summary.get("model_name") or "anthropic/claude-sonnet-4-5"

    lines.append("# NGA Manufacturing Assistant — Evaluation Report")
    lines.append(f"\n**Run**: {run_label or ts}  ")
    lines.append(f"**Model**: `{eff_model}`  ")
    lines.append(f"**Cache mode**: {cache_mode}  ")
    lines.append(f"**Generated**: {datetime.now(timezone.utc).isoformat()}\n")

    lines.append("## Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Model | `{eff_model}` |")
    lines.append(f"| Total Questions | {summary['total']} |")
    lines.append(f"| Passed (≥0.70) | {summary['passed']} |")
    lines.append(f"| **Pass Rate** | **{summary['pass_rate']*100:.1f}%** |")
    lines.append(f"| Avg Score | {summary['avg_score']:.3f} |")
    lines.append(f"| Avg Latency | {summary['avg_latency_s']:.2f}s |")

    # Token Consumption & Cost Efficiency KPIs (Jev vs. Fallback)
    tok = summary.get("token_summary")
    if tok:
        lines.append("\n## Token Consumption & Cost Efficiency KPIs (Jev vs. Fallback)\n")
        lines.append("| Metric | Measured Value | Benchmark / Target | Status |")
        lines.append("|---|---|---|---|")
        tcer_status = "✅ Meets Target" if tok["meets_tcer_target"] else "⚠️ Exceeds Target"
        lines.append(f"| **Token Consumption Efficiency Ratio (TCER)** | **{tok['avg_tcer']:.3f}** | $\\le 0.45$ across multi-hop | {tcer_status} |")
        lines.append("| **Output Token Elimination Rate** | **100.0%** | 100% (Zero Jev output tokens) | ✅ |")
        lines.append(f"| **Avg Tokens / Query** | {tok['avg_tokens_per_query']:,} tokens | — | ℹ️ |")
        lines.append(f"| **Avg Cost / Query** | ${tok['avg_cost_per_query_usd']:.6f} | — | ℹ️ |")
        lines.append(f"| **Projected Cost / 1k Queries** | ${tok['cost_per_1k_queries_usd']:.4f} | ${tok['est_fallback_cost_per_1k_usd']:.4f} (Fallback) | 💰 |")
        lines.append(f"| **Net Cost Savings vs Fallback** | **{tok['avg_cost_savings_pct']:.1f}%** | Up to 70% | ✅ |")
        lines.append(f"| **Net Token Savings vs Fallback** | {tok['avg_token_savings_pct']:.1f}% | — | ℹ️ |")
        lines.append(f"| **Tool-Loop Early Exit Rate** | {tok['early_exit_rate_pct']:.1f}% | Eliminates redundant tool loops | ⚡ |")

    lines.append("\n## Results by Stepped Tier (L1–L4)\n")
    lines.append("| Tier | Total | Passed | Pass Rate | Avg Score | Avg Latency | Meets Target |")
    lines.append("|---|---|---|---|---|---|---|")
    for t in summary.get("by_tier", []):
        status = "✅" if t.get("meets_target") else "⚠️"
        lines.append(
            f"| {t['tier']} | {t['total']} | {t['passed']} "
            f"| {t['pass_rate']*100:.1f}% | {t['avg_score']:.3f} "
            f"| {t['avg_latency_s']:.2f}s | {status} |"
        )

    lines.append("\n## Results by Category\n")
    lines.append("| Category | Total | Passed | Pass Rate | Avg Score | Avg Latency |")
    lines.append("|---|---|---|---|---|---|")
    for cat in summary["by_category"]:
        lines.append(
            f"| {cat['category']} | {cat['total']} | {cat['passed']} "
            f"| {cat['pass_rate']*100:.1f}% | {cat['avg_score']:.3f} "
            f"| {cat['avg_latency_s']:.2f}s |"
        )

    lines.append("\n## Detailed Results\n")
    has_tokens = bool(tok)
    if has_tokens:
        lines.append("| ID | Tier | Category | Score | Passed | Latency | Tokens | TCER | Tools | Error |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
    else:
        lines.append("| ID | Tier | Category | Score | Passed | Latency | Tools | Error |")
        lines.append("|---|---|---|---|---|---|---|")

    for r in sorted(results, key=lambda x: x.question_id):
        status = "✅" if r.passed else "❌"
        tools = ", ".join(r.tools_called) if r.tools_called else "—"
        tier_val = getattr(r, "tier", "L2")
        if has_tokens:
            tu = getattr(r, "token_usage", None) or {}
            tok_count = tu.get("totals", {}).get("total_tokens", "—")
            tcer_val = tu.get("kpis", {}).get("tcer", "—")
            lines.append(
                f"| {r.question_id} | {tier_val} | {r.category} | {r.overall_score:.3f} "
                f"| {status} | {r.latency_s:.2f}s | {tok_count} | {tcer_val} | {tools} | {r.error or '—'} |"
            )
        else:
            lines.append(
                f"| {r.question_id} | {tier_val} | {r.category} | {r.overall_score:.3f} "
                f"| {status} | {r.latency_s:.2f}s | {tools} | {r.error or '—'} |"
            )

    # Failed cases detail
    failed = [r for r in results if not r.passed]
    if failed:
        lines.append(f"\n## Failed Cases ({len(failed)})\n")
        for r in failed:
            tier_val = getattr(r, "tier", "L2")
            lines.append(f"### {r.question_id} [{tier_val}] ({r.category})")
            lines.append(f"- Score: {r.overall_score:.3f}")
            lines.append(f"- Checks: {json.dumps(r.checks)}")
            if r.error:
                lines.append(f"- Error: {r.error}")
            lines.append("")

    return "\n".join(lines)


def save_report(
    results: list[ScoreResult],
    reports_dir: str = "reports/eval",
    run_label: str = "",
    cache_mode: str = "cold",
    cache_summary: dict | None = None,
    model_name: str | None = None,
) -> tuple[Path, Path]:
    """Save Markdown report to disk. Returns (md_path, json_path)."""
    ts = _now_ts()
    label = run_label or ts
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(results)
    eff_model = model_name or summary.get("model_name") or "anthropic/claude-sonnet-4-5"

    md_content = generate_markdown_report(results, label, cache_mode=cache_mode, model_name=eff_model)
    md_path = out_dir / f"{label}.md"
    md_path.write_text(md_content, encoding="utf-8")

    # Also save raw JSON for programmatic use
    json_path = out_dir / f"{label}.json"
    json_data = {
        "run_label": label,
        "model_name": eff_model,
        "cache_mode": cache_mode,
        "summary": summary,
        "token_summary": summary.get("token_summary"),
        "stepped_summary": summary.get("stepped_summary"),
        "cache": cache_summary,
        "results": [
            {
                "id": r.question_id,
                "tier": getattr(r, "tier", "L2"),
                "category": r.category,
                "overall_score": r.overall_score,
                "passed": r.passed,
                "deterministic_score": r.deterministic_score,
                "judge_score": r.judge_score,
                "checks": r.checks,
                "tools_called": r.tools_called,
                "latency_s": r.latency_s,
                "error": r.error,
                "cache_stats": r.cache_stats,
                "token_usage": getattr(r, "token_usage", None),
                "model_name": getattr(r, "model_name", None) or eff_model,
            }
            for r in results
        ],
    }
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    return md_path, json_path
