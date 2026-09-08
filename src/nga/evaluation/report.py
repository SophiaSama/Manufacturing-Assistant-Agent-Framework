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

    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / max(total, 1), 3),
        "avg_score": round(sum(all_scores) / max(len(all_scores), 1), 3),
        "avg_latency_s": round(sum(all_lats) / max(len(all_lats), 1), 2),
        "by_category": cat_summary,
        "by_tier": stepped_sum.get("tiers", []),
        "stepped_summary": stepped_sum,
    }


def generate_markdown_report(
    results: list[ScoreResult],
    run_label: str = "",
    cache_mode: str = "cold",
) -> str:
    summary = build_summary(results)
    ts = _now_ts()
    lines: list[str] = []

    lines.append("# NGA Manufacturing Assistant — Evaluation Report")
    lines.append(f"\n**Run**: {run_label or ts}  ")
    lines.append(f"**Cache mode**: {cache_mode}  ")
    lines.append(f"**Generated**: {datetime.now(timezone.utc).isoformat()}\n")

    lines.append("## Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Total Questions | {summary['total']} |")
    lines.append(f"| Passed (≥0.70) | {summary['passed']} |")
    lines.append(f"| **Pass Rate** | **{summary['pass_rate']*100:.1f}%** |")
    lines.append(f"| Avg Score | {summary['avg_score']:.3f} |")
    lines.append(f"| Avg Latency | {summary['avg_latency_s']:.2f}s |")

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
    lines.append("| ID | Tier | Category | Score | Passed | Latency | Tools | Error |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: x.question_id):
        status = "✅" if r.passed else "❌"
        tools = ", ".join(r.tools_called) if r.tools_called else "—"
        tier_val = getattr(r, "tier", "L2")
        lines.append(
            f"| {r.question_id} | {tier_val} | {r.category} | {r.overall_score:.3f} "
            f"| {status} | {r.latency_s:.2f}s | {tools} | {err} |"
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
) -> tuple[Path, Path]:
    """Save Markdown report to disk. Returns (md_path, json_path)."""
    ts = _now_ts()
    label = run_label or ts
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_content = generate_markdown_report(results, label, cache_mode=cache_mode)
    md_path = out_dir / f"{label}.md"
    md_path.write_text(md_content, encoding="utf-8")

    # Also save raw JSON for programmatic use
    json_path = out_dir / f"{label}.json"
    summary = build_summary(results)
    json_data = {
        "run_label": label,
        "cache_mode": cache_mode,
        "summary": summary,
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
            }
            for r in results
        ],
    }
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    return md_path, json_path
