"""A/B Testing harness: Pure Vector Baseline vs Hybrid GraphRAG Candidate.

Automates dual evaluation runs:
- Baseline (A): Pure dense vector retrieval (Chroma, graph=None)
- Candidate (B): Full Hybrid GraphRAG (Chroma + NetworkX BFS + RRF)
Computes delta matrices and validates Hard Release Criteria gates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from nga.evaluation.eval_runner import compare_evaluation_runs
from nga.evaluation.release_gate import ReleaseGateResult, evaluate_release_gate
from nga.evaluation.stepped_suite import compute_stepped_summary

logger = logging.getLogger("nga.evaluation.baseline_eval")


def run_baseline_ab_analysis(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    reports_dir: str = "reports/eval",
    label: str = "ab_vector_vs_graphrag",
) -> dict[str, Any]:
    """Perform comprehensive A/B evaluation analysis between pure vector and candidate."""
    # Ensure stepped summaries are computed on both reports
    if "stepped_summary" not in baseline_report:
        baseline_report["stepped_summary"] = compute_stepped_summary(baseline_report.get("results", []))
    if "stepped_summary" not in candidate_report:
        candidate_report["stepped_summary"] = compute_stepped_summary(candidate_report.get("results", []))

    comparison = compare_evaluation_runs(baseline_report, candidate_report)
    release_gate: ReleaseGateResult = evaluate_release_gate(
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        comparison_data=comparison,
    )

    # Compute tier-by-tier deltas
    b_tiers = {t["tier"]: t for t in baseline_report.get("stepped_summary", {}).get("tiers", [])}
    c_tiers = {t["tier"]: t for t in candidate_report.get("stepped_summary", {}).get("tiers", [])}

    stepped_deltas = []
    for tier in ["L1", "L2", "L3", "L4"]:
        bt = b_tiers.get(tier, {"pass_rate": 0.0, "avg_score": 0.0, "avg_latency_s": 0.0})
        ct = c_tiers.get(tier, {"pass_rate": 0.0, "avg_score": 0.0, "avg_latency_s": 0.0})
        stepped_deltas.append({
            "tier": tier,
            "pass_rate_baseline": bt.get("pass_rate", 0.0),
            "pass_rate_candidate": ct.get("pass_rate", 0.0),
            "pass_rate_delta_pct": round((ct.get("pass_rate", 0.0) - bt.get("pass_rate", 0.0)) * 100, 1),
            "score_baseline": bt.get("avg_score", 0.0),
            "score_candidate": ct.get("avg_score", 0.0),
            "score_delta": round(ct.get("avg_score", 0.0) - bt.get("avg_score", 0.0), 3),
            "latency_baseline": bt.get("avg_latency_s", 0.0),
            "latency_candidate": ct.get("avg_latency_s", 0.0),
            "latency_delta": round(ct.get("avg_latency_s", 0.0) - bt.get("avg_latency_s", 0.0), 2),
        })

    result = {
        "analysis_label": label,
        "baseline_label": baseline_report.get("run_label", "Baseline (Pure Vector)"),
        "candidate_label": candidate_report.get("run_label", "Candidate (Hybrid GraphRAG)"),
        "release_gate": {
            "status": release_gate.status,
            "passed_checks": release_gate.passed_checks,
            "total_checks": release_gate.total_checks,
            "criteria_results": release_gate.criteria_results,
            "reasons": release_gate.reasons,
            "regressions": release_gate.regressions,
            "rollback_recommended": release_gate.rollback_recommended,
        },
        "summary_delta": comparison.get("summary_delta", {}),
        "stepped_deltas": stepped_deltas,
        "category_deltas": comparison.get("category_deltas", []),
        "counts": comparison.get("counts", {}),
        "regressions": comparison.get("regressions", []),
        "improvements": comparison.get("improvements", []),
    }

    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / f"{label}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved A/B baseline analysis to %s", report_file)

    return result
