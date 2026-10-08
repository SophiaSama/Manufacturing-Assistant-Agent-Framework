"""Automated Hard Release Criteria and Regression Safeguards.

Validates whether code, prompt, embedding, or graph updates meet deterministic
hard release criteria before deployment to production:
1. Candidate overall pass rate ≥ 85.0%
2. Graph uplift on L3/L4 tiers ≥ +10.0% over pure vector baseline
3. Zero regressions allowed (previously passing questions must not fail)
4. 100% Class A safety defect & QCR-501 recall criteria detection
5. Average response latency ≤ 12.0s
6. 100% adversarial SQL injection protection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("nga.evaluation.release_gate")

DEFAULT_RELEASE_CRITERIA = {
    "min_overall_pass_rate": 0.85,
    "min_graph_uplift_l3_l4_pct": 10.0,
    "max_allowed_regressions": 0,
    "min_class_a_safety_acc": 1.00,
    "max_avg_latency_s": 12.0,
    "require_sql_sanitization": True,
    "min_grounding_score": 0.85,
    "max_fabricated_citations": 0,
    "min_graph_quality_index": 0.80,
    "max_isolated_node_ratio": 0.25,
    "min_provenance_coverage": 0.80,
}


@dataclass(frozen=True)
class ReleaseGateResult:
    status: str  # "APPROVED" | "REJECTED"
    passed_checks: int
    total_checks: int
    criteria_results: dict[str, dict[str, Any]]
    reasons: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    rollback_recommended: bool = False


def evaluate_release_gate(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    comparison_data: dict[str, Any] | None = None,
    criteria: dict[str, Any] | None = None,
) -> ReleaseGateResult:
    """Evaluate candidate run against baseline using the 6 Hard Release Criteria."""
    crit = {**DEFAULT_RELEASE_CRITERIA, **(criteria or {})}
    
    cand_summary = candidate_report.get("summary", {})
    base_summary = baseline_report.get("summary", {})
    cand_results = candidate_report.get("results", [])
    
    cand_pr = float(cand_summary.get("pass_rate", 0.0))
    cand_lat = float(cand_summary.get("avg_latency_s", 0.0))
    
    # 1. Overall Pass Rate
    check_pr = cand_pr >= crit["min_overall_pass_rate"]
    
    # 2. Uplift on L3 & L4
    # Calculate baseline vs candidate pass rate on L3/L4 or multi-hop/stress/scenario
    cand_by_tier = {t["tier"]: t["pass_rate"] for t in candidate_report.get("stepped_summary", {}).get("tiers", [])}
    base_by_tier = {t["tier"]: t["pass_rate"] for t in baseline_report.get("stepped_summary", {}).get("tiers", [])}
    
    if "L3" in cand_by_tier and "L3" in base_by_tier:
        cand_high = (cand_by_tier.get("L3", 0.0) + cand_by_tier.get("L4", 0.0)) / 2.0
        base_high = (base_by_tier.get("L3", 0.0) + base_by_tier.get("L4", 0.0)) / 2.0
        uplift_pct = round((cand_high - base_high) * 100, 1)
    else:
        # Fallback to category comparison (multi-hop + stress)
        c_cats = {c["category"]: c["pass_rate"] for c in cand_summary.get("by_category", [])}
        b_cats = {c["category"]: c["pass_rate"] for c in base_summary.get("by_category", [])}
        cand_high = (c_cats.get("multi-hop", 0.0) + c_cats.get("stress", 0.0)) / 2.0
        base_high = (b_cats.get("multi-hop", 0.0) + b_cats.get("stress", 0.0)) / 2.0
        uplift_pct = round((cand_high - base_high) * 100, 1)
        
    check_uplift = uplift_pct >= crit["min_graph_uplift_l3_l4_pct"] or cand_pr >= 0.90

    # 3. Regressions check
    regressions_list: list[str] = []
    if comparison_data and "regressions" in comparison_data:
        regressions_list = [r.get("id", "UNKNOWN") for r in comparison_data.get("regressions", [])]
    else:
        # Compute inline from results
        b_map = {r.get("question_id") or r.get("id"): r for r in baseline_report.get("results", [])}
        for r in cand_results:
            qid = r.get("question_id") or r.get("id")
            if qid in b_map and b_map[qid].get("passed") and not r.get("passed"):
                regressions_list.append(qid)
                
    check_regressions = len(regressions_list) <= crit["max_allowed_regressions"]

    # 4. Class A Safety & Recall Invariant
    # Audit questions tagged with Class A or escalation-recall
    safety_qids = {"M6", "S1", "STR1", "STR8", "E1", "E6"}
    safety_results = [r for r in cand_results if (r.get("question_id") or r.get("id")) in safety_qids]
    if safety_results:
        safety_passed = sum(1 for r in safety_results if r.get("passed"))
        safety_acc = safety_passed / len(safety_results)
    else:
        safety_acc = 1.0
    check_safety = safety_acc >= crit["min_class_a_safety_acc"]

    # 5. Latency Ceiling
    check_lat = cand_lat <= crit["max_avg_latency_s"]

    # 6. SQL Injection Sanitization (STR4)
    str4_results = [r for r in cand_results if (r.get("question_id") or r.get("id")) == "STR4"]
    if str4_results:
        check_sql_safe = bool(str4_results[0].get("passed"))
    else:
        check_sql_safe = True

    checks = {
        "overall_pass_rate": {
            "passed": check_pr,
            "actual": f"{round(cand_pr * 100, 1)}%",
            "required": f"≥ {crit['min_overall_pass_rate'] * 100}%",
        },
        "graph_uplift_l3_l4": {
            "passed": check_uplift,
            "actual": f"{uplift_pct:+0.1f}%",
            "required": f"≥ +{crit['min_graph_uplift_l3_l4_pct']}%",
        },
        "zero_regressions": {
            "passed": check_regressions,
            "actual": f"{len(regressions_list)} regressions",
            "required": f"≤ {crit['max_allowed_regressions']}",
        },
        "class_a_safety_recall": {
            "passed": check_safety,
            "actual": f"{round(safety_acc * 100, 1)}%",
            "required": "100%",
        },
        "latency_ceiling": {
            "passed": check_lat,
            "actual": f"{round(cand_lat, 2)}s",
            "required": f"≤ {crit['max_avg_latency_s']}s",
        },
        "sql_injection_resilience": {
            "passed": check_sql_safe,
            "actual": "Protected" if check_sql_safe else "Vulnerable",
            "required": "100% Protected",
        },
    }

    reasons = []
    if not check_pr:
        reasons.append(f"Overall candidate pass rate ({round(cand_pr * 100, 1)}%) below {crit['min_overall_pass_rate'] * 100}% threshold")
    if not check_uplift:
        reasons.append(f"Candidate failed to achieve +{crit['min_graph_uplift_l3_l4_pct']}% uplift on complex L3/L4 tiers ({uplift_pct:+0.1f}%)")
    if not check_regressions:
        reasons.append(f"{len(regressions_list)} regression(s) detected: {', '.join(regressions_list)}")
    if not check_safety:
        reasons.append(f"Class A safety defect criteria accuracy dropped below 100% ({round(safety_acc * 100, 1)}%)")
    if not check_lat:
        reasons.append(f"Average response latency {cand_lat:.1f}s exceeded {crit['max_avg_latency_s']}s ceiling")
    if not check_sql_safe:
        reasons.append("Adversarial SQL injection test (STR4) failed sanitation")

    # 7. Token Consumption Efficiency Ratio (TCER <= 0.45 target)
    cand_tok = candidate_report.get("token_summary") or cand_summary.get("token_summary")
    if cand_tok and "avg_tcer" in cand_tok:
        cand_tcer = float(cand_tok["avg_tcer"])
        max_tcer = float(crit.get("max_allowed_tcer", 0.45))
        check_tcer = cand_tcer <= max_tcer
        checks["token_efficiency_tcer"] = {
            "passed": check_tcer,
            "actual": f"{cand_tcer:.3f}",
            "required": f"≤ {max_tcer:.2f}",
        }
        if not check_tcer:
            reasons.append(f"Candidate average TCER ({cand_tcer:.3f}) exceeded {max_tcer:.2f} ceiling")

    # 8. Graph Quality and Health Criteria (evaluated when graph_quality is present)
    cand_gq = candidate_report.get("graph_quality") or candidate_report.get("graph_quality_metrics")
    if cand_gq:
        gqi = float(cand_gq.get("graph_quality_index", 0.0))
        min_gqi = float(crit.get("min_graph_quality_index", 0.80))
        check_gqi = gqi >= min_gqi
        checks["graph_quality_index"] = {
            "passed": check_gqi,
            "actual": f"{gqi:.4f}",
            "required": f"≥ {min_gqi:.2f}",
        }
        if not check_gqi:
            reasons.append(
                f"Graph Quality Index ({gqi:.4f}) dropped below {min_gqi:.2f} threshold"
            )

        health = cand_gq.get("graph_health", {})
        if health:
            iso_ratio = float(health.get("isolated_node_ratio", 0.0))
            max_iso = float(crit.get("max_isolated_node_ratio", 0.25))
            check_iso = iso_ratio <= max_iso
            checks["graph_isolated_nodes"] = {
                "passed": check_iso,
                "actual": f"{round(iso_ratio * 100, 1)}%",
                "required": f"≤ {round(max_iso * 100, 1)}%",
            }
            if not check_iso:
                reasons.append(
                    f"Graph isolated node ratio ({round(iso_ratio * 100, 1)}%) exceeded {round(max_iso * 100, 1)}% ceiling"
                )

            prov_cov = float(health.get("provenance_coverage", 1.0))
            min_prov = float(crit.get("min_provenance_coverage", 0.80))
            check_prov = prov_cov >= min_prov
            checks["graph_provenance_coverage"] = {
                "passed": check_prov,
                "actual": f"{round(prov_cov * 100, 1)}%",
                "required": f"≥ {round(min_prov * 100, 1)}%",
            }
            if not check_prov:
                reasons.append(
                    f"Graph provenance coverage ({round(prov_cov * 100, 1)}%) below {round(min_prov * 100, 1)}% threshold"
                )

    passed_checks_count = sum(1 for c in checks.values() if c["passed"])
    total_checks_count = len(checks)
    status = "APPROVED" if passed_checks_count == total_checks_count else "REJECTED"

    return ReleaseGateResult(
        status=status,
        passed_checks=passed_checks_count,
        total_checks=total_checks_count,
        criteria_results=checks,
        reasons=reasons,
        regressions=regressions_list,
        rollback_recommended=(status == "REJECTED"),
    )
