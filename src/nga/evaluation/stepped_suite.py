"""The 4-Tier Stepped Evaluation Test Suite (阶梯式评测集).

Evaluates the agentic GraphRAG system across four cognitive difficulty levels:
- Level 1: Single-Hop Fact Questions (Baseline vector retrieval)
- Level 2: Two-Hop Explicit Relation Questions (Graph connectivity)
- Level 3: Three-Hop Cross-Document Reasoning (Hybrid GraphRAG traversal)
- Level 4: Multi-Domain Comprehensive Challenge (SQL + SOP + Safety + Anti-Hallucination)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("nga.evaluation.stepped_suite")

TIER_NAMES = {
    "L1": "Level 1 — Single-Hop Fact Questions (单跳事实题)",
    "L2": "Level 2 — Two-Hop Explicit Relation Questions (两跳显式关系题)",
    "L3": "Level 3 — Three-Hop Cross-Document Complex Reasoning (三跳跨文档复杂推理题)",
    "L4": "Level 4 — Multi-Domain Comprehensive Challenge (跨多条业务线的综合难题)",
}

TIER_TARGETS = {
    "L1": {"min_pass_rate": 0.95, "max_latency_s": 3.0},
    "L2": {"min_pass_rate": 0.90, "max_latency_s": 5.0},
    "L3": {"min_pass_rate": 0.82, "max_latency_s": 9.0},
    "L4": {"min_pass_rate": 0.75, "max_latency_s": 15.0},
}


def load_stepped_questions(
    questions_path: str = "eval-questions/questions.json",
    tier: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load evaluation questions filtered by tier (L1, L2, L3, L4)."""
    path = Path(questions_path)
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    if tier and tier.upper() != "ALL":
        tier_clean = tier.upper()
        questions = [q for q in questions if q.get("tier", "").upper() == tier_clean]

    if limit and limit > 0:
        questions = questions[:limit]

    return questions


def compute_stepped_summary(results: list[Any]) -> dict[str, Any]:
    """Aggregate per-question eval results into a 4-tier stepped summary report."""
    tier_buckets: dict[str, list[dict[str, Any]]] = {
        "L1": [],
        "L2": [],
        "L3": [],
        "L4": [],
    }

    for r in results:
        # Accept either ScoreResult dataclass or dict
        if hasattr(r, "__dict__"):
            data = r.__dict__
        elif isinstance(r, dict):
            data = r
        else:
            continue

        qid = data.get("question_id") or data.get("id", "")
        tier = data.get("tier")

        # Fallback tier inference from id or category if not explicitly set
        if not tier:
            if qid.startswith("R") or qid in ["SQL1", "SQL2", "SQL3", "SQL4", "SQL5", "E1", "E2", "E3"]:
                tier = "L1"
            elif qid.startswith("M") and int(qid[1:]) <= 4:
                tier = "L2"
            elif qid.startswith("M") and int(qid[1:]) > 4:
                tier = "L3"
            elif qid.startswith("STR") or qid in ["S6", "S7", "S8"]:
                tier = "L4"
            else:
                tier = "L2"

        tier = tier.upper()
        if tier in tier_buckets:
            tier_buckets[tier].append(data)

    tier_stats = []
    pass_rates = {}

    for t in ["L1", "L2", "L3", "L4"]:
        items = tier_buckets[t]
        count = len(items)
        passed = sum(1 for item in items if item.get("passed"))
        pr = round(passed / max(count, 1), 3)
        pass_rates[t] = pr
        scores = [float(item.get("overall_score", item.get("score", 0.0))) for item in items]
        lats = [float(item.get("latency_s", 0.0)) for item in items]
        avg_score = round(sum(scores) / max(count, 1), 3)
        avg_lat = round(sum(lats) / max(count, 1), 2)

        target = TIER_TARGETS[t]
        meets_target = pr >= target["min_pass_rate"] and avg_lat <= target["max_latency_s"]

        tier_stats.append({
            "tier": t,
            "name": TIER_NAMES[t],
            "total": count,
            "passed": passed,
            "failed": count - passed,
            "pass_rate": pr,
            "avg_score": avg_score,
            "avg_latency_s": avg_lat,
            "target_pass_rate": target["min_pass_rate"],
            "target_latency_s": target["max_latency_s"],
            "meets_target": meets_target,
        })

    # Stepped degradation slope: (PassRate L1 - PassRate L4) / 3
    # Ideal: slope is small (≤ 0.07/tier), meaning reasoning doesn't drop off precipitously
    degradation_slope = round((pass_rates.get("L1", 0.0) - pass_rates.get("L4", 0.0)) / 3.0, 3)

    # Dominant failure mode diagnostics
    diagnostics = []
    if pass_rates.get("L1", 1.0) < 0.90:
        diagnostics.append("Level 1 Warning: Vector retrieval baseline underperforming. Verify embedding consistency or chunk overlap.")
    if pass_rates.get("L2", 1.0) < 0.85:
        diagnostics.append("Level 2 Warning: Direct graph connectivity gap. Missing key entity-relation links in graph store.")
    if pass_rates.get("L3", 1.0) < 0.75:
        diagnostics.append("Level 3 Warning: Cross-document hybrid traversal bottleneck. Check BFS hop depth or RRF merge cutoffs.")
    if pass_rates.get("L4", 1.0) < 0.70:
        diagnostics.append("Level 4 Warning: Multi-domain tool orchestration or safety criteria divergence under complex constraints.")

    return {
        "tiers": tier_stats,
        "pass_rates": pass_rates,
        "degradation_slope": degradation_slope,
        "diagnostics": diagnostics,
        "total_evaluated": sum(len(b) for b in tier_buckets.values()),
    }
