"""The Four Quantitative Graph Quality Metrics evaluation engine.

Evaluates offline knowledge graph extraction and construction across:
1. Entity Extraction Accuracy (Precision, Recall, F1)
2. Relation Extraction Accuracy (Ontology compliance & validity)
3. Entity Alignment Success Rate (Synonym resolution & canonical ID convergence)
4. Knowledge Coverage Rate (Atomic business facts reachability)

Matching is strict: identifiers are compared by canonical ID or declared
alias only (no substring matching), relation recall honours relation type and
direction, and a fact is covered only when its endpoints are connected.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from nga.ingestion.normalize import (
    NGA_ONTOLOGY_RULES,  # noqa: F401  (re-exported for backwards compatibility)
    canonical_id,
    check_relation,
)

logger = logging.getLogger("nga.evaluation.graph_eval")


def load_gold_benchmark(benchmark_path: str = "eval-graph/gold_graph_benchmark.json") -> dict[str, Any]:
    """Load the gold standard knowledge graph benchmark."""
    path = Path(benchmark_path)
    if not path.exists():
        raise FileNotFoundError(f"Gold graph benchmark not found at: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _canonical_slug(text: str) -> str:
    """Normalize text into an entity ID slug for comparison."""
    return canonical_id(text)


def build_alias_index(graph: Any) -> dict[str, Any]:
    """Map canonical ID of every node id / name / alias to its node.

    Node IDs take priority over names and aliases, so an alias that is also a
    separate node (an unmerged duplicate) resolves to that separate node.
    """
    index: dict[str, Any] = {}
    if graph is None:
        return index
    for node, data in graph.nodes(data=True):
        for label in [data.get("name", ""), *(data.get("aliases", []) or [])]:
            key = canonical_id(label)
            if key:
                index.setdefault(key, node)
    for node in graph.nodes:
        key = canonical_id(node)
        if key:
            index[key] = node
    return index


def resolve_node(graph: Any, ident: str, index: dict[str, Any] | None = None) -> Any | None:
    """Resolve an id, name or alias to a node in the graph (exact match only)."""
    idx = index if index is not None else build_alias_index(graph)
    return idx.get(canonical_id(ident))


def _edge_dicts(graph: Any, u: Any, v: Any) -> list[dict[str, Any]]:
    """All edge attribute dicts from u to v (direction-aware if directed)."""
    if not graph.has_edge(u, v):
        return []
    data = graph.get_edge_data(u, v)
    if graph.is_multigraph():
        return [dict(d) for d in data.values()]
    return [dict(data)]


def evaluate_entity_extraction(
    graph: Any,
    gold_entities: list[dict[str, Any]],
    gold_negative_entities: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate entity extraction Precision, Recall, and F1.

    Precision is measured against ``gold_negative_entities`` (known-wrong
    entities) when provided; otherwise against every extracted node.
    """
    negatives = gold_negative_entities or []
    if graph is None or graph.number_of_nodes() == 0:
        return {
            "precision": 0.0,
            "raw_precision": 0.0,
            "precision_basis": "negatives" if negatives else "extracted",
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "extracted_count": 0,
            "gold_count": len(gold_entities),
            "false_positive_count": 0,
            "by_type": {},
        }

    index = build_alias_index(graph)
    matched_nodes: set[Any] = set()
    type_stats: dict[str, dict[str, int]] = {}
    tp = 0

    for gold in gold_entities:
        gtype = gold.get("type", "General")
        stats = type_stats.setdefault(gtype, {"gold": 0, "tp": 0})
        stats["gold"] += 1
        node = index.get(canonical_id(gold.get("id", ""))) or index.get(canonical_id(gold.get("name", "")))
        if node is not None:
            tp += 1
            stats["tp"] += 1
            matched_nodes.add(node)

    negative_hits = 0
    for neg in negatives:
        if index.get(canonical_id(neg.get("id", ""))) is not None:
            negative_hits += 1

    extracted_count = graph.number_of_nodes()
    gold_count = len(gold_entities)

    raw_precision = round(len(matched_nodes) / max(extracted_count, 1), 4)
    if negatives:
        precision = round(len(matched_nodes) / max(len(matched_nodes) + negative_hits, 1), 4)
        basis = "negatives"
    else:
        precision = raw_precision
        basis = "extracted"
    recall = round(tp / max(gold_count, 1), 4)
    f1 = round((2 * precision * recall) / max(precision + recall, 1e-9), 4)

    return {
        "precision": precision,
        "raw_precision": raw_precision,
        "precision_basis": basis,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "extracted_count": extracted_count,
        "gold_count": gold_count,
        "false_positive_count": negative_hits,
        "by_type": {
            t: {
                "gold": s["gold"],
                "matched": s["tp"],
                "recall": round(s["tp"] / max(s["gold"], 1), 3),
            }
            for t, s in type_stats.items()
        },
    }


def evaluate_relation_extraction(
    graph: Any,
    gold_relations: list[dict[str, Any]],
    gold_negative_relations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate relation accuracy, ontology compliance, and gold edge recall.

    Unknown relation types and edges with missing endpoint types are invalid.
    Gold recall requires the same relation type, and the same direction when
    the graph is directed.
    """
    if graph is None or graph.number_of_edges() == 0:
        return {
            "relation_accuracy": 0.0,
            "ontology_compliance_rate": 0.0,
            "gold_relation_recall": 0.0,
            "audited_edges": 0,
            "valid_edges": 0,
            "type_missing_count": 0,
            "violations_count": 0,
            "violations": [],
        }

    edges = list(graph.edges(data=True))
    total_edges = len(edges)
    valid_count = 0
    type_missing = 0
    violations: list[dict[str, Any]] = []
    violations_by_relation: Counter[str] = Counter()
    directed = graph.is_directed()

    for u, v, data in edges:
        rel_type = str(data.get("relation", "related")).lower().strip()
        u_type = str(graph.nodes.get(u, {}).get("type", "")).lower()
        v_type = str(graph.nodes.get(v, {}).get("type", "")).lower()

        reason = check_relation(rel_type, u_type, v_type)
        if reason is not None and not directed:
            # Undirected edges have no meaningful orientation: accept either.
            if check_relation(rel_type, v_type, u_type) is None:
                reason = None
        if reason is None:
            valid_count += 1
        else:
            if not u_type or not v_type:
                type_missing += 1
            violations_by_relation[rel_type] += 1
            violations.append({
                "edge": f"({u}) --[{rel_type}]--> ({v})",
                "source_type": u_type,
                "target_type": v_type,
                "reason": reason,
            })

    ontology_compliance_rate = round(valid_count / max(total_edges, 1), 4)

    index = build_alias_index(graph)
    gold_hits = 0
    for gold in gold_relations:
        src = resolve_node(graph, gold["source"], index)
        tgt = resolve_node(graph, gold["target"], index)
        if src is None or tgt is None:
            continue
        want = str(gold.get("relation", "")).lower().strip()
        candidates = _edge_dicts(graph, src, tgt)
        if any(
            not want or str(c.get("relation", "")).lower().strip() == want
            for c in candidates
        ):
            gold_hits += 1

    gold_recall = round(gold_hits / max(len(gold_relations), 1), 4)

    negative_hits = 0
    for neg in gold_negative_relations or []:
        src = resolve_node(graph, neg["source"], index)
        tgt = resolve_node(graph, neg["target"], index)
        if src is None or tgt is None:
            continue
        want = str(neg.get("relation", "")).lower().strip()
        if any(
            not want or str(c.get("relation", "")).lower().strip() == want
            for c in _edge_dicts(graph, src, tgt)
        ):
            negative_hits += 1

    return {
        "relation_accuracy": ontology_compliance_rate,
        "ontology_compliance_rate": ontology_compliance_rate,
        "gold_relation_recall": gold_recall,
        "audited_edges": total_edges,
        "valid_edges": valid_count,
        "type_missing_count": type_missing,
        "violations_count": len(violations),
        "violations_by_relation": dict(violations_by_relation),
        "negative_relation_hits": negative_hits,
        "violations": violations[:10],  # show top 10 violations
    }


def evaluate_entity_alignment(graph: Any, gold_synonyms: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate entity alignment success rate (alias resolution to canonical ID).

    An alias is resolved only if it maps to the *same* node as the canonical
    ID. An alias that still exists as its own node is an unmerged duplicate.
    """
    if not gold_synonyms:
        return {"alignment_success_rate": 1.0, "total_pairs": 0, "successful_pairs": 0}

    index = build_alias_index(graph)
    total_pairs = 0
    successful_pairs = 0
    details = []

    for item in gold_synonyms:
        canonical_node = index.get(canonical_id(item["canonical_id"]))
        for alias in item.get("aliases", []):
            total_pairs += 1
            resolved_node = index.get(canonical_id(alias))
            resolved = canonical_node is not None and resolved_node == canonical_node
            if resolved:
                successful_pairs += 1
            details.append({
                "alias": alias,
                "canonical_expected": canonical_id(item["canonical_id"]),
                "resolved_node": str(resolved_node) if resolved_node is not None else None,
                "success": resolved,
            })

    rate = round(successful_pairs / max(total_pairs, 1), 4)
    return {
        "alignment_success_rate": rate,
        "total_pairs": total_pairs,
        "successful_pairs": successful_pairs,
        "details": details,
    }


def evaluate_knowledge_coverage(
    graph: Any, gold_atomic_facts: list[dict[str, Any]], max_hops: int = 3
) -> dict[str, Any]:
    """Evaluate knowledge coverage rate.

    A fact is covered only when both its subject and object exist **and** are
    connected within ``max_hops`` (treating edges as undirected for reach).
    """
    if not gold_atomic_facts:
        return {"knowledge_coverage_rate": 1.0, "total_facts": 0, "covered_facts": 0}

    import networkx as nx

    index = build_alias_index(graph)
    undirected = graph.to_undirected(as_view=True) if graph is not None else None

    covered_facts = 0
    fact_results = []

    for fact in gold_atomic_facts:
        sub_node = index.get(canonical_id(fact.get("subject", "")))
        obj_node = index.get(canonical_id(fact.get("object", "")))
        sub_present = sub_node is not None
        obj_present = obj_node is not None

        connected = False
        if sub_present and obj_present and undirected is not None:
            if sub_node == obj_node:
                connected = True
            else:
                try:
                    connected = nx.shortest_path_length(undirected, sub_node, obj_node) <= max_hops
                except nx.NetworkXNoPath:
                    connected = False

        is_covered = sub_present and obj_present and connected
        if is_covered:
            covered_facts += 1

        fact_results.append({
            "id": fact.get("id", "FACT"),
            "fact": fact.get("fact", ""),
            "subject": canonical_id(fact.get("subject", "")),
            "object": canonical_id(fact.get("object", "")),
            "subject_present": sub_present,
            "object_present": obj_present,
            "connected": connected,
            "covered": is_covered,
        })

    rate = round(covered_facts / max(len(gold_atomic_facts), 1), 4)
    return {
        "knowledge_coverage_rate": rate,
        "total_facts": len(gold_atomic_facts),
        "covered_facts": covered_facts,
        "facts": fact_results,
    }


def evaluate_graph_health(graph: Any) -> dict[str, Any]:
    """Structural health checks that need no gold labels."""
    if graph is None or graph.number_of_nodes() == 0:
        return {
            "node_count": 0, "edge_count": 0, "isolated_node_ratio": 0.0,
            "duplicate_name_clusters": 0, "edges_without_provenance": 0,
            "provenance_coverage": 0.0, "degree_outliers": [],
        }

    import networkx as nx

    n = graph.number_of_nodes()
    isolated = sum(1 for _ in nx.isolates(graph))

    by_name: dict[str, list[Any]] = {}
    for node, data in graph.nodes(data=True):
        key = canonical_id(data.get("name", ""))
        if key:
            by_name.setdefault(key, []).append(node)
    dup_clusters = sum(1 for v in by_name.values() if len(v) > 1)

    edges = list(graph.edges(data=True))
    no_prov = sum(1 for _, _, d in edges if not d.get("sources"))
    degrees = dict(graph.degree())
    mean_deg = sum(degrees.values()) / max(n, 1)
    threshold = max(10, 5 * mean_deg)
    outliers = [
        {"node": str(k), "degree": d}
        for k, d in sorted(degrees.items(), key=lambda kv: -kv[1])
        if d >= threshold
    ][:10]

    return {
        "node_count": n,
        "edge_count": len(edges),
        "isolated_node_ratio": round(isolated / n, 4),
        "duplicate_name_clusters": dup_clusters,
        "edges_without_provenance": no_prov,
        "provenance_coverage": round(1 - no_prov / max(len(edges), 1), 4) if edges else 0.0,
        "degree_outliers": outliers,
    }


def evaluate_graph_quality(
    graph: Any,
    benchmark_path: str = "eval-graph/gold_graph_benchmark.json",
    reports_dir: str = "reports/eval",
) -> dict[str, Any]:
    """Run full evaluation across the four quantitative graph quality metrics."""
    benchmark = load_gold_benchmark(benchmark_path)

    entity_metrics = evaluate_entity_extraction(
        graph, benchmark.get("gold_entities", []), benchmark.get("gold_negative_entities", [])
    )
    relation_metrics = evaluate_relation_extraction(
        graph, benchmark.get("gold_relations", []), benchmark.get("gold_negative_relations", [])
    )
    alignment_metrics = evaluate_entity_alignment(graph, benchmark.get("gold_synonyms", []))
    coverage_metrics = evaluate_knowledge_coverage(graph, benchmark.get("gold_atomic_facts", []))
    health = evaluate_graph_health(graph)

    # Graph Quality Index (weighted average of the 4 metrics)
    # Weights: Entity F1 (30%), Relation Acc (25%), Alignment Rate (25%), Coverage Rate (20%)
    gqi = round(
        0.30 * entity_metrics["f1"]
        + 0.25 * relation_metrics["relation_accuracy"]
        + 0.25 * alignment_metrics["alignment_success_rate"]
        + 0.20 * coverage_metrics["knowledge_coverage_rate"],
        4,
    )

    targets = {
        "entity_extraction_f1": 0.90,
        "relation_accuracy": 0.85,
        "alignment_success_rate": 0.95,
        "knowledge_coverage_rate": 0.90,
        "graph_quality_index": 0.90,
    }

    status = {
        "entity_f1_pass": entity_metrics["f1"] >= targets["entity_extraction_f1"],
        "relation_acc_pass": relation_metrics["relation_accuracy"] >= targets["relation_accuracy"],
        "alignment_pass": alignment_metrics["alignment_success_rate"] >= targets["alignment_success_rate"],
        "coverage_pass": coverage_metrics["knowledge_coverage_rate"] >= targets["knowledge_coverage_rate"],
        "overall_pass": gqi >= targets["graph_quality_index"],
    }

    result = {
        "timestamp": benchmark.get("version", "1.0"),
        "node_count": graph.number_of_nodes() if graph else 0,
        "edge_count": graph.number_of_edges() if graph else 0,
        "graph_quality_index": gqi,
        "status": status,
        "targets": targets,
        "entity_extraction": entity_metrics,
        "relation_extraction": relation_metrics,
        "entity_alignment": alignment_metrics,
        "knowledge_coverage": coverage_metrics,
        "graph_health": health,
    }

    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "graph_quality_metrics.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Graph quality metrics report saved to %s", report_file)

    return result
