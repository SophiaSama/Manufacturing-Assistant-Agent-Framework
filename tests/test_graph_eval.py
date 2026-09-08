"""Tests for the Four Quantitative Graph Quality Metrics."""

import networkx as nx
import pytest
from nga.evaluation.graph_eval import (
    evaluate_entity_alignment,
    evaluate_entity_extraction,
    evaluate_graph_quality,
    evaluate_knowledge_coverage,
    evaluate_relation_extraction,
    load_gold_benchmark,
)


@pytest.fixture
def gold_benchmark():
    return load_gold_benchmark("eval-graph/gold_graph_benchmark.json")


@pytest.fixture
def sample_graph():
    G = nx.Graph()
    # Add canonical nodes
    G.add_node("sop-opr-101", type="SOP", name="SOP-OPR-101 Wheel Lug Nut Tightening")
    G.add_node("station-144", type="Station", name="Station 144 Chassis Assembly")
    G.add_node("tq-6012", type="Machine", name="TorqMaster TQ-6012 DC Electric Nutrunner", aliases=["TQ6012", "TorqMaster 6012"])
    G.add_node("torque-target-105nm", type="Threshold", name="105 Nm ± 5% Target Torque")
    G.add_node("tq-6018", type="Machine", name="TorqMaster TF-6000 (TQ-6018) Audit Wrench", aliases=["TF-6000", "TQ-6018"])
    G.add_node("sop-tec-214", type="SOP", name="SOP-TEC-214 Torque Tool Calibration")
    G.add_node("drift-threshold-2pct", type="Threshold", name="2.0% Maximum Drift Threshold")
    G.add_node("class-a-defect", type="DefectCode", name="Class A Safety Critical Defect")
    G.add_node("esc-402", type="EscalationLevel", name="ESC-402 Plant Escalation Protocol")
    G.add_node("qcr-501", type="RecallCriteria", name="QCR-501 Quality Recall Criteria")
    G.add_node("fap-401", type="Procedure", name="FAP-401 Failure Analysis & 8D Process")
    G.add_node("aurora-au-2025", type="Part", name="Aurora AU-2025 Vehicle Platform", aliases=["Aurora", "AU-2025", "Aurora AU-2025", "AU25", "Aurora 2025"])
    G.add_node("solstice-so-2025", type="Part", name="Solstice SO-2025 Vehicle Platform", aliases=["Solstice", "SO-2025", "Solstice SO-2025", "SO25", "Solstice 2025"])

    # Add edges
    G.add_edge("sop-opr-101", "tq-6012", relation="requires")
    G.add_edge("tq-6012", "torque-target-105nm", relation="monitors")
    G.add_edge("sop-opr-101", "station-144", relation="documented_in")
    G.add_edge("tq-6012", "tq-6018", relation="audited_by")
    G.add_edge("tq-6012", "sop-tec-214", relation="documented_in")
    G.add_edge("sop-tec-214", "drift-threshold-2pct", relation="threshold_for")
    G.add_edge("class-a-defect", "esc-402", relation="escalates_to")
    G.add_edge("class-a-defect", "qcr-501", relation="triggers_recall")
    G.add_edge("class-a-defect", "fap-401", relation="requires")
    return G


def test_gold_benchmark_structure(gold_benchmark):
    assert "gold_entities" in gold_benchmark
    assert "gold_relations" in gold_benchmark
    assert "gold_synonyms" in gold_benchmark
    assert "gold_atomic_facts" in gold_benchmark
    assert len(gold_benchmark["gold_entities"]) > 0
    assert len(gold_benchmark["gold_relations"]) > 0


def test_entity_extraction_evaluation(sample_graph, gold_benchmark):
    res = evaluate_entity_extraction(sample_graph, gold_benchmark["gold_entities"])
    assert res["true_positives"] > 0
    assert res["precision"] > 0.0
    assert res["recall"] > 0.0
    assert "by_type" in res


def test_relation_extraction_evaluation(sample_graph, gold_benchmark):
    res = evaluate_relation_extraction(sample_graph, gold_benchmark["gold_relations"])
    assert res["audited_edges"] == sample_graph.number_of_edges()
    assert res["relation_accuracy"] > 0.80
    assert res["gold_relation_recall"] > 0.0


def test_entity_alignment_evaluation(sample_graph, gold_benchmark):
    res = evaluate_entity_alignment(sample_graph, gold_benchmark["gold_synonyms"])
    assert res["total_pairs"] > 0
    assert res["alignment_success_rate"] > 0.50


def test_knowledge_coverage_evaluation(sample_graph, gold_benchmark):
    res = evaluate_knowledge_coverage(sample_graph, gold_benchmark["gold_atomic_facts"])
    assert res["total_facts"] > 0
    assert res["covered_facts"] > 0
    assert res["knowledge_coverage_rate"] > 0.0


def test_full_graph_quality_suite(sample_graph, tmp_path):
    res = evaluate_graph_quality(
        sample_graph,
        benchmark_path="eval-graph/gold_graph_benchmark.json",
        reports_dir=str(tmp_path),
    )
    assert "graph_quality_index" in res
    assert res["graph_quality_index"] > 0.0
    assert "status" in res
    assert "targets" in res
    assert (tmp_path / "graph_quality_metrics.json").exists()
