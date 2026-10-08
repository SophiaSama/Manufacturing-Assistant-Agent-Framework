"""Tests for graph construction: validation, provenance, retry, directedness."""

import json
from types import SimpleNamespace

import networkx as nx

from nga.ingestion.build_graph import (
    build_graph_from_documents,
    extract_entities_and_relations,
)
from nga.ingestion.normalize import canonical_id, check_relation


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        reply = self.replies.pop(0)
        return SimpleNamespace(content=reply if isinstance(reply, str) else json.dumps(reply))


def doc(text="x", **meta):
    return SimpleNamespace(page_content=text, metadata={"source": "SOP-1", **meta})


def ent(eid, etype, conf=0.9, **kw):
    return {"id": eid, "type": etype, "name": eid, "confidence": conf, **kw}


def test_canonical_id_normalizes_variants():
    assert canonical_id("TQ_6012") == canonical_id("tq 6012") == canonical_id("TQ-6012") == "tq-6012"


def test_check_relation_rules():
    assert check_relation("requires", "sop", "machine") is None
    assert check_relation("requires", "sop", "part") is None
    assert check_relation("requires", "sop", "station") is None
    assert check_relation("requires", "workorder", "machine") is None
    assert check_relation("documented_in", "procedure", "sop") is None
    assert check_relation("requires", "machine", "sop") is not None
    assert check_relation("bogus", "sop", "machine") is not None
    assert check_relation("requires", "", "machine") is not None


def test_build_validates_and_records_provenance():
    payload = {
        "entities": [
            ent("SOP_101", "SOP"),
            ent("TQ 6012", "Machine", aliases=["TQ6012"]),
            ent("weird", "Spaceship"),
            ent("maybe", "Machine", conf=0.2),
        ],
        "relations": [
            {"source": "sop-101", "relation": "requires", "target": "tq-6012", "confidence": 0.9, "evidence": "q"},
            {"source": "sop-101", "relation": "monitors", "target": "tq-6012", "confidence": 0.9},  # ontology violation, no valid direction
            {"source": "sop-101", "relation": "likes", "target": "tq-6012"},  # unknown type
            {"source": "sop-101", "relation": "requires", "target": "ghost"},  # unknown endpoint
        ],
    }
    g = build_graph_from_documents([doc(chunk_id=3)], FakeLLM([payload]), min_confidence=0.5)
    assert isinstance(g, nx.MultiDiGraph)
    assert set(g.nodes) == {"sop-101", "tq-6012"}
    assert g.number_of_edges() == 1
    edge = g.edges["sop-101", "tq-6012", "requires"]
    assert edge["sources"][0]["document"] == "SOP-1" and edge["sources"][0]["chunk"] == 3
    assert g.nodes["tq-6012"]["aliases"] == ["TQ6012"]
    rep = g.graph["build_report"]
    assert rep["entities_dropped_invalid_type"] == 1
    assert rep["entities_low_confidence"] == 1
    assert rep["relations_dropped_ontology"] == 1
    assert rep["relations_dropped_invalid_type"] == 1
    assert rep["relations_dropped_unknown_endpoint"] == 1
    assert len(g.graph["review_queue"]) == 5


def test_cross_chunk_relation_and_source_merge():
    d1 = {"entities": [ent("sop-1", "SOP")], "relations": [
        {"source": "sop-1", "relation": "requires", "target": "m-1"}]}
    d2 = {"entities": [ent("m-1", "Machine", level_rank=2), ent("sop-1", "SOP", description="later")], "relations": []}
    g = build_graph_from_documents([doc(chunk_id=1), doc(chunk_id=2)], FakeLLM([d1, d2]))
    assert g.has_edge("sop-1", "m-1", key="requires")
    assert len(g.nodes["sop-1"]["sources"]) == 2
    assert g.nodes["m-1"]["level_rank"] == 2


def test_parse_failure_retries_then_succeeds():
    good = {"entities": [ent("a", "Machine")], "relations": []}
    llm = FakeLLM(["not json at all", json.dumps(good)])
    from collections import Counter
    stats = Counter()
    out = extract_entities_and_relations("text", llm, stats=stats)
    assert llm.calls == 2
    assert stats["parse_retries"] == 1 and stats["failed_windows"] == 0
    assert len(out["entities"]) == 1


def test_parse_failure_twice_is_counted():
    llm = FakeLLM(["bad", "still bad"])
    from collections import Counter
    stats = Counter()
    out = extract_entities_and_relations("text", llm, stats=stats)
    assert out == {"entities": [], "relations": []}
    assert stats["failed_windows"] == 1


def test_long_text_is_windowed_not_truncated():
    empty = {"entities": [], "relations": []}
    llm = FakeLLM([empty] * 5)
    extract_entities_and_relations("x" * 7000, llm)
    assert llm.calls >= 3


def test_search_traverses_incoming_edges():
    from nga.graphrag.search import build_graph_evidence

    g = nx.MultiDiGraph()
    g.add_node("seed", name="seed", level_rank=1)
    g.add_node("upstream", name="up", level_rank=1)
    g.add_edge("upstream", "seed", key="requires", relation="requires", level_rank=1)
    emb = SimpleNamespace(embed_query=lambda q: [1.0])
    g.nodes["seed"]["embedding"] = [1.0]
    ev = build_graph_evidence(g, "q", emb, k_entities=1, max_hops=2)
    assert {e["id"] for e in ev["entities"]} == {"seed", "upstream"}


def test_reversed_relation_direction_is_repaired():
    payload = {
        "entities": [ent("thr", "Threshold"), ent("sop-1", "SOP")],
        "relations": [{"source": "thr", "relation": "threshold_for", "target": "sop-1", "confidence": 0.9}],
    }
    g = build_graph_from_documents([doc()], FakeLLM([payload]))
    assert g.has_edge("sop-1", "thr", key="threshold_for")
    assert g.graph["build_report"]["relations_direction_fixed"] == 1


def test_provenance_uses_doc_id():
    payload = {"entities": [ent("m-1", "Machine")], "relations": []}
    d = SimpleNamespace(page_content="x", metadata={"doc_id": "SOP-OPR-101", "chunk_id": "SOP-OPR-101:c0:c"})
    g = build_graph_from_documents([d], FakeLLM([payload]))
    assert g.nodes["m-1"]["sources"][0]["document"] == "SOP-OPR-101"


class ExplodingLLM:
    def invoke(self, _messages):
        raise RuntimeError("403 Key limit exceeded")


def test_repeated_api_errors_abort_build():
    import pytest

    from nga.ingestion.build_graph import ExtractionAborted

    with pytest.raises(ExtractionAborted):
        build_graph_from_documents([doc() for _ in range(10)], ExplodingLLM())


def test_mostly_failed_windows_refuse_partial_graph():
    import pytest

    from nga.ingestion.build_graph import ExtractionAborted

    good = {"entities": [ent("a", "Machine")], "relations": []}
    llm = FakeLLM([good, "bad", "bad", "bad", "bad", "bad", "bad"])
    with pytest.raises(ExtractionAborted):
        build_graph_from_documents([doc(), doc(), doc()], llm)
