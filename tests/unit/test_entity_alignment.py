"""Tests for knowledge graph entity alignment using TypeSafe AI."""

from unittest.mock import MagicMock

import networkx as nx
import pytest

from nga.ingestion.entity_alignment import (
    align_entities,
    generate_candidate_pairs,
    get_typesafe_api_key,
    get_typesafe_model,
    merge_entities,
    route,
)


@pytest.fixture
def sample_graph_with_duplicates():
    """Build a graph with known duplicates from the manufacturing corpus."""
    G = nx.Graph()

    # RB-07 robot duplicates
    G.add_node("weld-robot-07", type="Machine", name="WELD-ROBOT-07 (RB-07)", description="RoboTech RX-700 welding robot", level_rank=2)
    G.add_node("rb-07", type="Machine", name="RB-07", description="Welding robot at station 42", level_rank=1)
    G.add_node("machine-rb-07", type="Machine", name="RB-07", description="Robot on line CV-2", level_rank=1)

    # Different machine that shouldn't be merged
    G.add_node("tq-6012", type="Machine", name="TorqMaster TQ-6012", description="Torque nutrunner tool", level_rank=1)

    # FAP-401 procedure duplicates
    G.add_node("fap-401", type="Procedure", name="FAP-401", description="Failure analysis 8D procedure", level_rank=1)
    G.add_node("procedure-fap-401", type="Procedure", name="FAP-401", description="8D failure analysis protocol", level_rank=1)

    # Connected edges
    G.add_edge("weld-robot-07", "station-42", relation="located_at")
    G.add_node("station-42", type="Station", name="Station 42", description="Body shop welding station", level_rank=1)

    return G


def test_env_resolution_defaults(monkeypatch):
    """Test dynamic model and API key environment variable resolution."""
    # Test model defaults back to jev-1.12 if unset
    monkeypatch.delenv("TYPESAFE_MODEL", raising=False)
    assert get_typesafe_model() == "jev-1.12"

    # Test model override via env var
    monkeypatch.setenv("TYPESAFE_MODEL", "jev-custom-test")
    assert get_typesafe_model() == "jev-custom-test"

    # Test API key resolution: TYPESAFE_API_KEY takes precedence
    monkeypatch.setenv("TYPESAFE_API_KEY", "key-main")
    monkeypatch.setenv("TYPESAFE_API_TEST", "key-test")
    assert get_typesafe_api_key() == "key-main"

    # Test fallback to TYPESAFE_API_TEST (for GitHub Actions CI secret)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert get_typesafe_api_key() == "key-test"

    # Test None when neither is set
    monkeypatch.delenv("TYPESAFE_API_TEST", raising=False)
    assert get_typesafe_api_key() is None


def test_missing_api_key_fails_ci(monkeypatch):
    """Verify that align_entities raises ValueError when no key is set (failing CI)."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_TEST", raising=False)

    G = nx.Graph()
    G.add_node("node1", name="Test Node 1")

    with pytest.raises(ValueError, match="Neither TYPESAFE_API_KEY nor TYPESAFE_API_TEST is set"):
        align_entities(G, require_api_key=True)


def test_generate_candidate_pairs(sample_graph_with_duplicates):
    """Test candidate blocking identifies the true duplicate pairs."""
    candidates = generate_candidate_pairs(sample_graph_with_duplicates)

    # Pairs are sorted tuples of IDs
    pair_set = set(candidates)

    # Should find RB-07 variants
    assert ("machine-rb-07", "rb-07") in pair_set or ("rb-07", "machine-rb-07") in pair_set
    assert ("rb-07", "weld-robot-07") in pair_set

    # Should find FAP-401 variants
    assert ("fap-401", "procedure-fap-401") in pair_set

    # Should NOT pair RB-07 with TQ-6012
    assert ("rb-07", "tq-6012") not in pair_set
    assert ("weld-robot-07", "tq-6012") not in pair_set


def test_routing_logic():
    """Test outcome mapping across the three Score levels."""
    # Level 0 (< 0.5): leave separate
    assert route(0.0) == "leave_separate"
    assert route(0.3) == "leave_separate"
    assert route(0.49) == "leave_separate"

    # Level 1 (0.5 <= score < 1.5): curator review queue
    assert route(0.5) == "review"
    assert route(1.0) == "review"
    assert route(1.4) == "review"

    # Level 2 (>= 1.5): merge
    assert route(1.5) == "merge"
    assert route(1.8) == "merge"
    assert route(2.0) == "merge"


def test_merge_entities(sample_graph_with_duplicates):
    """Test graph surgery: canonical selection, attribute preservation, and edge rewiring."""
    G = sample_graph_with_duplicates
    initial_nodes = G.number_of_nodes()

    # Merge the three RB-07 duplicates
    pairs = [("weld-robot-07", "rb-07"), ("rb-07", "machine-rb-07")]
    merged_count = merge_entities(G, pairs)

    assert merged_count == 2
    assert G.number_of_nodes() == initial_nodes - 2

    # Canonical node should be rb-07 (cleaner/shorter slug preferred over weld-robot-07 and machine-rb-07)
    assert "rb-07" in G
    assert "machine-rb-07" not in G
    assert "weld-robot-07" not in G

    canonical_data = G.nodes["rb-07"]
    # Check max level rank preserved (weld-robot-07 was rank 2)
    assert canonical_data["level_rank"] == 2
    # Check aliases list contains all IDs and names
    assert "weld-robot-07" in canonical_data["aliases"]
    assert "machine-rb-07" in canonical_data["aliases"]
    assert "WELD-ROBOT-07 (RB-07)" in canonical_data["aliases"]

    # Check edge to station-42 was preserved and rewired
    assert G.has_edge("rb-07", "station-42")


def test_align_entities_full_workflow(sample_graph_with_duplicates, monkeypatch, tmp_path):
    """Test full alignment workflow with mocked TypeSafe client."""
    monkeypatch.setenv("TYPESAFE_API_TEST", "mock-test-key")

    mock_client = MagicMock()

    def mock_system_one(state, questions, model):
        entity_a = state["entity_a"]
        entity_b = state["entity_b"]
        name_a = entity_a.get("name", "").lower()
        name_b = entity_b.get("name", "").lower()

        mock_resp = MagicMock()
        mock_resp.usage.input_tokens = 50
        mock_resp.usage.output_tokens = 5

        # If both are RB-07 or both are FAP-401, predict merge (Score = 1.9)
        if ("rb-07" in name_a and "rb-07" in name_b) or ("fap-401" in name_a and "fap-401" in name_b):
            mock_score = MagicMock(score=1.9, probabilities=[0.05, 0.05, 0.90], confidence=0.85)
        else:
            mock_score = MagicMock(score=0.1, probabilities=[0.90, 0.05, 0.05], confidence=0.85)

        mock_noul = MagicMock(noul=0.95)
        mock_resp.answers = {
            "link_state": mock_score,
            "same_name": mock_noul,
            "same_type": mock_noul,
            "same_description": mock_noul,
        }
        return mock_resp

    mock_client.system_one.side_effect = mock_system_one

    review_file = tmp_path / "review.json"
    report = align_entities(
        sample_graph_with_duplicates,
        client=mock_client,
        review_queue_path=str(review_file),
    )

    assert report["status"] == "completed"
    assert report["candidates_found"] > 0
    assert report["merged_count"] > 0
    assert sample_graph_with_duplicates.number_of_nodes() < 7
