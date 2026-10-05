"""Unit tests for NgaEpisodicMemory."""

import pytest

from nga.memory.headroom_memory import NgaEpisodicMemory


class TestNgaEpisodicMemory:
    @pytest.fixture
    def memory_store(self, tmp_path):
        db_file = tmp_path / "test_headroom_memory.db"
        return NgaEpisodicMemory(db_path=str(db_file))

    def test_record_and_recall(self, memory_store):
        # 1. Save an approved incident resolution
        mem_id = memory_store.record_approved_decision(
            question="Spindle torque drift on Station 112 with tool TQ-6012",
            recommendation="Replaced dynamic transducer and re-calibrated per SOP-TEC-214",
            category="TEC",
            class_a_alert=True,
            approver="Lead Engineer Jane",
            user_id="engineer",
        )
        assert mem_id is not None

        # 2. Search for similar incidents
        results = memory_store.recall_similar_incidents(
            query="Torque drift on Station 112 spindle",
            user_id="engineer",
            top_k=2,
        )

        assert len(results) >= 1
        top_match = results[0]
        assert "Station 112" in top_match["content"]
        assert "SOP-TEC-214" in top_match["content"]
        assert top_match["score"] > 0.0

    def test_fallback_on_uninitialized_backend(self):
        # Non-existent or broken configuration handles errors gracefully
        mem = NgaEpisodicMemory(db_path="/invalid/nonexistent/path/db.sqlite")
        results = mem.recall_similar_incidents("Any query", top_k=2)
        assert results == []
