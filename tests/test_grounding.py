"""Unit tests for deterministic retrieval-grounding hallucination evaluation.

No LLM, no vector store, no API key required.
"""

from __future__ import annotations

import json

from nga.evaluation.grounding import (
    _CHUNK_ID_RE,
    GroundingResult,
    _extract_answer_citations,
    _extract_chunk_ids_from_tool_outputs,
    _extract_retrieved_doc_ids,
    evaluate_citation_existence,
    evaluate_grounding,
    evaluate_retrieval_provenance,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

CORPUS_MANIFEST = {
    "SOP-OPR-101",
    "SOP-OPR-114",
    "SOP-OPR-127",
    "SOP-OPR-142",
    "SOP-OPR-158",
    "SOP-TEC-214",
    "SOP-TEC-314",
    "TEC-301",
    "ESC-402",
    "QCR-501",
    "FAP-401",
    "SOP-OPR-160",
    "TR-2025-030",
}


def _tool_output(references: list[dict], results: list[dict] | None = None) -> str:
    payload: dict = {"query": "test", "categories": ["OPR"], "references": references}
    if results is not None:
        payload["results"] = results
    return json.dumps(payload)


def _ref(doc_id: str, chunk_id: str, source: str = "canonical") -> dict:
    return {
        "doc_id": doc_id,
        "doc_name": f"{doc_id}.md",
        "section": "1",
        "chunk_id": chunk_id,
        "category": "OPR",
        "corpus_source": source,
    }


# ── Tier 1: Citation Existence ───────────────────────────────────────────────

class TestCitationExistence:
    def test_all_valid(self):
        evidence = [{"citation": "SOP-OPR-101", "source_type": "document"}]
        result = evaluate_citation_existence(
            "See [SOP-OPR-101] for details.",
            evidence,
            [],
            CORPUS_MANIFEST,
        )
        assert result["citation_validity_score"] == 1.0
        assert result["fabricated_citations"] == []
        assert result["valid_citations"] == 1
        assert result["total_citations"] == 1

    def test_fabricated_citation(self):
        evidence = [
            {"citation": "SOP-OPR-101", "source_type": "document"},
            {"citation": "SOP-OPR-999", "source_type": "document"},
        ]
        result = evaluate_citation_existence(
            "See [SOP-OPR-101] and [SOP-OPR-999].",
            evidence,
            [],
            CORPUS_MANIFEST,
        )
        assert result["citation_validity_score"] == 0.5
        assert "SOP-OPR-999" in result["fabricated_citations"]
        assert result["valid_citations"] == 1
        assert result["total_citations"] == 2

    def test_inline_citations_extracted(self):
        result = evaluate_citation_existence(
            "Per [SOP-OPR-101] and [ESC-402], escalation is required.",
            [],  # no structured evidence
            [],
            CORPUS_MANIFEST,
        )
        assert result["citation_validity_score"] == 1.0
        assert result["total_citations"] == 2

    def test_no_citations_returns_perfect(self):
        """An answer with no citations scores 1.0 (nothing to validate)."""
        result = evaluate_citation_existence(
            "The torque is 105 Nm.",
            [],
            [],
            CORPUS_MANIFEST,
        )
        assert result["citation_validity_score"] == 1.0
        assert result["total_citations"] == 0

    def test_chunk_id_well_formed(self):
        assert _CHUNK_ID_RE.match("SOP-OPR-101:c0:can")
        assert _CHUNK_ID_RE.match("QCR-501:c12:var")
        assert not _CHUNK_ID_RE.match("FAKE:xyz")
        assert not _CHUNK_ID_RE.match("no-colon")

    def test_invalid_chunk_ids_detected(self):
        tool_out = _tool_output(
            [_ref("SOP-OPR-101", "SOP-OPR-101:c0:can")],
            [{"doc_id": "SOP-OPR-101", "chunk_id": "MALFORMED_CHUNK"}],
        )
        result = evaluate_citation_existence(
            "See [SOP-OPR-101].",
            [{"citation": "SOP-OPR-101", "source_type": "document"}],
            [tool_out],
            CORPUS_MANIFEST,
        )
        assert "MALFORMED_CHUNK" in result["invalid_chunk_ids"]

    def test_chunk_id_with_unknown_doc_prefix(self):
        tool_out = _tool_output(
            [_ref("SOP-OPR-101", "SOP-OPR-101:c0:can")],
            [{"doc_id": "FAKE-999", "chunk_id": "FAKE-999:c0:can"}],
        )
        result = evaluate_citation_existence(
            "See [SOP-OPR-101].",
            [{"citation": "SOP-OPR-101", "source_type": "document"}],
            [tool_out],
            CORPUS_MANIFEST,
        )
        # Well-formed pattern but doc_id not in manifest
        assert "FAKE-999:c0:can" in result["invalid_chunk_ids"]


# ── Tier 2: Retrieval Provenance ─────────────────────────────────────────────

class TestRetrievalProvenance:
    def test_all_retrieved(self):
        tool_out = _tool_output([_ref("SOP-OPR-101", "SOP-OPR-101:c0:can")])
        result = evaluate_retrieval_provenance(
            "See [SOP-OPR-101].",
            [{"citation": "SOP-OPR-101", "source_type": "document"}],
            [tool_out],
        )
        assert result["provenance_score"] == 1.0
        assert result["orphan_citations"] == []

    def test_orphan_citation(self):
        """Answer cites a doc that was never retrieved."""
        tool_out = _tool_output([_ref("SOP-OPR-101", "SOP-OPR-101:c0:can")])
        result = evaluate_retrieval_provenance(
            "See [SOP-OPR-101] and [ESC-402].",
            [
                {"citation": "SOP-OPR-101", "source_type": "document"},
                {"citation": "ESC-402", "source_type": "document"},
            ],
            [tool_out],
        )
        assert result["provenance_score"] == 0.5
        assert "ESC-402" in result["orphan_citations"]

    def test_no_citations_returns_perfect(self):
        result = evaluate_retrieval_provenance(
            "The torque is 105 Nm.", [], [],
        )
        assert result["provenance_score"] == 1.0

    def test_no_tool_outputs(self):
        """Citations exist but no retrieval tool ran → all orphaned."""
        result = evaluate_retrieval_provenance(
            "See [SOP-OPR-101].",
            [{"citation": "SOP-OPR-101", "source_type": "document"}],
            [],  # no tool outputs
        )
        assert result["provenance_score"] == 0.0
        assert "SOP-OPR-101" in result["orphan_citations"]

    def test_retrieval_refs_metadata_completeness(self):
        complete_ref = _ref("SOP-OPR-101", "SOP-OPR-101:c0:can")
        incomplete_ref = {"doc_id": "SOP-OPR-114", "doc_name": "SOP-OPR-114.md"}
        tool_out = _tool_output([complete_ref, incomplete_ref])
        result = evaluate_retrieval_provenance(
            "See [SOP-OPR-101].",
            [{"citation": "SOP-OPR-101", "source_type": "document"}],
            [tool_out],
        )
        assert result["retrieval_refs_valid"] == 1
        assert result["retrieval_refs_total"] == 2


# ── Composite ────────────────────────────────────────────────────────────────

class TestComposite:
    def test_perfect_grounding(self):
        tool_out = _tool_output([_ref("SOP-OPR-101", "SOP-OPR-101:c0:can")])
        result = evaluate_grounding(
            answer_text="See [SOP-OPR-101] for details.",
            evidence=[{"citation": "SOP-OPR-101", "source_type": "document"}],
            tool_outputs=[tool_out],
            corpus_manifest=CORPUS_MANIFEST,
            question_id="TEST-1",
        )
        assert result.grounding_score == 1.0
        assert result.grounded is True
        assert result.fabricated_citations == []
        assert result.orphan_citations == []

    def test_composite_50_50_weighting(self):
        """citation_validity=1.0, provenance=0.0 → composite=0.5."""
        result = evaluate_grounding(
            answer_text="See [SOP-OPR-101].",
            evidence=[{"citation": "SOP-OPR-101", "source_type": "document"}],
            tool_outputs=[],  # no retrieval output → provenance=0
            corpus_manifest=CORPUS_MANIFEST,
            question_id="TEST-2",
        )
        assert result.citation_validity_score == 1.0
        assert result.provenance_score == 0.0
        assert result.grounding_score == 0.5
        assert result.grounded is False  # 0.5 < 0.70 threshold

    def test_fabricated_and_orphan(self):
        tool_out = _tool_output([_ref("SOP-OPR-101", "SOP-OPR-101:c0:can")])
        result = evaluate_grounding(
            answer_text="See [SOP-OPR-101] and [SOP-OPR-999].",
            evidence=[
                {"citation": "SOP-OPR-101", "source_type": "document"},
                {"citation": "SOP-OPR-999", "source_type": "document"},
            ],
            tool_outputs=[tool_out],
            corpus_manifest=CORPUS_MANIFEST,
            question_id="TEST-3",
        )
        assert "SOP-OPR-999" in result.fabricated_citations
        assert "SOP-OPR-999" in result.orphan_citations
        assert result.citation_validity_score == 0.5
        assert result.provenance_score == 0.5
        assert result.grounding_score == 0.5

    def test_empty_evidence_scores_zero_citation_validity(self):
        result = evaluate_grounding(
            answer_text="No sources cited.",
            evidence=[],
            tool_outputs=[],
            corpus_manifest=CORPUS_MANIFEST,
            question_id="TEST-4",
        )
        # No citations at all → citation_validity=1.0, provenance=1.0
        # (nothing to validate = vacuously true)
        assert result.citation_validity_score == 1.0
        assert result.provenance_score == 1.0
        assert result.grounded is True

    def test_grounding_result_fields(self):
        result = evaluate_grounding(
            answer_text="test",
            evidence=[],
            tool_outputs=[],
            corpus_manifest=set(),
            question_id="FIELD-TEST",
        )
        assert isinstance(result, GroundingResult)
        assert hasattr(result, "question_id")
        assert hasattr(result, "citation_validity_score")
        assert hasattr(result, "fabricated_citations")
        assert hasattr(result, "invalid_chunk_ids")
        assert hasattr(result, "provenance_score")
        assert hasattr(result, "orphan_citations")
        assert hasattr(result, "retrieval_refs_valid")
        assert hasattr(result, "retrieval_refs_total")
        assert hasattr(result, "grounding_score")
        assert hasattr(result, "grounded")

    def test_custom_threshold(self):
        tool_out = _tool_output([_ref("SOP-OPR-101", "SOP-OPR-101:c0:can")])
        result = evaluate_grounding(
            answer_text="See [SOP-OPR-101].",
            evidence=[{"citation": "SOP-OPR-101", "source_type": "document"}],
            tool_outputs=[tool_out],
            corpus_manifest=CORPUS_MANIFEST,
            question_id="THRESH-TEST",
            threshold=0.99,
        )
        assert result.grounding_score == 1.0
        assert result.grounded is True  # 1.0 >= 0.99


# ── Helper function tests ───────────────────────────────────────────────────

class TestHelpers:
    def test_extract_answer_citations_dedup(self):
        """Evidence and inline citations should be deduplicated."""
        citations = _extract_answer_citations(
            "See [SOP-OPR-101].",
            [{"citation": "SOP-OPR-101", "source_type": "document"}],
        )
        assert citations.count("SOP-OPR-101") == 1

    def test_extract_chunk_ids(self):
        tool_out = _tool_output(
            [_ref("SOP-OPR-101", "SOP-OPR-101:c0:can")],
            [{"doc_id": "SOP-OPR-101", "chunk_id": "SOP-OPR-101:c1:can"}],
        )
        chunk_ids = _extract_chunk_ids_from_tool_outputs([tool_out])
        assert "SOP-OPR-101:c0:can" in chunk_ids
        assert "SOP-OPR-101:c1:can" in chunk_ids

    def test_extract_retrieved_doc_ids(self):
        tool_out = _tool_output(
            [_ref("SOP-OPR-101", "SOP-OPR-101:c0:can")],
            [{"doc_id": "ESC-402", "chunk_id": "ESC-402:c0:can"}],
        )
        ids = _extract_retrieved_doc_ids([tool_out])
        assert "SOP-OPR-101" in ids
        assert "ESC-402" in ids

    def test_malformed_json_tool_output_ignored(self):
        ids = _extract_retrieved_doc_ids(["not-json", ""])
        assert ids == set()

    def test_non_dict_json_ignored(self):
        ids = _extract_retrieved_doc_ids([json.dumps([1, 2, 3])])
        assert ids == set()
