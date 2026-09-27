"""Tests for retrieval provenance rendering (corpus_source tags)."""

from __future__ import annotations

from nga.tools.retrieval_tool import (
    build_retrieval_payload,
    format_retrieval_results,
)


def _result(source: str) -> dict:
    return {
        "category": "QCR",
        "doc_id": "QCR-501",
        "doc_name": "QCR-501-recall-trigger-criteria.md",
        "section": "3",
        "chunk_id": "QCR-501:c0",
        "text": "Recall threshold text",
        "access_level": "manager",
        "level_rank": 4,
        "corpus_source": source,
    }


class TestFormatting:
    def test_variant_tag_rendered(self):
        block = format_retrieval_results([_result("variant")])
        assert "[VARIANT]" in block
        assert "[QCR-501]" in block

    def test_no_tag_for_canonical(self):
        block = format_retrieval_results([_result("canonical")])
        assert "[VARIANT]" not in block

    def test_mixed_sources_tagged_individually(self):
        block = format_retrieval_results(
            [_result("canonical"), _result("variant")]
        )
        # only one VARIANT tag, but both citations present
        assert block.count("[VARIANT]") == 1
        assert block.count("[QCR-501]") == 2

    def test_payload_references_include_source(self):
        payload = build_retrieval_payload(
            "q", ["QCR"], [_result("variant")]
        )
        refs = payload["references"]
        assert refs[0]["corpus_source"] == "variant"

    def test_empty_results(self):
        block = format_retrieval_results([])
        assert "no relevant document passages" in block
