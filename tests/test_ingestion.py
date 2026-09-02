"""Unit tests for corpus-profile ingestion (docs/variant-corpus-ingestion-design.md).

These tests exercise discovery/chunking only — no embedding provider needed.
"""

from __future__ import annotations

from pathlib import Path

from nga.ingestion.build_vector_store import (
    EXCLUDED_FILES,
    build_documents,
    discover_documents,
    variant_diff_map,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANT_DIR = PROJECT_ROOT / "variant-corpus"

# The 8 planted inconsistencies (ledger: variant-corpus/planted-inconsistencies.md)
EXPECTED_CHANGED_FILES = {
    "operator-sops/SOP-OPR-101-wheel-installation-torque.md",
    "operator-sops/SOP-OPR-114-windshield-installation.md",
    "operator-sops/SOP-OPR-127-engine-mount-installation.md",
    "operator-sops/SOP-OPR-142-brake-fluid-fill.md",
    "technician-sops/SOP-TEC-214-torque-tool-calibration-drift.md",
    "machine-details/TEC-301-welding-robot-specification.md",
    "failure-analysis/ESC-402-escalation-workflow-matrix.md",
    "recall-quality/QCR-501-recall-trigger-criteria.md",
}


class TestDiscovery:
    def test_main_profile_canonical_only(self):
        docs = discover_documents(PROJECT_ROOT, "main")
        assert len(docs) == 25
        assert all(source == "canonical" for _, source in docs)

    def test_variant_profile_variant_only(self):
        docs = discover_documents(
            PROJECT_ROOT, "variant", variant_dir=VARIANT_DIR
        )
        assert len(docs) == 16
        assert all(source == "variant" for _, source in docs)

    def test_conflict_profile_canonical_plus_differences(self):
        docs = discover_documents(
            PROJECT_ROOT, "conflict", variant_dir=VARIANT_DIR
        )
        sources = [s for _, s in docs]
        assert len(docs) == 33
        assert sources.count("canonical") == 25
        assert sources.count("variant") == 8

    def test_excluded_files_never_ingested(self):
        for profile in ("main", "variant", "conflict"):
            docs = discover_documents(
                PROJECT_ROOT, profile, variant_dir=VARIANT_DIR
            )
            names = {p.name for p, _ in docs}
            assert EXCLUDED_FILES.isdisjoint(names), profile

    def test_unknown_profile_rejected(self):
        import pytest

        with pytest.raises(ValueError):
            discover_documents(PROJECT_ROOT, "bogus")

    def test_variant_requires_dir(self):
        import pytest

        with pytest.raises(ValueError):
            discover_documents(PROJECT_ROOT, "variant", variant_dir=None)


class TestVariantDiff:
    def test_exactly_8_files_differ(self):
        diff = variant_diff_map(PROJECT_ROOT, VARIANT_DIR)
        changed = {k for k, v in diff.items() if v}
        assert changed == EXPECTED_CHANGED_FILES

    def test_unchanged_copies_marked_false(self):
        diff = variant_diff_map(PROJECT_ROOT, VARIANT_DIR)
        unchanged = [k for k, v in diff.items() if not v]
        assert len(unchanged) == 8  # the remaining copies are byte-identical


class TestBuildDocuments:
    def test_main_documents_metadata(self):
        docs = build_documents(PROJECT_ROOT, "main")
        assert docs
        for d in docs:
            assert d.metadata["corpus_source"] == "canonical"
            assert "chunk_id" in d.metadata
            assert "level_rank" in d.metadata

    def test_conflict_chunk_ids_unique_across_sources(self):
        docs = build_documents(
            PROJECT_ROOT, "conflict", variant_dir=VARIANT_DIR
        )
        chunk_ids = [d.metadata["chunk_id"] for d in docs]
        assert len(chunk_ids) == len(set(chunk_ids)), "chunk_id collision!"

    def test_conflict_metadata_source_tags(self):
        docs = build_documents(
            PROJECT_ROOT, "conflict", variant_dir=VARIANT_DIR
        )
        sources = {d.metadata["corpus_source"] for d in docs}
        assert sources == {"canonical", "variant"}

    def test_conflict_rbac_preserved(self):
        docs = build_documents(
            PROJECT_ROOT, "conflict", variant_dir=VARIANT_DIR
        )
        # OPR docs → operator (level 1); QCR docs → manager (level 4)
        opr = [d for d in docs if d.metadata["category"] == "OPR"]
        qcr = [d for d in docs if d.metadata["category"] == "QCR"]
        assert opr and qcr
        assert all(d.metadata["level_rank"] == 1 for d in opr)
        assert all(d.metadata["level_rank"] == 4 for d in qcr)

    def test_variant_doc_contains_conflicting_value(self):
        docs = build_documents(
            PROJECT_ROOT, "conflict", variant_dir=VARIANT_DIR
        )
        variant_opr101 = [
            d for d in docs
            if d.metadata["doc_id"] == "SOP-OPR-101"
            and d.metadata["corpus_source"] == "variant"
        ]
        assert variant_opr101
        text = "\n".join(d.page_content for d in variant_opr101)
        assert "108 Nm" in text  # planted value present in variant source
