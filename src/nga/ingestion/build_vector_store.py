"""Document ingestion pipeline for the NGA Manufacturing Assistant.

Ingests Markdown documents from the manufacturing_agent corpus into a
ChromaDB persistent vector store with RBAC metadata.

Document categories and access levels are derived server-side from
folder paths — never from caller input.

Corpus profiles (docs/variant-corpus-ingestion-design.md):
  main      — canonical corpus only (default, production)
  variant   — variant corpus only (isolation testing)
  conflict  — canonical + only the variant files that DIFFER by content hash
              (conflict-detection integration tests)

Run: uv run python -m nga.ingestion.build_vector_store [--profile main|variant|conflict]
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

from nga.config import Settings
from nga.providers.factory import make_embeddings
from nga.rag_agent.rbac import ACCESS_LEVELS, level_from_path

logger = logging.getLogger(__name__)

# Corpus profiles → collection names
CORPUS_PROFILES = ("main", "variant", "conflict")
COLLECTION_NAMES: dict[str, str] = {
    "main":     "nga_reference_docs",
    "variant":  "nga_variant_docs",
    "conflict": "nga_conflict_docs",
}
SOURCE_TAG = {"canonical": "c", "variant": "v"}

# Map folder → document category tag
FOLDER_CATEGORY_MAP: dict[str, str] = {
    "operator-sops":                        "OPR",
    "technician-sops":                      "TEC",
    "machine-details":                      "TEC",
    "failure-analysis":                     "FA",
    "recall-quality":                       "QCR",
    "additional-docs/maintenance-work-orders": "WO",
    "additional-docs/supplier-quality":     "SQ",
    "additional-docs/training":             "TR",
}

# Files to exclude from agent retrieval corpus (contain ground truth)
EXCLUDED_FILES = {
    "ground-truth.md",
    "planted-inconsistencies.md",
    "database-readme.md",
    "example-queries.sql",
}

# Known corpus subdirectories (relative to the corpus root)
CORPUS_ROOTS: tuple[str, ...] = (
    "operator-sops",
    "technician-sops",
    "machine-details",
    "failure-analysis",
    "recall-quality",
    "additional-docs",
)


def _is_dimension_mismatch(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        m in text
        for m in ("invaliddimensionexception", "embedding dimension",
                  "dimension mismatch", "dimension does not match")
    )


def _get_category(path: Path) -> str:
    path_str = str(path)
    for folder, category in FOLDER_CATEGORY_MAP.items():
        if folder in path_str:
            return category
    return "OPR"  # safe default


def _get_doc_id(path: Path) -> str:
    """Extract SOP/doc ID from filename (e.g. SOP-OPR-101 from SOP-OPR-101-wheel...)"""
    stem = path.stem
    patterns = [
        r"(SOP-[A-Z]+-\d+)",
        r"(TEC-\d+)",
        r"(FAP-\d+)",
        r"(ESC-\d+)",
        r"(QCR-\d+)",
        r"(WO-\d+-\d+)",
        r"(SCAR-\d+-\d+)",
        r"(AUD-\d+-\d+)",
        r"(APP-\d+)",
        r"(TR-\d+-\d+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, stem, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return stem


def _collect_md(root: Path) -> list[Path]:
    """All *.md files under root, skipping EXCLUDED_FILES (by filename)."""
    if not root.exists():
        return []
    return [
        p for p in sorted(root.rglob("*.md"))
        if p.name not in EXCLUDED_FILES
    ]


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variant_diff_map(base_dir: Path, variant_dir: Path) -> dict[str, bool]:
    """Map relative path → True when the variant file differs from its canonical
    counterpart (or has none). Byte-identical copies are False (excluded in
    conflict profile to avoid retrieval noise).
    """
    result: dict[str, bool] = {}
    for vf in _collect_md(variant_dir):
        rel = vf.relative_to(variant_dir)
        canonical = base_dir / rel
        if not canonical.exists():
            result[str(rel)] = True  # no counterpart → include (fail-open)
            continue
        result[str(rel)] = _file_hash(vf) != _file_hash(canonical)
    return result


def discover_documents(
    base_dir: Path,
    profile: str = "main",
    variant_dir: Path | None = None,
) -> list[tuple[Path, str]]:
    """Discover corpus Markdown files for a profile.

    Returns (path, corpus_source) pairs. `variant_dir` is only required for
    the variant/conflict profiles.
    """
    if profile not in CORPUS_PROFILES:
        raise ValueError(f"Unknown corpus profile: {profile!r}")

    if profile == "main":
        return [
            (p, "canonical")
            for root_name in CORPUS_ROOTS
            for p in _collect_md(base_dir / root_name)
        ]

    if variant_dir is None:
        raise ValueError("variant_dir is required for profile={profile!r}")

    if profile == "variant":
        return [
            (p, "variant")
            for root_name in CORPUS_ROOTS
            for p in _collect_md(variant_dir / root_name)
        ]

    # conflict: canonical + differing variant files
    diff_map = variant_diff_map(base_dir, variant_dir)
    docs: list[tuple[Path, str]] = [
        (p, "canonical")
        for root_name in CORPUS_ROOTS
        for p in _collect_md(base_dir / root_name)
    ]
    for rel, differs in diff_map.items():
        if differs:
            docs.append((variant_dir / rel, "variant"))
    return docs


def build_documents(
    base_dir: Path,
    profile: str = "main",
    variant_dir: Path | None = None,
) -> list[Document]:
    """Chunk corpus Markdown files into langchain Documents with RBAC metadata
    and corpus_source provenance."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    documents: list[Document] = []

    for md_path, source in discover_documents(base_dir, profile, variant_dir):
        text = md_path.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)

        path_str = str(md_path)
        access_level = level_from_path(path_str)
        level_rank = ACCESS_LEVELS[access_level]
        category = _get_category(md_path)
        doc_id = _get_doc_id(md_path)
        tag = SOURCE_TAG[source]

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "doc_id":       doc_id,
                        "doc_name":     md_path.name,
                        "category":     category,
                        "chunk_id":     f"{doc_id}:c{i}:{tag}",
                        "access_level": access_level,
                        "level_rank":   level_rank,
                        "corpus_source": source,
                        "source_path":  path_str,
                    },
                )
            )
        logger.info(
            "Ingested %s → doc_id=%s category=%s level=%s source=%s chunks=%d",
            md_path.name, doc_id, category, access_level, source, len(chunks),
        )

    return documents


def profile_store_dir(settings: Settings, profile: str) -> str:
    """Resolve the Chroma persist directory for a corpus profile."""
    if profile == "variant":
        return settings.variant_vector_store_dir
    if profile == "conflict":
        return settings.conflict_vector_store_dir
    return settings.vector_store_dir


def profile_collection_name(profile: str) -> str:
    return COLLECTION_NAMES[profile]


def open_vector_store(
    settings: Settings,
    embeddings: Any | None = None,
    profile: str | None = None,
) -> Chroma:
    """Open (not build) the Chroma store for a corpus profile.

    Uses the profile's collection name and persist directory so callers
    (CLI, UI server, tests) stay profile-aware without duplicating logic.
    """
    resolved = profile or settings.corpus_profile
    return Chroma(
        collection_name=profile_collection_name(resolved),
        embedding_function=embeddings or make_embeddings(settings),
        persist_directory=profile_store_dir(settings, resolved),
    )


def build_vector_store(
    settings: Settings,
    base_dir: Path | None = None,
    profile: str | None = None,
) -> Chroma:
    """Build or rebuild the ChromaDB vector store for a corpus profile."""
    profile = profile or settings.corpus_profile
    if profile not in CORPUS_PROFILES:
        raise ValueError(f"Unknown corpus profile: {profile!r}")

    if base_dir is None:
        base_dir = Path(settings.documents_dir) if settings.documents_dir else Path(".")

    variant_dir: Path | None = None
    if profile in ("variant", "conflict"):
        variant_dir = base_dir / settings.variant_corpus_dir
        if not variant_dir.exists():
            raise FileNotFoundError(
                f"Variant corpus not found at {variant_dir} for profile={profile!r}"
            )

    documents = build_documents(base_dir, profile=profile, variant_dir=variant_dir)
    if not documents:
        raise RuntimeError(
            f"No documents extracted from {base_dir} for profile={profile!r}. "
            "Check that the corpus Markdown files are present."
        )

    embeddings = make_embeddings(settings)
    store_dir = profile_store_dir(settings, profile)
    collection_name = profile_collection_name(profile)

    def _build() -> Chroma:
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=store_dir,
            collection_name=collection_name,
        )

    try:
        return _build()
    except Exception as exc:
        if not _is_dimension_mismatch(exc):
            raise
        logger.warning(
            "Embedding dimension mismatch at %s — rebuilding.", store_dir
        )
        if Path(store_dir).exists():
            shutil.rmtree(store_dir)
        return _build()


def main() -> None:
    import argparse
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Build NGA Chroma vector store")
    parser.add_argument(
        "--profile",
        choices=CORPUS_PROFILES,
        default=None,
        help="Corpus profile (default: from CORPUS_PROFILE env, 'main')",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    profile = args.profile or settings.corpus_profile
    # Auto-detect corpus root (parent of database/)
    base_dir = Path(settings.nga_db_path).parent.parent
    logger.info("Ingesting NGA corpus (profile=%s) from: %s", profile, base_dir)
    store = build_vector_store(settings, base_dir=base_dir, profile=profile)
    store_dir = profile_store_dir(settings, profile)
    logger.info(
        "Vector store built at %s — %d chunks embedded.",
        store_dir,
        store._collection.count(),
    )


if __name__ == "__main__":
    main()
