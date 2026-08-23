"""Document ingestion pipeline for the NGA Manufacturing Assistant.

Ingests all Markdown documents from the manufacturing_agent corpus into
a ChromaDB persistent vector store with RBAC metadata.

Document categories and access levels are derived server-side from
folder paths — never from caller input.

Run: uv run python -m nga.ingestion.build_vector_store
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

from nga.config import Settings
from nga.providers.factory import make_embeddings
from nga.rag_agent.rbac import ACCESS_LEVELS, level_from_path

logger = logging.getLogger(__name__)

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
    # Try to match known doc ID patterns
    import re
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


def discover_documents(base_dir: Path) -> list[Path]:
    """Discover NGA corpus Markdown documents from known corpus subdirectories only."""
    # Restrict to known NGA corpus folders to avoid picking up venv/cache .md files
    corpus_roots = [
        "operator-sops",
        "technician-sops",
        "machine-details",
        "failure-analysis",
        "recall-quality",
        "additional-docs",
    ]
    all_md: list[Path] = []
    for root_name in corpus_roots:
        root = base_dir / root_name
        if not root.exists():
            logger.debug("Corpus directory not found, skipping: %s", root)
            continue
        for md_file in root.rglob("*.md"):
            if md_file.name in EXCLUDED_FILES:
                continue
            all_md.append(md_file)
    return sorted(all_md)


def build_documents(base_dir: Path) -> list[Document]:
    """Chunk all corpus Markdown files into langchain Documents with RBAC metadata."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    documents: list[Document] = []

    for md_path in discover_documents(base_dir):
        text = md_path.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)

        path_str = str(md_path)
        access_level = level_from_path(path_str)
        level_rank = ACCESS_LEVELS[access_level]
        category = _get_category(md_path)
        doc_id = _get_doc_id(md_path)

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
                        "chunk_id":     f"{doc_id}:c{i}",
                        "access_level": access_level,
                        "level_rank":   level_rank,
                        "source_path":  path_str,
                    },
                )
            )
        logger.info(
            "Ingested %s → doc_id=%s category=%s level=%s chunks=%d",
            md_path.name, doc_id, category, access_level, len(chunks),
        )

    return documents


def build_vector_store(settings: Settings, base_dir: Path | None = None) -> Chroma:
    """Build or rebuild the ChromaDB vector store for the NGA corpus."""
    if base_dir is None:
        base_dir = Path(settings.documents_dir) if settings.documents_dir else Path(".")

    documents = build_documents(base_dir)
    if not documents:
        raise RuntimeError(
            f"No documents extracted from {base_dir}. "
            "Check that the corpus Markdown files are present."
        )

    embeddings = make_embeddings(settings)

    def _build() -> Chroma:
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=settings.vector_store_dir,
            collection_name="nga_reference_docs",
        )

    try:
        return _build()
    except Exception as exc:
        if not _is_dimension_mismatch(exc):
            raise
        store_dir = Path(settings.vector_store_dir)
        logger.warning(
            "Embedding dimension mismatch at %s — rebuilding.", store_dir
        )
        if store_dir.exists():
            shutil.rmtree(store_dir)
        return _build()


def main() -> None:
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    settings = Settings.from_env()
    # Auto-detect corpus root (parent of database/)
    base_dir = Path(settings.nga_db_path).parent.parent
    logger.info("Ingesting NGA corpus from: %s", base_dir)
    store = build_vector_store(settings, base_dir=base_dir)
    logger.info(
        "Vector store built at %s — %d chunks embedded.",
        settings.vector_store_dir,
        store._collection.count(),
    )


if __name__ == "__main__":
    main()
