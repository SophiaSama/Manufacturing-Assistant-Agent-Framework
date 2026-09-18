"""Generic multi-format document loading, parsing, and metadata extraction.

Supports Markdown (.md), Plain Text (.txt), JSON (.json), CSV (.csv),
and PDF (.pdf, via pypdf when installed).
"""

from __future__ import annotations

import abc
import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from enterprise_agent.config.domain_config import DocumentConfig, RbacConfig

logger = logging.getLogger(__name__)


@dataclass
class RawDocument:
    """Loaded document before chunking."""
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDocumentParser(abc.ABC):
    """Abstract parser for a specific document format."""

    @abc.abstractmethod
    def parse(self, path: Path) -> RawDocument:
        """Parse file into raw text and extracted metadata."""


class TextParser(BaseDocumentParser):
    """Parser for plain text files."""

    def parse(self, path: Path) -> RawDocument:
        content = path.read_text(encoding="utf-8", errors="replace")
        return RawDocument(content=content, metadata={"source": str(path)})


class MarkdownParser(BaseDocumentParser):
    """Parser for Markdown files with optional YAML frontmatter."""

    def parse(self, path: Path) -> RawDocument:
        raw = path.read_text(encoding="utf-8", errors="replace")
        metadata: dict[str, Any] = {"source": str(path)}

        # Simple YAML frontmatter extraction
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) >= 3:
                frontmatter_text = parts[1].strip()
                content = parts[2].strip()
                for line in frontmatter_text.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        metadata[k.strip().lower()] = v.strip().strip("\"'")
                return RawDocument(content=content, metadata=metadata)

        return RawDocument(content=raw, metadata=metadata)


class JsonDocumentParser(BaseDocumentParser):
    """Parser for structured JSON records into indexable text."""

    def parse(self, path: Path) -> RawDocument:
        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON file %s: %e", path, e)
            return RawDocument(content=raw, metadata={"source": str(path)})

        metadata: dict[str, Any] = {"source": str(path)}
        if isinstance(data, dict):
            # Extract top-level metadata if available
            for k in ("id", "title", "category", "version", "doc_id"):
                if k in data:
                    metadata[k] = data[k]
            content = json.dumps(data, indent=2, ensure_ascii=False)
        elif isinstance(data, list):
            lines = [f"Record Count: {len(data)}"]
            for idx, item in enumerate(data):
                lines.append(f"--- Record {idx + 1} ---")
                lines.append(json.dumps(item, indent=2, ensure_ascii=False))
            content = "\n".join(lines)
        else:
            content = str(data)

        return RawDocument(content=content, metadata=metadata)


class CsvDocumentParser(BaseDocumentParser):
    """Parser for CSV documents."""

    def parse(self, path: Path) -> RawDocument:
        lines: list[str] = []
        metadata: dict[str, Any] = {"source": str(path)}

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            metadata["columns"] = headers
            lines.append(f"Table columns: {', '.join(headers)}")
            for idx, row in enumerate(reader):
                row_str = ", ".join(f"{k}: {v}" for k, v in row.items() if v)
                lines.append(f"Row {idx + 1}: {row_str}")

        return RawDocument(content="\n".join(lines), metadata=metadata)


class PdfDocumentParser(BaseDocumentParser):
    """Parser for PDF documents with fallback if pypdf is not installed."""

    def parse(self, path: Path) -> RawDocument:
        metadata: dict[str, Any] = {"source": str(path)}
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            pages_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages_text.append(f"[Page {i + 1}]\n{text}")
            content = "\n\n".join(pages_text)
            metadata["page_count"] = len(reader.pages)
            return RawDocument(content=content, metadata=metadata)
        except ImportError:
            logger.warning(
                "pypdf is not installed. Unable to parse PDF: %s. "
                "Install with 'pip install pypdf'.", path
            )
            return RawDocument(
                content=f"[PDF file: {path.name} (pypdf not installed)]",
                metadata=metadata,
            )


class DocumentLoaderRegistry:
    """Registry mapping file extensions to parser implementations."""

    def __init__(self):
        self._parsers: dict[str, BaseDocumentParser] = {
            ".md": MarkdownParser(),
            ".markdown": MarkdownParser(),
            ".txt": TextParser(),
            ".json": JsonDocumentParser(),
            ".csv": CsvDocumentParser(),
            ".pdf": PdfDocumentParser(),
        }

    def register_parser(self, extension: str, parser: BaseDocumentParser) -> None:
        ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        self._parsers[ext] = parser

    def get_parser(self, path: Path) -> BaseDocumentParser | None:
        return self._parsers.get(path.suffix.lower())


class DocumentIngestionPipeline:
    """End-to-end ingestion pipeline producing chunked LangChain Documents with RBAC metadata."""

    def __init__(
        self,
        doc_config: DocumentConfig,
        rbac_config: RbacConfig,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        registry: DocumentLoaderRegistry | None = None,
    ):
        self.doc_config = doc_config
        self.rbac_config = rbac_config
        self.registry = registry or DocumentLoaderRegistry()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        )

    def load_file(self, file_path: Path) -> list[Document]:
        """Load and chunk a single file, applying RBAC and category metadata."""
        parser = self.registry.get_parser(file_path)
        if not parser:
            logger.debug("Skipping unsupported file extension: %s", file_path)
            return []

        raw_doc = parser.parse(file_path)
        if not raw_doc.content.strip():
            return []

        # Derive metadata
        doc_id = raw_doc.metadata.get("doc_id") or self.doc_config.extract_doc_id(file_path)
        category = raw_doc.metadata.get("category") or self.doc_config.get_category_for_path(file_path)
        level_rank = self.rbac_config.level_from_path(file_path)

        metadata: dict[str, Any] = {
            "source": str(file_path),
            "doc_id": doc_id,
            "category": category,
            "level_rank": level_rank,
            "file_name": file_path.name,
        }
        metadata.update(raw_doc.metadata)

        chunks = self.splitter.split_text(raw_doc.content)
        return [
            Document(
                page_content=chunk,
                metadata={**metadata, "chunk_index": i},
            )
            for i, chunk in enumerate(chunks)
        ]

    def load_directory(self, root_dir: Path) -> list[Document]:
        """Recursively load and chunk all supported documents in a directory."""
        documents: list[Document] = []
        for path in sorted(root_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.name in self.doc_config.excluded_files:
                continue
            if path.suffix.lower() not in self.doc_config.supported_extensions:
                continue

            docs = self.load_file(path)
            documents.extend(docs)

        logger.info("Loaded %d chunks from %s", len(documents), root_dir)
        return documents
