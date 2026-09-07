# Workstream A — Anchored Ingestion v2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ingestion with content-addressed, header-aware chunking carrying the full anchored-metadata contract, and publish each build as a versioned registry generation.

**Architecture:** Three new pure modules (`metadata.py` = contract, `chunking.py` = Chunker seam, `registry.py` = IndexRegistry) plus integration into `build_vector_store.py` (anchored chunks, generation publication), config knobs, and a cache corpus-version preference for the registry manifest. Retrieval formatting now renders real `section_path` values, fixing the dead `section` metadata bug. Pure functions first, integration second; every task keeps `uv run pytest` and `uv run ruff check src/nga tests` green.

**Tech Stack:** Python 3.11, langchain `Document`/`RecursiveCharacterTextSplitter`, ChromaDB, `hashlib`/`shutil`/`os.replace` (registry), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-09-07-graph-rag-reference-architecture-design.md` — this plan implements §4 (contracts) and §5 (Workstream A) only. Subsequent workstreams (B–E) get their own plans after review.

## Global Constraints

(From the spec; every task's requirements implicitly include these.)

- Canonical metadata keys (spec §4.1): `doc_id, doc_name, category, source_path, corpus_source, access_level, level_rank, chunk_id, section_path, section_title, headings, chunk_order, doc_hash, ingested_at, doc_version, embedding_model, splitter, token_count`. `level_rank`/`access_level` are ALWAYS derived server-side from `source_path` — never from caller input.
- Chunk identity (spec §4.2): `chunk_id = f"{doc_id}:{source_tag}:{section_key}:{chunk_order:03d}:{content_h8}"` where `source_tag ∈ {"c","v"}`, `section_key = slug(section_path)[:48]` (`"." → "root"`), `content_h8 = sha256(chunk_text)[:8]`. `make_chunk_id` raises `ValueError` on empty `text` or bad `source_tag`.
- Registry rules (spec §4.4): builders write into a fresh `generations/<g>/` dir, then flip `current.json` by atomic rename (temp file + `os.replace`). A served generation is never mutated in place. `corpus_version` is content-derived (`doc_hash` + `chunk_ids` per file). `config_hash` = sha256 over `{chunker, embedding_model, metadata_schema_version}`.
- Cache: keys scope by `corpus_version`; once a committed registry manifest exists for the main profile, `get_corpus_version()` returns the manifest's `corpus_version` (mtime fallback otherwise) — no cache sweep needed.
- Defaults: chunker `header`, `chunk_size=800`, `chunk_overlap=120`, `metadata_schema_version=2`, registry root `data/index_registry` (env `INDEX_REGISTRY_DIR`, `NGA_CHUNKER`, `NGA_CHUNK_SIZE`, `NGA_CHUNK_OVERLAP`, `NGA_METADATA_SCHEMA_VERSION`).
- RBAC invariant (spec §4.3): an unreadable unit is never included — failure mode is deny.
- Repo hygiene: line length 88, ruff select `E,F,I`; run `uv run ruff check src/nga tests` after each task.

---

### Task 1: Metadata contract module

**Files:**
- Create: `src/nga/ingestion/metadata.py`
- Test: `tests/test_metadata.py`

**Interfaces:**
- Consumes: nothing (stdlib only).
- Produces: `CHUNK_META_KEYS: tuple[str, ...]`, `SOURCE_TAGS = ("c", "v")`, `doc_hash(data: bytes) -> str`, `section_key(section_path: str) -> str`, `estimate_tokens(text: str) -> int`, `make_chunk_id(*, doc_id: str, source_tag: str, section_path: str, chunk_order: int, text: str) -> str`, `utc_now_iso() -> str`. Later tasks (2, 4) import these.

- [ ] **Step 1: Write the failing test**

Create `tests/test_metadata.py`:

```python
"""Unit tests for the canonical chunk-metadata contract (spec §4.1–4.2)."""

from __future__ import annotations

import pytest

from nga.ingestion.metadata import (
    CHUNK_META_KEYS,
    SOURCE_TAGS,
    doc_hash,
    estimate_tokens,
    make_chunk_id,
    section_key,
)


class TestDocHash:
    def test_stable_and_hex(self):
        h = doc_hash(b"SOP text")
        assert h == doc_hash(b"SOP text")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_content_sensitive(self):
        assert doc_hash(b"a") != doc_hash(b"b")


class TestSectionKey:
    def test_root_for_empty_and_dot(self):
        assert section_key(".") == "root"
        assert section_key("") == "root"

    def test_slugifies_and_truncates(self):
        assert section_key("Standard Operating Procedure/1. Purpose") == (
            "standard-operating-procedure-1-purpose"
        )
        long = "x/" * 60
        assert len(section_key(long)) <= 48


class TestEstimateTokens:
    def test_ceil_chars_over_four(self):
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("abcde") == 2


class TestMakeChunkId:
    def test_format(self):
        cid = make_chunk_id(
            doc_id="SOP-OPR-101",
            source_tag="c",
            section_path="Standard Operating Procedure/1. Purpose",
            chunk_order=0,
            text="Purpose text here",
        )
        assert cid.startswith("SOP-OPR-101:c:standard-operating-procedure-1-purpose:000:")
        assert len(cid.split(":")[-1]) == 8

    def test_deterministic(self):
        kwargs = dict(
            doc_id="SOP-OPR-101", source_tag="c",
            section_path="1. Purpose", chunk_order=3, text="same text",
        )
        assert make_chunk_id(**kwargs) == make_chunk_id(**kwargs)

    def test_text_change_changes_hash_part_only(self):
        a = make_chunk_id(doc_id="D", source_tag="c", section_path="1",
                          chunk_order=0, text="alpha")
        b = make_chunk_id(doc_id="D", source_tag="c", section_path="1",
                          chunk_order=0, text="beta")
        assert a.split(":")[-1] != b.split(":")[-1]
        assert a.rsplit(":", 1)[0] == b.rsplit(":", 1)[0]

    def test_source_tags_disambiguate(self):
        a = make_chunk_id(doc_id="SOP-OPR-101", source_tag="c",
                          section_path="1. Purpose", chunk_order=0, text="same")
        b = make_chunk_id(doc_id="SOP-OPR-101", source_tag="v",
                          section_path="1. Purpose", chunk_order=0, text="same")
        assert a != b

    def test_rejects_empty_text(self):
        with pytest.raises(ValueError):
            make_chunk_id(doc_id="D", source_tag="c", section_path="1",
                          chunk_order=0, text="   ")

    def test_rejects_bad_source_tag(self):
        with pytest.raises(ValueError):
            make_chunk_id(doc_id="D", source_tag="x", section_path="1",
                          chunk_order=0, text="hello")

    def test_meta_key_set_is_complete(self):
        expected = {
            "doc_id", "doc_name", "category", "source_path", "corpus_source",
            "access_level", "level_rank", "chunk_id", "section_path",
            "section_title", "headings", "chunk_order", "doc_hash",
            "ingested_at", "doc_version", "embedding_model", "splitter",
            "token_count",
        }
        assert set(CHUNK_META_KEYS) == expected
        assert SOURCE_TAGS == ("c", "v")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metadata.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nga.ingestion.metadata'`

- [ ] **Step 3: Write minimal implementation**

Create `src/nga/ingestion/metadata.py`:

```python
"""Canonical chunk-metadata contract for anchored ingestion (spec §4.1–4.2).

Pure helpers: content hashing, deterministic chunk identity, token estimate.
No I/O beyond bytes hashing — safe to import from tests and cache layers.
"""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime, timezone

# Canonical metadata keys every chunk carries (spec §4.1).
CHUNK_META_KEYS: tuple[str, ...] = (
    "doc_id", "doc_name", "category", "source_path", "corpus_source",
    "access_level", "level_rank", "chunk_id", "section_path", "section_title",
    "headings", "chunk_order", "doc_hash", "ingested_at", "doc_version",
    "embedding_model", "splitter", "token_count",
)

# Corpus source discriminator embedded in every chunk_id (spec §4.2).
SOURCE_TAGS: tuple[str, ...] = ("c", "v")


def doc_hash(data: bytes) -> str:
    """sha256 hex of raw file bytes — content etag (spec §4.1 `doc_hash`)."""
    return hashlib.sha256(data).hexdigest()


def section_key(section_path: str) -> str:
    """Deterministic slug for a section path; '.'/'' → 'root' (spec §4.2)."""
    if not section_path or section_path == ".":
        return "root"
    slug = re.sub(r"[^a-z0-9]+", "-", section_path.lower()).strip("-")
    return slug[:48] or "root"


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: ceil(chars / 4)."""
    return math.ceil(len(text) / 4)


def make_chunk_id(
    *,
    doc_id: str,
    source_tag: str,
    section_path: str,
    chunk_order: int,
    text: str,
) -> str:
    """Content-addressed chunk identity (spec §4.2).

    Same file content + splitter + section structure ⇒ identical id across
    rebuilds. Changed text ⇒ changed trailing content hash.
    """
    if not text.strip():
        raise ValueError("chunk text must be non-empty")
    if source_tag not in SOURCE_TAGS:
        raise ValueError(f"source_tag must be one of {SOURCE_TAGS}, got: {source_tag!r}")
    content_h8 = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return (
        f"{doc_id}:{source_tag}:{section_key(section_path)}:"
        f"{chunk_order:03d}:{content_h8}"
    )


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp for `ingested_at`."""
    return datetime.now(timezone.utc).isoformat()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_metadata.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check src/nga/ingestion/metadata.py tests/test_metadata.py
git add src/nga/ingestion/metadata.py tests/test_metadata.py
git commit -m "feat: canonical chunk metadata contract and content-addressed chunk ids"
```

---

### Task 2: Chunker seam + header-aware markdown splitter

**Files:**
- Create: `src/nga/ingestion/chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Consumes: nothing (own dataclasses; no import from metadata needed yet).
- Produces: `Chunk` (frozen dataclass: `text`, `section_path`, `section_title`, `headings: tuple[str, ...]`, `chunk_order`), `DocumentChunker` (Protocol: `split(text) -> list[Chunk]`), `HeaderAwareMarkdownChunker(chunk_size=800, chunk_overlap=120)` with attribute `kind = "header"`, `RecursiveFallbackChunker(chunk_size=800, chunk_overlap=120)` with `kind = "recursive"`, `make_chunker(*, kind=None, chunk_size=800, chunk_overlap=120) -> DocumentChunker`, constants `DEFAULT_CHUNK_SIZE = 800`, `DEFAULT_CHUNK_OVERLAP = 120`. Task 4 consumes these.

- [ ] **Step 1: Write the failing test**

Create `tests/test_chunking.py`:

```python
"""Unit tests for the header-aware markdown chunker (spec §5)."""

from __future__ import annotations

from nga.ingestion.chunking import (
    HeaderAwareMarkdownChunker,
    make_chunker,
)

DOC = """# Standard Operating Procedure

Intro paragraph that belongs to the document root section.

## 1. Purpose

The purpose of this SOP is wheel installation.

## 2. Scope

This procedure applies to Station 144 line operators.

## 3. Required Equipment

TQ-6012 torque wrench.

## 4. Procedure

### 4.1 Pre-Operation Checks

Verify calibration sticker is current.

### 4.2 Wheel Installation

Fit wheel over studs.

## 5. Quality Requirements

Torque must be 105 Nm +- 5 percent.
"""


class TestHeaderAwareMarkdownChunker:
    def test_no_chunk_crosses_section_boundary(self):
        chunks = HeaderAwareMarkdownChunker().split(DOC)
        # Tokens unique to two different '##' sections must never co-occur.
        pairs = [("wheel installation", "Station 144"), ("wheel installation", "105 Nm")]
        for c in chunks:
            for a, b in pairs:
                assert not (a in c.text and b in c.text), (
                    f"chunk crosses section boundary: {c.text!r}"
                )

    def test_headings_chain_recorded(self):
        chunks = HeaderAwareMarkdownChunker().split(DOC)
        purpose = next(c for c in chunks if "wheel installation" in c.text)
        assert purpose.section_path == (
            "Standard Operating Procedure/1. Purpose"
        )
        assert purpose.section_title == "1. Purpose"
        assert purpose.headings == (
            "Standard Operating Procedure", "1. Purpose",
        )

    def test_subsection_chain(self):
        chunks = HeaderAwareMarkdownChunker().split(DOC)
        checks = next(c for c in chunks if "Verify calibration sticker" in c.text)
        assert checks.section_path == (
            "Standard Operating Procedure/4. Procedure/4.1 Pre-Operation Checks"
        )
        assert checks.headings[-1] == "4.1 Pre-Operation Checks"

    def test_root_content_before_first_heading(self):
        chunks = HeaderAwareMarkdownChunker().split(DOC)
        root = next(c for c in chunks if "Intro paragraph" in c.text)
        assert root.section_path == "."

    def test_deterministic_boundaries(self):
        a = HeaderAwareMarkdownChunker().split(DOC)
        b = HeaderAwareMarkdownChunker().split(DOC)
        assert [(c.text, c.section_path, c.chunk_order) for c in a] == [
            (c.text, c.section_path, c.chunk_order) for c in b
        ]

    def test_long_paragraph_hard_split_with_overlap(self):
        para = "word " * 400  # ~2000 chars > chunk_size
        text = f"## Only Section\n\n{para}\n"
        chunks = HeaderAwareMarkdownChunker(chunk_size=400, chunk_overlap=50).split(text)
        assert len(chunks) >= 4
        assert all(c.section_path == "Only Section" for c in chunks)
        # overlap: tail of chunk N appears as a prefix inside chunk N+1
        assert chunks[0].text[-50:] in chunks[1].text

    def test_empty_text(self):
        assert HeaderAwareMarkdownChunker().split("") == []

    def test_no_headings_document_is_root(self):
        chunks = HeaderAwareMarkdownChunker().split("just prose\n\nmore prose\n")
        assert len(chunks) == 1
        assert chunks[0].section_path == "."

    def test_make_chunker_defaults_and_validation(self):
        import pytest

        assert isinstance(make_chunker(), HeaderAwareMarkdownChunker)
        assert make_chunker(kind="recursive").kind == "recursive"
        assert make_chunker(kind="header").kind == "header"
        with pytest.raises(ValueError):
            make_chunker(kind="bogus")

    def test_heading_only_section_produces_chunk(self):
        chunks = HeaderAwareMarkdownChunker().split("## 9. Revision History\n")
        assert chunks and chunks[0].section_path.endswith("9. Revision History")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nga.ingestion.chunking'`

- [ ] **Step 3: Write minimal implementation**

Create `src/nga/ingestion/chunking.py`:

```python
"""Chunker seam + reference implementations (spec §5).

Header-aware splitting: chunks never cross a markdown heading boundary, each
chunk records its heading chain, and chunk boundaries are a pure function of
the text + parameters (deterministic across rebuilds).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*$")


@dataclass(frozen=True)
class Chunk:
    text: str
    section_path: str
    section_title: str
    headings: tuple[str, ...]
    chunk_order: int


class DocumentChunker(Protocol):
    kind: str

    def split(self, text: str) -> list[Chunk]: ...


def _split_paragraphs(lines: list[str]) -> list[str]:
    """Group non-heading lines into paragraphs on blank-line boundaries."""
    paras: list[str] = []
    current: list[str] = []
    for ln in lines:
        if ln.strip():
            current.append(ln)
        elif current:
            paras.append("\n".join(current))
            current = []
    if current:
        paras.append("\n".join(current))
    return paras


class HeaderAwareMarkdownChunker:
    """Reference chunker: markdown heading-aware, paragraph-assembled."""

    kind = "header"

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if chunk_size < 1 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("need 1 <= chunk_size and 0 <= chunk_overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[Chunk]:
        if not text:
            return []
        lines = text.splitlines()
        sections: list[tuple[tuple[str, ...], list[str]]] = []
        stack: list[tuple[int, str]] = []  # (heading level, title)
        current_body: list[str] = []

        def flush() -> None:
            if current_body or not sections:
                sections.append((tuple(t for _, t in stack), current_body[:]))
            current_body.clear()

        for ln in lines:
            m = _HEADING_RE.match(ln)
            if m:
                flush()
                level = len(m.group(1))
                title = m.group(2).strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                current_body.append(ln)  # heading text belongs to its section
            else:
                current_body.append(ln)
        flush()

        chunks: list[Chunk] = []
        order = 0
        for headings, body in sections:
            pieces: list[str] = []
            for para in _split_paragraphs(body):
                pieces.extend(_hard_split(para, self.chunk_size, self.chunk_overlap))
            current = ""
            for piece in pieces:
                if not current:
                    current = piece
                    continue
                if len(current) + 2 + len(piece) > self.chunk_size:
                    chunks.append(_make_chunk(headings, current, order))
                    order += 1
                    margin = (
                        current[-self.chunk_overlap :] if self.chunk_overlap else ""
                    )
                    current = f"{margin}\n\n{piece}" if margin else piece
                else:
                    current = f"{current}\n\n{piece}"
            if current:
                chunks.append(_make_chunk(headings, current, order))
                order += 1
        return chunks


def _hard_split(paragraph: str, chunk_size: int, overlap: int) -> list[str]:
    """Split an oversized paragraph at char boundaries with overlap margin."""
    if len(paragraph) <= chunk_size:
        return [paragraph]
    out: list[str] = []
    n = len(paragraph)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        out.append(paragraph[start:end])
        if end >= n:
            break
        start = max(end - overlap, end - 1)  # always make progress
    return out


def _make_chunk(headings: tuple[str, ...], text: str, order: int) -> Chunk:
    section_path = "/".join(headings) if headings else "."
    return Chunk(
        text=text,
        section_path=section_path,
        section_title=headings[-1] if headings else "",
        headings=headings,
        chunk_order=order,
    )


class RecursiveFallbackChunker:
    """Fallback chunker wrapping RecursiveCharacterTextSplitter."""

    kind = "recursive"

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[Chunk]:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        return [
            _make_chunk((), part, i)
            for i, part in enumerate(splitter.split_text(text))
            if part.strip()
        ]


def make_chunker(
    *,
    kind: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> DocumentChunker:
    """Factory: None/'header' → HeaderAwareMarkdownChunker (default)."""
    if kind in (None, "header"):
        return HeaderAwareMarkdownChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if kind == "recursive":
        return RecursiveFallbackChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raise ValueError(f"unknown chunker kind: {kind!r} (expected 'header' or 'recursive')")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: PASS (11 tests). If `test_long_paragraph_hard_split_with_overlap` fails on the margin assertion, confirm the overlap tail logic in `split` (margin must come from `current`, the flushed chunk text) rather than weakening the test.

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check src/nga/ingestion/chunking.py tests/test_chunking.py
git add src/nga/ingestion/chunking.py tests/test_chunking.py
git commit -m "feat: header-aware markdown chunker seam with deterministic boundaries"
```

---

### Task 3: IndexRegistry (generations, manifest, atomic pointer)

**Files:**
- Create: `src/nga/ingestion/registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: nothing (stdlib + `pathlib` only).
- Produces (used by Task 4/5 and later workstreams):
  - `registry_root(settings_or_root)` — callers pass a `Path` root explicitly.
  - `generation_dir(root: Path, generation: str) -> Path`
  - `manifest_path(root: Path, generation: str) -> Path`
  - `next_generation(root: Path) -> str` — `"g<int>"` = max existing + 1, else `"g1"`.
  - `write_manifest(root, generation, *, files: list[dict], config: dict, parent_generation: str | None = None, changelog: list[str] | None = None, built_at: str | None = None, status: str = "committed") -> dict` — computes and stores `corpus_version` (content-derived) and writes the JSON manifest.
  - `read_manifest(root, generation) -> dict | None`
  - `read_current(root) -> dict | None` — follows `current.json`, returns the manifest or `None`.
  - `current_generation_id(root) -> str | None`
  - `publish_generation(root, generation, *, source_dir: Path | str, files: list[dict], config: dict, changelog: list[str] | None = None) -> dict` — copies `source_dir` tree into `generations/<g>/vector_store`, writes the manifest, then atomically flips `current.json` (temp file + `os.replace`). Returns the manifest.
  - `compute_corpus_version(files: list[dict]) -> str` — `sha256` over sorted `(doc_id, corpus_source, doc_hash, ','.join(chunk_ids))` lines; prefix `"cv-"`, 12 hex chars.
  - `build_config_hash(*, chunker: dict, embedding_model: str, metadata_schema_version: int) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_registry.py`:

```python
"""Unit tests for the IndexRegistry (spec §4.4)."""

from __future__ import annotations

import json

from nga.ingestion.registry import (
    build_config_hash,
    compute_corpus_version,
    current_generation_id,
    generation_dir,
    next_generation,
    publish_generation,
    read_current,
    read_manifest,
    write_manifest,
)

FILES = [
    {
        "doc_id": "SOP-OPR-101", "path": "operator-sops/SOP-OPR-101.md",
        "doc_hash": "a" * 64, "chunk_ids": ["SOP-OPR-101:c:1-purpose:000:abc12345"],
        "category": "OPR", "level_rank": 1, "corpus_source": "canonical",
    },
    {
        "doc_id": "QCR-501", "path": "recall-quality/QCR-501.md",
        "doc_hash": "b" * 64, "chunk_ids": ["QCR-501:c:3-triggers:000:def67890"],
        "category": "QCR", "level_rank": 4, "corpus_source": "canonical",
    },
]

CONFIG = {"chunker": {"type": "header", "chunk_size": 800},
          "embedding": {"model": "qwen/qwen3-embedding-4b", "dim": 1024},
          "metadata_schema_version": 2}


class TestGenerations:
    def test_next_generation_monotonic(self, tmp_path):
        assert next_generation(tmp_path) == "g1"
        (generation_dir(tmp_path, "g1")).mkdir(parents=True)
        (generation_dir(tmp_path, "g3")).mkdir(parents=True)
        assert next_generation(tmp_path) == "g4"


class TestManifestRoundTrip:
    def test_write_and_read(self, tmp_path):
        manifest = write_manifest(
            tmp_path, "g1", files=FILES, config=CONFIG,
            parent_generation=None, changelog=["initial build"],
        )
        assert manifest["generation"] == "g1"
        assert manifest["status"] == "committed"
        assert manifest["corpus_version"].startswith("cv-")
        assert read_manifest(tmp_path, "g1") == manifest
        assert read_manifest(tmp_path, "g99") is None


class TestCorpusVersion:
    def test_content_derived_and_stable(self):
        v1 = compute_corpus_version(FILES)
        v2 = compute_corpus_version(FILES)
        assert v1 == v2
        assert v1.startswith("cv-")

    def test_changes_when_doc_hash_changes(self):
        changed = [dict(FILES[0], doc_hash="c" * 64), FILES[1]]
        assert compute_corpus_version(changed) != compute_corpus_version(FILES)

    def test_changes_when_chunk_ids_change(self):
        changed = [dict(FILES[0], chunk_ids=["NEW-ID"]), FILES[1]]
        assert compute_corpus_version(changed) != compute_corpus_version(FILES)

    def test_order_independent(self):
        assert compute_corpus_version(list(reversed(FILES))) == compute_corpus_version(FILES)


class TestConfigHash:
    def test_stable_and_sensitive(self):
        a = build_config_hash(chunker={"type": "header"}, embedding_model="m",
                              metadata_schema_version=2)
        b = build_config_hash(chunker={"type": "header"}, embedding_model="m",
                              metadata_schema_version=2)
        assert a == b
        c = build_config_hash(chunker={"type": "recursive"}, embedding_model="m",
                              metadata_schema_version=2)
        assert c != a


class TestPublish:
    def test_publish_copies_and_flips_pointer(self, tmp_path):
        src = tmp_path / "chroma_src"
        (src / "sub").mkdir(parents=True)
        (src / "data.sqlite3").write_text("payload")
        manifest = publish_generation(
            tmp_path, "g1", source_dir=src, files=FILES, config=CONFIG,
            changelog=["initial build"],
        )
        dst = generation_dir(tmp_path, "g1") / "vector_store" / "data.sqlite3"
        assert dst.read_text() == "payload"
        assert current_generation_id(tmp_path) == "g1"
        assert read_current(tmp_path)["generation"] == "g1"

    def test_pointer_is_valid_json_and_atomic_target(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "f").write_text("x")
        publish_generation(tmp_path, "g1", source_dir=src, files=FILES, config=CONFIG)
        pointer = tmp_path / "current.json"
        assert json.loads(pointer.read_text()) == {"generation": "g1"}

    def test_no_registry_returns_none(self, tmp_path):
        assert current_generation_id(tmp_path) is None
        assert read_current(tmp_path) is None

    def test_publish_then_next_generation(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "f").write_text("x")
        publish_generation(tmp_path, "g1", source_dir=src, files=FILES, config=CONFIG)
        assert next_generation(tmp_path) == "g2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nga.ingestion.registry'`

- [ ] **Step 3: Write minimal implementation**

Create `src/nga/ingestion/registry.py`:

```python
"""IndexRegistry — versioned index generations with atomic switching (spec §4.4).

Layout under the registry root:

    current.json                 {"generation": "g7"}   (atomic pointer)
    generations/<g>/manifest.json
    generations/<g>/vector_store/...    (copy-on-write Chroma persist dir)

Builders write a fresh generation dir, then flip current.json by atomic
rename. A served generation is never mutated in place; rollback = repoint.
Storage seam: swap this module's Path plumbing for object-store/catalog in
a real deployment without changing callers.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

GENERATION_PREFIX = "g"


def generation_dir(root: Path | str, generation: str) -> Path:
    return Path(root) / "generations" / generation


def manifest_path(root: Path | str, generation: str) -> Path:
    return generation_dir(root, generation) / "manifest.json"


def _pointer_path(root: Path | str) -> Path:
    return Path(root) / "current.json"


def _read_pointer(root: Path | str) -> dict | None:
    p = _pointer_path(root)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def next_generation(root: Path | str) -> str:
    """Next generation id: max existing numeric suffix + 1, else g1."""
    gens_dir = Path(root) / "generations"
    highest = 0
    if gens_dir.exists():
        for child in gens_dir.iterdir():
            name = child.name
            if name.startswith(GENERATION_PREFIX) and name[len(GENERATION_PREFIX):].isdigit():
                highest = max(highest, int(name[len(GENERATION_PREFIX):]))
    return f"{GENERATION_PREFIX}{highest + 1}"


def compute_corpus_version(files: list[dict]) -> str:
    """Content-derived corpus version over (doc_id, source, hash, chunk_ids)."""
    hasher = hashlib.sha256()
    for f in sorted(files, key=lambda x: (x["doc_id"], x["corpus_source"])):
        line = (
            f"{f['doc_id']}|{f['corpus_source']}|{f['doc_hash']}|"
            f"{','.join(f['chunk_ids'])}|"
        )
        hasher.update(line.encode("utf-8"))
    return f"cv-{hasher.hexdigest()[:12]}"


def build_config_hash(
    *,
    chunker: dict,
    embedding_model: str,
    metadata_schema_version: int,
) -> str:
    """Reproducibility hash over global index semantics (spec §4.4)."""
    payload = {
        "chunker": chunker,
        "embedding_model": embedding_model,
        "metadata_schema_version": metadata_schema_version,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_manifest(
    root: Path | str,
    generation: str,
    *,
    files: list[dict],
    config: dict,
    parent_generation: str | None = None,
    changelog: list[str] | None = None,
    built_at: str | None = None,
    status: str = "committed",
) -> dict:
    """Write generations/<g>/manifest.json; returns the manifest dict."""
    from nga.ingestion.metadata import utc_now_iso

    manifest = {
        "generation": generation,
        "parent_generation": parent_generation,
        "corpus_version": compute_corpus_version(files),
        "built_at": built_at or utc_now_iso(),
        "config_hash": build_config_hash(
            chunker=config.get("chunker", {}),
            embedding_model=config.get("embedding", {}).get("model", ""),
            metadata_schema_version=config.get("metadata_schema_version", 1),
        ),
        "metadata_schema_version": config.get("metadata_schema_version", 1),
        "chunker": config.get("chunker", {}),
        "embedding": config.get("embedding", {}),
        "files": files,
        "changelog": changelog or [],
        "status": status,
    }
    dst = manifest_path(root, generation)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def read_manifest(root: Path | str, generation: str) -> dict | None:
    p = manifest_path(root, generation)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def current_generation_id(root: Path | str) -> str | None:
    pointer = _read_pointer(root)
    if not pointer:
        return None
    gen = pointer.get("generation")
    return gen if isinstance(gen, str) else None


def read_current(root: Path | str) -> dict | None:
    gen = current_generation_id(root)
    return read_manifest(root, gen) if gen else None


def publish_generation(
    root: Path | str,
    generation: str,
    *,
    source_dir: Path | str,
    files: list[dict],
    config: dict,
    changelog: list[str] | None = None,
    parent_generation: str | None = None,
) -> dict:
    """Snapshot source_dir → generations/<g>/vector_store, write manifest,
    then atomically flip current.json. Returns the published manifest."""
    dst = generation_dir(root, generation)
    vector_dst = dst / "vector_store"
    if vector_dst.exists():
        shutil.rmtree(vector_dst)
    shutil.copytree(Path(source_dir), vector_dst)
    manifest = write_manifest(
        root, generation,
        files=files, config=config,
        parent_generation=parent_generation or current_generation_id(root),
        changelog=changelog,
    )
    pointer = _pointer_path(root)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    tmp = pointer.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"generation": generation}, sort_keys=True), encoding="utf-8"
    )
    os.replace(tmp, pointer)
    return manifest
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check src/nga/ingestion/registry.py tests/test_registry.py
git add src/nga/ingestion/registry.py tests/test_registry.py
git commit -m "feat: IndexRegistry generations with atomic pointer and content-derived corpus version"
```

---

### Task 4: build_vector_store adopts anchored chunking + full metadata contract

**Files:**
- Modify: `src/nga/ingestion/build_vector_store.py`
- Modify: `src/nga/tools/retrieval_tool.py` (section_path fix)
- Modify: `tests/test_ingestion.py` (extend, keep all existing assertions)
- Test: `tests/test_ingestion.py`, `tests/test_retrieval_formatting.py`

**Interfaces:**
- Consumes: `metadata.make_chunk_id`, `metadata.doc_hash`, `metadata.utc_now_iso`, `metadata.estimate_tokens`, `chunking.make_chunker`, `chunking.DEFAULT_CHUNK_SIZE`, `chunking.DEFAULT_CHUNK_OVERLAP`.
- Produces (Task 5 uses these):
  - `build_documents(base_dir, profile="main", variant_dir=None, *, chunker=None, doc_version="g0", embedding_model="unknown", now=None) -> list[Document]` — every chunk carries the full §4.1 metadata key set; `chunk_id` via `make_chunk_id`; `splitter` from `chunker.kind`.
  - `build_vector_store(settings, base_dir=None, profile=None, *, doc_version=None, embedding_model=None)` — forwards params to `build_documents`; default `doc_version="g0"`, default `embedding_model=settings.embedding_model`.
  - `manifest_files_from_collection(store) -> list[dict]` — reads Chroma `get(include=["metadatas"])` and groups by `source_path` into registry `files` entries (`doc_id, path, doc_hash, chunk_ids, category, level_rank, corpus_source`), chunk_ids sorted.
  - `retrieval_tool._search_one_category` sets `"section": doc.metadata.get("section_path") or doc.metadata.get("section")`.

- [ ] **Step 1: Write the failing test additions**

Append to `tests/test_ingestion.py`:

```python
class TestAnchoredMetadata:
    def test_full_metadata_contract_on_every_chunk(self):
        docs = build_documents(PROJECT_ROOT, "main")
        assert docs
        required = {
            "doc_id", "doc_name", "category", "source_path", "corpus_source",
            "access_level", "level_rank", "chunk_id", "section_path",
            "section_title", "headings", "chunk_order", "doc_hash",
            "ingested_at", "doc_version", "embedding_model", "splitter",
            "token_count",
        }
        for d in docs:
            missing = required - set(d.metadata)
            assert not missing, f"chunk missing metadata keys: {missing}"
            assert d.metadata["splitter"] == "header"
            assert d.metadata["token_count"] >= 1
            assert len(d.metadata["doc_hash"]) == 64
            assert "T" in d.metadata["ingested_at"]  # ISO timestamp

    def test_rebuild_is_deterministic(self):
        a = build_documents(PROJECT_ROOT, "main")
        b = build_documents(PROJECT_ROOT, "main")
        ids_a = [d.metadata["chunk_id"] for d in a]
        ids_b = [d.metadata["chunk_id"] for d in b]
        assert ids_a == ids_b

    def test_section_path_populated_for_headed_docs(self):
        docs = build_documents(PROJECT_ROOT, "main")
        headed = [d for d in docs if d.metadata["doc_name"].startswith("SOP-OPR-101")]
        assert headed
        assert any(d.metadata["section_path"] != "." for d in headed)

    def test_injected_clock_and_params(self):
        docs = build_documents(
            PROJECT_ROOT, "main",
            doc_version="g42", embedding_model="test-model",
            now="2026-09-07T00:00:00+00:00",
        )
        for d in docs:
            assert d.metadata["doc_version"] == "g42"
            assert d.metadata["embedding_model"] == "test-model"
            assert d.metadata["ingested_at"] == "2026-09-07T00:00:00+00:00"

    def test_conflict_chunk_ids_still_unique(self):
        docs = build_documents(PROJECT_ROOT, "conflict", variant_dir=VARIANT_DIR)
        ids = [d.metadata["chunk_id"] for d in docs]
        assert len(ids) == len(set(ids))
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_ingestion.py -k Anchored -v`
Expected: FAIL — `section_path`/`doc_hash`/`splitter`/etc. missing from metadata (`KeyError`/assertions).

- [ ] **Step 3: Modify build_vector_store.py**

Replace the module docstring's "Run:" line block only if needed (no-op). Concretely:

(a) Add imports at the top of `src/nga/ingestion/build_vector_store.py` AND remove the now-unused splitter import (the module no longer calls `RecursiveCharacterTextSplitter` directly — the chunker owns it):

Remove:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

Add:
```python
from nga.ingestion.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    make_chunker,
)
from nga.ingestion.metadata import (
    doc_hash,
    estimate_tokens,
    make_chunk_id,
    utc_now_iso,
)
```

(b) Replace `build_documents` (keep `discover_documents` and all corpus-profile logic unchanged):

```python
def build_documents(
    base_dir: Path,
    profile: str = "main",
    variant_dir: Path | None = None,
    *,
    chunker: Any | None = None,
    doc_version: str = "g0",
    embedding_model: str = "unknown",
    now: str | None = None,
) -> list[Document]:
    """Chunk corpus Markdown files into langchain Documents with the full
    anchored-metadata contract (spec §4.1) and content-addressed chunk ids.

    `chunker` defaults to HeaderAwareMarkdownChunker(800, 120). `now` injects
    the ISO timestamp for deterministic tests. RBAC stays server-derived from
    the folder path — never from caller input.
    """
    if chunker is None:
        chunker = make_chunker(chunk_size=DEFAULT_CHUNK_SIZE,
                               chunk_overlap=DEFAULT_CHUNK_OVERLAP)
    ingested_at = now or utc_now_iso()
    documents: list[Document] = []

    for md_path, source in discover_documents(base_dir, profile, variant_dir):
        content = md_path.read_bytes()
        text = content.decode("utf-8")
        raw_chunks = chunker.split(text)

        path_str = str(md_path)
        access_level = level_from_path(path_str)
        level_rank = ACCESS_LEVELS[access_level]
        category = _get_category(md_path)
        doc_id = _get_doc_id(md_path)
        tag = SOURCE_TAG[source]
        file_hash = doc_hash(content)

        for c in raw_chunks:
            chunk_text = c.text.strip()
            if not chunk_text:
                continue
            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "doc_id":         doc_id,
                        "doc_name":       md_path.name,
                        "category":       category,
                        "source_path":    path_str,
                        "corpus_source":  source,
                        "access_level":   access_level,
                        "level_rank":     level_rank,
                        "chunk_id":       make_chunk_id(
                            doc_id=doc_id, source_tag=tag,
                            section_path=c.section_path,
                            chunk_order=c.chunk_order,
                            text=chunk_text,
                        ),
                        "section_path":   c.section_path,
                        "section_title":  c.section_title,
                        "headings":       list(c.headings),
                        "chunk_order":    c.chunk_order,
                        "doc_hash":       file_hash,
                        "ingested_at":    ingested_at,
                        "doc_version":    doc_version,
                        "embedding_model": embedding_model,
                        "splitter":       getattr(chunker, "kind", "header"),
                        "token_count":    estimate_tokens(chunk_text),
                    },
                )
            )
        logger.info(
            "Ingested %s → doc_id=%s category=%s level=%s source=%s chunks=%d",
            md_path.name, doc_id, category, access_level, source, len(raw_chunks),
        )

    return documents
```

(c) Update `build_vector_store` to forward the new params:

```python
def build_vector_store(
    settings: Settings,
    base_dir: Path | None = None,
    profile: str | None = None,
    *,
    doc_version: str | None = None,
    embedding_model: str | None = None,
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

    documents = build_documents(
        base_dir,
        profile=profile,
        variant_dir=variant_dir,
        doc_version=doc_version or "g0",
        embedding_model=embedding_model or settings.embedding_model,
    )
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
```

(d) Add the manifest-helper function at the end of `build_vector_store.py` (before `main`):

```python
def manifest_files_from_collection(store: Chroma) -> list[dict]:
    """Derive registry `files` entries from a Chroma collection's metadata.

    Groups chunks by source_path (one entry per ingested file) with sorted
    chunk_ids — feeds IndexRegistry manifests (spec §4.4).
    """
    all_meta = store.get(include=["metadatas"])["metadatas"]
    by_path: dict[str, dict] = {}
    for meta in all_meta:
        path = meta.get("source_path", "?")
        entry = by_path.setdefault(
            path,
            {
                "doc_id": meta.get("doc_id"),
                "path": path,
                "doc_hash": meta.get("doc_hash", ""),
                "chunk_ids": [],
                "category": meta.get("category"),
                "level_rank": meta.get("level_rank", 1),
                "corpus_source": meta.get("corpus_source", "canonical"),
            },
        )
        cid = meta.get("chunk_id")
        if cid:
            entry["chunk_ids"].append(cid)
    for entry in by_path.values():
        entry["chunk_ids"].sort()
    return sorted(by_path.values(), key=lambda e: (e["doc_id"], e["path"]))
```

- [ ] **Step 4: Fix the dead section metadata in retrieval_tool.py**

In `src/nga/tools/retrieval_tool.py`, `_search_one_category`, change the result-dict line:

```python
            "section": doc.metadata.get("section"),
```
to:
```python
            "section": doc.metadata.get("section_path") or doc.metadata.get("section"),
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
uv run pytest tests/test_ingestion.py tests/test_retrieval_formatting.py -v
```
Expected: PASS (existing + 5 new anchored-metadata tests). If a discovery-count test fails, chunk counts changed — that is expected and fine (assertions are on *file* discovery counts, not chunk counts); do not revert the splitter. Verify `tests/test_retrieval_formatting.py` still passes unchanged (it builds result dicts directly).

- [ ] **Step 6: Lint + full unit suite + commit**

```bash
uv run ruff check src/nga tests
uv run pytest tests/test_ingestion.py tests/test_retrieval_formatting.py tests/test_chunking.py tests/test_metadata.py tests/test_registry.py
git add src/nga/ingestion/build_vector_store.py src/nga/tools/retrieval_tool.py tests/test_ingestion.py
git commit -m "feat: anchored chunking with full metadata contract in ingestion pipeline"
```

---

### Task 5: Config knobs + generation publication + registry-aware corpus version

**Files:**
- Modify: `src/nga/config.py`
- Modify: `src/nga/ingestion/build_vector_store.py` (`main`)
- Modify: `src/nga/cache/layers.py` (`get_corpus_version`)
- Test: `tests/test_registry.py` (add publication integration), `tests/test_cache.py` (registry preference)

**Interfaces:**
- Consumes: Task 3 registry functions; Task 4 `build_vector_store(..., doc_version=...)`, `manifest_files_from_collection`.
- Produces:
  - `Settings` new fields: `index_registry_dir: str`, `chunker_kind: str`, `chunk_size: int`, `chunk_overlap: int`, `metadata_schema_version: int`.
  - `build_vector_store.main()` — for `profile == "main"`, reserves `generation = next_generation(registry_root)` before building, embeds it as `doc_version`, then publishes (`publish_generation`) with a `config` dict and changelog `["initial registry build"]`.
  - `nga.cache.layers.get_corpus_version()` — prefers the committed registry manifest's `corpus_version` for the main profile; existing mtime fallback otherwise.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_registry.py`:

```python
class TestPublishIntegration:
    def test_publish_then_read_current_roundtrip(self, tmp_path):
        src = tmp_path / "chroma"
        (src / "x").mkdir(parents=True)
        (src / "chunks.sqlite3").write_text("blob")
        manifest = publish_generation(
            tmp_path, "g1", source_dir=src, files=FILES, config=CONFIG,
            changelog=["initial registry build"],
        )
        current = read_current(tmp_path)
        assert current["generation"] == "g1"
        assert current["parent_generation"] is None
        assert current["changelog"] == ["initial registry build"]
        assert manifest["corpus_version"] == current["corpus_version"]

    def test_second_publish_records_parent(self, tmp_path):
        src = tmp_path / "chroma"
        (src / "x").mkdir(parents=True)
        (src / "chunks.sqlite3").write_text("blob")
        publish_generation(tmp_path, "g1", source_dir=src, files=FILES, config=CONFIG)
        other = [dict(FILES[0], doc_hash="f" * 64)] + FILES[1:]
        publish_generation(tmp_path, "g2", source_dir=src, files=other, config=CONFIG,
                           changelog=["delta: modify SOP-OPR-101"])
        current = read_current(tmp_path)
        assert current["generation"] == "g2"
        assert current["parent_generation"] == "g1"
```

Append to `tests/test_cache.py`:

```python
def test_corpus_version_prefers_registry_manifest(tmp_path, monkeypatch):
    """Spec §4.5: once a committed manifest exists, its content-derived
    corpus_version wins over the mtime heuristic."""
    from types import SimpleNamespace

    from nga.cache.layers import get_corpus_version
    from nga.ingestion.registry import publish_generation, read_current

    # Build a committed registry generation in a tmp root.
    src = tmp_path / "chroma"
    (src / "x").mkdir(parents=True)
    (src / "chunks.sqlite3").write_text("blob")
    files = [{
        "doc_id": "SOP-OPR-101", "path": "operator-sops/SOP-OPR-101.md",
        "doc_hash": "a" * 64, "chunk_ids": ["SOP-OPR-101:c:1:000:abc12345"],
        "category": "OPR", "level_rank": 1, "corpus_source": "canonical",
    }]
    publish_generation(tmp_path, "g1", source_dir=src, files=files,
                       config={"metadata_schema_version": 2})
    expected = read_current(tmp_path)["corpus_version"]
    assert expected.startswith("cv-")

    # Stub Settings so get_corpus_version reads the tmp registry root.
    import nga.config as config_mod

    monkeypatch.setattr(
        config_mod.Settings, "from_env",
        classmethod(lambda cls: SimpleNamespace(
            corpus_profile="main",
            index_registry_dir=str(tmp_path),
            vector_store_dir=str(tmp_path / "store"),
        )),
    )
    assert get_corpus_version() == expected
```

Note: `get_corpus_version` memoizes per `vector_store_dir`; the stub uses a unique tmp path so no cross-test pollution. `nga.config.Settings.from_env` is a classmethod imported function-locally inside `get_corpus_version`, so patching it on the class works.

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_registry.py -k PublishIntegration tests/test_cache.py::test_corpus_version_prefers_registry_manifest -v`
Expected: `test_second_publish_records_parent` and `test_corpus_version_prefers_registry_manifest` FAIL — `get_corpus_version` does not yet read the registry manifest (Task 5 step 5). `test_publish_then_read_current_roundtrip` PASSes already (functions exist from Task 3) — that is expected; it guards the publish contract for later tasks.

- [ ] **Step 3: Add Settings fields**

In `src/nga/config.py`:
(a) Add to the dataclass field block (after `conflict_vector_store_dir`):

```python
    index_registry_dir: str
    chunker_kind: str
    chunk_size: int
    chunk_overlap: int
    metadata_schema_version: int
```

(b) Add a parser near `_parse_corpus_profile`:

```python
_VALID_CHUNKER_KINDS = {"header", "recursive"}


def _parse_chunker_kind() -> str:
    raw = (os.getenv("NGA_CHUNKER", "header") or "header").strip().lower()
    if raw not in _VALID_CHUNKER_KINDS:
        raise ValueError(
            f"NGA_CHUNKER must be one of {sorted(_VALID_CHUNKER_KINDS)}, got: {raw!r}"
        )
    return raw
```

(c) In `from_env`, after the `conflict_vector_store_dir=...` line, add:

```python
            index_registry_dir=os.getenv(
                "INDEX_REGISTRY_DIR", "data/index_registry"
            ),
            chunker_kind=_parse_chunker_kind(),
            chunk_size=int(os.getenv("NGA_CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("NGA_CHUNK_OVERLAP", "120")),
            metadata_schema_version=int(
                os.getenv("NGA_METADATA_SCHEMA_VERSION", "2")
            ),
```

- [ ] **Step 4: Publish a generation in build_vector_store.main()**

In `src/nga/ingestion/build_vector_store.py`, replace the body of `main()` (keep argparse/logging setup) so the main profile publishes a registry generation:

```python
    settings = Settings.from_env()
    profile = args.profile or settings.corpus_profile
    # Auto-detect corpus root (parent of database/)
    base_dir = Path(settings.nga_db_path).parent.parent
    logger.info("Ingesting NGA corpus (profile=%s) from: %s", profile, base_dir)

    registry_root = Path(settings.index_registry_dir)
    generation: str | None = None
    if profile == "main":
        from nga.ingestion.registry import next_generation

        generation = next_generation(registry_root)
        logger.info("Publishing main build as generation=%s", generation)

    store = build_vector_store(
        settings,
        base_dir=base_dir,
        profile=profile,
        doc_version=generation or "g0",
    )
    store_dir = profile_store_dir(settings, profile)
    logger.info(
        "Vector store built at %s — %d chunks embedded.",
        store_dir,
        store._collection.count(),
    )

    if generation is not None:
        from nga.ingestion.registry import publish_generation

        config = {
            "chunker": {"type": settings.chunker_kind,
                        "chunk_size": settings.chunk_size,
                        "chunk_overlap": settings.chunk_overlap},
            "embedding": {"model": settings.embedding_model},
            "metadata_schema_version": settings.metadata_schema_version,
        }
        files = manifest_files_from_collection(store)
        manifest = publish_generation(
            registry_root,
            generation,
            source_dir=store_dir,
            files=files,
            config=config,
            changelog=["initial registry build"],
        )
        logger.info(
            "Published generation=%s corpus_version=%s files=%d",
            manifest["generation"], manifest["corpus_version"], len(files),
        )
```

- [ ] **Step 5: Registry-aware corpus version in cache/layers.py**

In `src/nga/cache/layers.py`, replace `get_corpus_version` so it prefers the committed registry manifest when one exists for the main profile:

```python
def get_corpus_version() -> str:
    """Memoized corpus version. Prefers the IndexRegistry manifest's
    content-derived corpus_version (spec §4.4–4.5); falls back to the
    mtime heuristic when no committed generation exists (unit tests,
    pre-registry stores)."""
    cache = _corpus_version_cache
    key = "default"
    try:
        from nga.config import Settings

        settings = Settings.from_env()
        store_dir = settings.vector_store_dir
        if store_dir:
            key = str(Path(store_dir).resolve())
        if key not in cache:
            if settings.corpus_profile == "main":
                from nga.ingestion.registry import read_current

                manifest = read_current(settings.index_registry_dir)
                if manifest and manifest.get("corpus_version"):
                    cache[key] = manifest["corpus_version"]
                    return cache[key]
            cache[key] = compute_corpus_version(store_dir)
        return cache[key]
    except Exception:
        return cache.get(key, "cv-default")
```

- [ ] **Step 6: Verify behavior + full suite + commit**

Smoke check (registry publish path, no real corpus needed):

```bash
cd "$(git rev-parse --show-toplevel)"
uv run python - <<'PY'
import json, tempfile
from pathlib import Path
from nga.ingestion.registry import publish_generation, read_current

root = Path(tempfile.mkdtemp())
src = root / "chroma"; src.mkdir(); (src / "x").write_text("y")
files = [{"doc_id": "SOP-OPR-101", "path": "operator-sops/SOP-OPR-101.md",
          "doc_hash": "a"*64, "chunk_ids": ["SOP-OPR-101:c:1:000:abc12345"],
          "category": "OPR", "level_rank": 1, "corpus_source": "canonical"}]
cfg = {"metadata_schema_version": 2}
publish_generation(root, "g1", source_dir=src, files=files, config=cfg,
                   changelog=["initial registry build"])
m = read_current(root)
print("generation:", m["generation"], "| corpus_version:", m["corpus_version"])
assert m["corpus_version"].startswith("cv-")
print("OK")
PY
```

Then run the whole unit suite and lint:

```bash
uv run ruff check src/nga tests
uv run pytest tests/ -v --ignore=tests/integration
```

Expected: all unit tests PASS (registry preference test included). Then commit:

```bash
git add src/nga/config.py src/nga/ingestion/build_vector_store.py src/nga/cache/layers.py tests/test_registry.py tests/test_cache.py
git commit -m "feat: registry generation publication on main builds; registry-aware cache corpus version"
```

---

### Task 6: Documentation + end-to-end verification

**Files:**
- Modify: `.env.example` (document new knobs)
- Modify: `README.md` (ingestion section — registry generation step)

**Interfaces:** Consumes the CLI surface produced in Tasks 4–5. No code contracts.

- [ ] **Step 1: Document env knobs in .env.example**

Append to `.env.example`:

```bash
# ── Anchored ingestion (Workstream A) ─────────────────────────────
# Chunker: header (markdown heading-aware, default) | recursive (legacy)
NGA_CHUNKER=header
NGA_CHUNK_SIZE=800
NGA_CHUNK_OVERLAP=120
# IndexRegistry root for generation snapshots + atomic pointer
INDEX_REGISTRY_DIR=data/index_registry
# Bump when the chunk metadata contract changes (invalidates config_hash)
NGA_METADATA_SCHEMA_VERSION=2
```

- [ ] **Step 2: Update README ingestion commands**

In `README.md` "Build the Retrieval Bases" section, change the `build_vector_store` bullet text to note the registry generation:

```markdown
# Ingests all 48 SOPs/specs with RBAC metadata (header-aware, content-addressed)
# Publishes a versioned IndexRegistry generation (data/index_registry/) —
# cache corpus version derives from the committed manifest.
uv run python -m nga.ingestion.build_vector_store
```

- [ ] **Step 3: End-to-end verification**

If a real `.env`/provider is available, run the real pipeline:

```bash
uv run python -m nga.ingestion.build_vector_store
uv run python -m nga.ingestion.build_graph   # still consumes the current collection
ls -R data/index_registry
```

Expected: `data/index_registry/current.json` → `{"generation": "g1"}` and `data/index_registry/generations/g1/{manifest.json,vector_store/}` exist. If no provider/key is configured, skip this step and rely on the unit suite + smoke test from Task 5.

- [ ] **Step 4: Lint + full suite + commit**

```bash
uv run ruff check src/nga tests
uv run pytest tests/test_ingestion.py tests/test_chunking.py tests/test_metadata.py tests/test_registry.py tests/test_cache.py tests/test_retrieval_formatting.py -v
git add .env.example README.md
git commit -m "docs: anchored ingestion env knobs, registry generation steps"
```

---

## Self-Review Notes

- **Spec coverage (§5 + §4):** metadata keys/chunk-id formula (§4.1/4.2) → Task 1; Chunker seam + header splitter (§5) → Task 2; registry + corpus version + config hash (§4.4) → Tasks 3/5; build integration + section_path fix (§5, known defect) → Task 4; cache corpus-version preference (§4.5) → Task 5; env/README → Task 6. RBAC derivation untouched (invariant preserved). Deterministic rebuild, cross-source chunk_id uniqueness, and profile behavior tests retained.
- **Placeholder scan:** no TBD/TODO; every step has concrete code or commands.
- **Type consistency:** `make_chunk_id(doc_id, source_tag, section_path, chunk_order, text)` used identically in metadata.py (Task 1) and build_vector_store (Task 4); `Chunk.text/section_path/section_title/headings/chunk_order` names match chunking.py and the Task 4 metadata mapping; registry signatures (`publish_generation(root, generation, *, source_dir, files, config, changelog)`, `read_current`, `next_generation`) are identical between Task 3 tests and Task 5 `main()`; Settings field names match config parsing in Task 5. `manifest_files_from_collection` keys (`doc_id, path, doc_hash, chunk_ids, category, level_rank, corpus_source`) match the Task 3 `FILES` fixtures.
- **Known accepted limitation:** on a failed main build that reserved a generation id, a rerun publishes under the next id while chunk metadata records the reserved one (`doc_version`); harmless (doc_version is informational) and documented in the spec's error-handling note.
