# Graph RAG Reference Architecture — NGA Manufacturing Assistant (Master Design)

**Status:** Approved in design review (2026-09-07)
**Owners:** Platform/Agent team
**Drivers (ranked):** C — production-grade reference pattern; B — latency/cost at query time; A — retrieval quality ceiling
**Fidelity frame:** Blueprint with seams — ideal production components defined behind clean interfaces; local stand-ins used where the existing stack can support them honestly; full-fidelity infrastructure explicitly marked **deferred**.
**Related docs:** `docs/cache-design.md`, `docs/variant-corpus-ingestion-design.md`, `docs/model-capability-gating-design.md`, `docs/superpowers/plans/2026-09-07-ws-a-anchored-ingestion.md`
**Implementation:** Per-workstream plans, sequenced below; each plan implements one subsystem from this spec.

---

## 1. Purpose & Scope

The NGA Manufacturing Assistant already ships a Graph RAG system (Chroma vector store + NetworkX entity graph + LangGraph orchestration + HITL + caching + CI-tracked eval). This spec upgrades it into a **reference architecture** for realistic enterprise Graph RAG deployment. The synthetic plant corpus is the testbed; the *pattern* is the product.

Design philosophy, from the ranked drivers:

1. **Expensive work belongs at ingestion time.** Entity extraction, alignment, indexing, and reranker *training data* are build-time. The query path must be cheap: intent routing (no LLM), parallel local retrievers, RRF, and a rerank cut to 3–5 chunks.
2. **Nothing overwrites in place.** Every index artifact is an immutable, generation-tagged snapshot behind an `IndexRegistry`. Rollback is a pointer flip. Delta updates publish new generations through a confidence gate.
3. **Every claim and every graph element carries provenance.** Answers cite `[chunk_id]`/`[node_id]` from the provided context; graph elements record `source_chunks[]`; snapshots record their build config hash.
4. **RBAC is an invariant, not a filter.** Access level is derived server-side from document folder paths at ingestion and enforced at *every* retrieval stage (dense filter, BM25 filter, graph traversal prune, context assembly).

## 2. Current State (what this builds on)

| Capability | Where it lives today | Gap this spec closes |
|---|---|---|
| LangGraph orchestration (`prepare → agent ⇄ tools → synthesis → hitl`) | `src/nga/graph/orchestrator.py`, `nodes.py`, `state.py` | Tool seam formalized as MCP gateway (§9); claim-level citation post-check |
| Vector store + RBAC metadata filter | `src/nga/ingestion/build_vector_store.py`; Chroma; per-category parallel search in `retrieval_tool.py` | Anchored metadata schema, section hierarchy, deterministic chunk IDs, registry publication (§5) |
| NetworkX entity graph + LLM extraction + Louvain | `src/nga/ingestion/build_graph.py`, `src/nga/graphrag/search.py` | 3-step extraction pipeline, entity alignment, provenance, confidence gating, community summaries as first-class artifacts (§6) |
| Hybrid dense + graph evidence | `retrieval_tool.retrieve_documents` + `retrieve_graph_evidence` (separate, unranked) | BM25, intent router, RRF fusion, reranker seam, unified context assembly (§7) |
| Heuristic complexity classifier → model tier | `src/nga/rag_agent/classifier.py` | Intent routing decoupled from model-tier routing (§7) |
| Caching (L0 emb / L1 retr / L2 sql) | `src/nga/cache/` | Corpus version becomes registry-generation derived (content-based, not mtime) |
| Eval: 85 Q, 6 categories, CI tracker, comparison studio | `eval-questions/questions.json`, `src/nga/evaluation/` | L1–L4 tier crosswalk, evidence-hit harness, stepped rollout gate (§8) |
| HITL approval queue | `src/nga/hitl/approval.py`, `memory/decision_log.py`, UI queue | Extended to ingestion review (graph element `pending_review`) |
| SQL tool w/ SQLGlot read-only guard | `src/nga/tools/sql_tool.py` | Unchanged (reference implementation for DB access) |

Known defect this spec fixes: `retrieval_tool.py` reads `metadata["section"]`, but ingestion never populates it.

## 3. Target Topology

```
┌─ Governance plane ───────────────────────────────────────────────┐
│ IndexRegistry (generation snapshots · atomic pointer · rollback) │
│ QualityMetrics · CircuitBreaker + HITL ingestion review queue    │
│ Provenance/audit: element→source_chunks, answer claim→source     │
└──────────────▲───────────────────────────────▲───────────────────┘
┌─ Ingestion plane ────┐        ┌─ Query plane ───────────────────┐
│ A. Anchored chunker  │        │ IntentRouter → strategy          │
│    (header-aware,    │        │ Dense · BM25 · Graph tri-hybrid  │
│    anchored metadata)│        │ RRF fusion → Reranker(seam) →3-5 │
│ B. KG v2 (3-step,    │        │ Context assembly: chunks +       │
│    provenance, conf) │        │ logical chains + community sums  │
└────────┬─────────────┘        └──────────────▲───────────────────┘
         │ publish generations                 │ open per request
         └──────────► IndexRegistry ◄──────────┘
┌─ Eval/observability plane ───────────────────────────────────────┐
│ L1–L4 tier crosswalk · evidence-hit harness · stepped rollout    │
│ regression gate · CI tracker (existing) extended                 │
└──────────────────────────────────────────────────────────────────┘
```

All planes are implemented as plain Python modules in `src/nga/`. No new external services are required for the reference implementation; seams that would require them are marked **deferred** and documented with their interface contract so a deployment can substitute real infrastructure without changing callers.

## 4. Cross-Cutting Contracts

### 4.1 Canonical chunk metadata schema

Every chunk carries exactly these metadata keys (constants live in `nga.ingestion.metadata`). `level_rank`/`access_level` are **always derived server-side from `source_path`** (never from caller input).

| Key | Type | Source |
|---|---|---|
| `doc_id` | str | filename ID pattern (existing) |
| `doc_name` | str | filename (existing) |
| `category` | str ∈ {OPR,TEC,FA,QCR,WO,SQ,TR} | folder map (existing) |
| `source_path` | str | relative corpus path (existing) |
| `corpus_source` | str ∈ {canonical, variant} | profile (existing) |
| `access_level` | str | RBAC table (existing) |
| `level_rank` | int 1–4 | RBAC table (existing) |
| `chunk_id` | str | content-addressed, §4.2 (new semantics) |
| `section_path` | str | heading chain joined by `/`; `"."` if no headings (new) |
| `section_title` | str | deepest heading text; `""` if none (new) |
| `headings` | list[str] | full heading chain for the chunk (new) |
| `chunk_order` | int | ordinal of chunk within document (new) |
| `doc_hash` | str | sha256 hex of source file bytes (new) |
| `ingested_at` | str | ISO-8601 UTC timestamp (new) |
| `doc_version` | str | corpus generation id at ingestion, §4.4 (new) |
| `embedding_model` | str | embedding model id (new) |
| `splitter` | str | `header` \| `recursive` (new) |
| `token_count` | int | estimate `ceil(chars/4)` (new) |

### 4.2 Deterministic chunk identity

```
chunk_id = f"{doc_id}:{source_tag}:{section_key}:{chunk_order:03d}:{content_h8}"
source_tag  = "c" | "v"   # canonical | variant (existing SOURCE_TAG, prevents cross-source collisions)
section_key = slug(section_path)[:48]  ("." → "root"); slug = lowercase, [^a-z0-9]+ → "-"
content_h8  = sha256(chunk_text)[:8]
```

Properties (required by delta ingestion, cache stability, snapshot diffing):
- Same file content + same splitter + same section structure ⇒ identical `chunk_id` across rebuilds.
- Changed text ⇒ changed `content_h8` ⇒ the affected chunk's id changes and its neighbors' ids are stable unless ordering shifts.
- The `chunk_id` uniqueness invariant across `canonical`/`variant` sources is guaranteed by the embedded `source_tag` — a canonical and a variant copy of the same document share unchanged sections but never share `chunk_id`s (conflict profile keeps them in one collection). The existing cross-source uniqueness test must keep passing (see §5 testing).

### 4.3 RBAC invariant

Every retrievable unit (chunk, graph node, graph edge, BM25 doc) carries `level_rank`. Every retrieval path — dense filter, BM25 filter, graph BFS prune, context assembly, reranker candidate set — filters `level_rank <= user_level` before the unit is visible to the model. Failure mode is *deny*: an unreadable unit is never included.

### 4.4 IndexRegistry & generations

`data/index_registry/` is the single source of truth for what is currently served:

```
data/index_registry/
  current.json                  # {"generation": "g7"}  (atomic pointer)
  generations/
    g7/
      manifest.json             # full build record (below)
      vector_store/             # Chroma persist dir for this generation
      bm25/                     # BM25 index artifacts (workstream C)
      graph.json                # NetworkX serialization (workstream B)
      communities.json          # community membership + summaries
```

`manifest.json` record:
```json
{
  "generation": "g7",
  "parent_generation": "g6",
  "corpus_version": "cv-<hash>",
  "built_at": "ISO-8601",
  "config_hash": "sha256 of {chunker, embedding_model, metadata_schema_version, graph_extraction_model}",
  "metadata_schema_version": 2,
  "chunker": {"type": "header", "chunk_size": 800, "chunk_overlap": 120},
  "embedding": {"model": "...", "dim": 1024},
  "files": [
    {"doc_id": "SOP-OPR-101", "path": "operator-sops/SOP-OPR-101-wheel-installation-torque.md",
     "doc_hash": "sha256", "chunk_ids": ["SOP-OPR-101:1-purpose:000:ab12cd34", "..."],
     "category": "OPR", "level_rank": 1, "corpus_source": "canonical"}
  ],
  "changelog": ["delta: add SOP-OPR-999", "delta: modify SOP-TEC-214"],
  "status": "committed"
}
```

Rules:
- Builders write into a fresh `generations/<g>/` directory, then flip `current.json` by atomic rename. A served generation is never mutated in place.
- Rollback = repoint `current.json` to the previous committed generation.
- `corpus_version` in the manifest is **content-derived** (hash of `doc_hash` + `chunk_ids` per file) — this becomes the cache-key corpus version (§4.5), replacing the current mtime-based `compute_corpus_version`.
- `config_hash` mismatch (chunker params, embedding model, metadata schema, extraction model changed) ⇒ full re-ingest forced; delta path refuses to proceed.
- **Seam/deferred:** real deployments would host this as an object-store + catalog (S3 + registry table). Local filesystem directory + `current.json` is the reference implementation; the `IndexRegistry` API (§5) abstracts storage so callers do not change.

### 4.5 Cache versioning

Cache keys already scope by `corpus_version` (`docs/cache-design.md` §4). The registry generation's `corpus_version` becomes the value returned by `nga.cache.layers.get_corpus_version()` (with mtime-based fallback until a registry manifest exists, so unit tests and pre-A stores stay valid). Publishing a new generation therefore invalidates stale cache entries by construction — no cache sweep needed.

### 4.6 Provenance & audit

- Graph nodes/edges record `source_chunks[]` (chunk IDs that evidence them), `confidence`, `extraction_model`, `first_seen`, `last_updated`, `version`, `status`.
- Answers persist a claim→source map: each `FinalAnswer.evidence` entry must reference a `chunk_id`/`node_id` present in the provided context (structural post-check, §9).
- `decisions_log` (existing HITL audit) is extended with ingestion-review actions (`approve/reject/edit` of pending graph elements).

## 5. Workstream A — Anchored Ingestion v2 (spec-ready)

**Goal:** Content-addressed, header-aware chunking with the full §4.1 metadata contract; ingestion publishes registry generations.

### Components

**`src/nga/ingestion/metadata.py`** (new) — the contract module.
- `CHUNK_META_KEYS: tuple[str, ...]` — canonical key set (§4.1).
- `doc_hash(path_or_bytes) -> str` — sha256 hex.
- `make_chunk_id(*, doc_id, source_tag, section_path, chunk_order, text) -> str` — §4.2 formula; `source_tag` is `"c"`/`"v"` (existing `SOURCE_TAG`); raises `ValueError` if `text` empty or `source_tag` not in `{"c", "v"}`.
- `section_key(section_path) -> str` — slugification helper.
- `estimate_tokens(text) -> int` — `ceil(len(text)/4)`.

**`src/nga/ingestion/chunking.py`** (new) — the `Chunker` seam.
- `DocumentChunker` protocol: `split(text: str) -> list[Chunk]` where `Chunk = dataclass(text, section_path, section_title, headings, chunk_order)`.
- `HeaderAwareMarkdownChunker(chunk_size=800, chunk_overlap=120)` — reference implementation:
  - Parses ATX headings (`^#{1,6} `). The `#` document title starts a root section; `##`/`###` open child sections.
  - Splits the document into heading-delimited sections, carrying the full heading chain.
  - Within a section, assembles non-heading paragraphs into chunks of ≤ `chunk_size` characters; a chunk never spans a heading boundary.
  - When a section is longer than `chunk_size`, splits on paragraph boundaries (blank line), falling back to a hard split at `chunk_size`; appends the last `chunk_overlap` chars of the previous chunk as a margin where that does not cross a heading.
  - Content before the first heading and heading *text itself* are included in the owning section's chunks.
- `RecursiveFallbackChunker(chunk_size, chunk_overlap)` — wraps the existing `RecursiveCharacterTextSplitter` for non-Markdown or when `NGA_CHUNKER=recursive`.
- `make_chunker(*, kind, chunk_size, chunk_overlap) -> DocumentChunker`.
- Splitting is a pure function of text: same text + params ⇒ identical chunk boundaries (testable determinism).

**Modify `src/nga/ingestion/build_vector_store.py`** — `build_documents(...)` gains `chunker: DocumentChunker | None` param (default from `Settings`); each chunk gets full §4.1 metadata with `chunk_id = make_chunk_id(...)`, `section_path/section_title/headings`, `chunk_order`, `doc_hash`, `ingested_at` (UTC now, injectable clock for tests), `doc_version`, `embedding_model`, `splitter`. Discovery, corpus profiles, RBAC path derivation, `SOURCE_TAG`, and `EXCLUDED_FILES` behavior unchanged.

**`src/nga/ingestion/registry.py`** (new) — minimal `IndexRegistry`.
- `next_generation(root) -> str` — `g<int>` from highest existing + 1.
- `write_manifest(root, generation, *, parent_generation, files, config, changelog, status="committed") -> dict`
- `read_manifest(root, generation) -> dict | None`; `read_current(root) -> dict | None` (follows `current.json`).
- `publish_generation(root, generation_dir_map) -> None` — atomically writes `current.json` via temp-file + `os.replace`.
- `snapshot_artifacts(src_dir, dst_dir)` — copy-on-write directory copy used when publishing the Chroma persist dir.
- `compute_corpus_version(files_meta) -> str` — content hash over `(doc_hash, chunk_ids)` per file (§4.4).
- `build_config_hash(*, chunker, embedding_model, schema_version) -> str`.
- Storage seam: all paths resolved through `Settings.index_registry_dir`; callers never touch paths directly beyond this module.

**Modify `src/nga/ingestion/build_vector_store.py::main`** — after a successful build, publish a generation: snapshot the Chroma dir into `generations/<g>/vector_store`, write `manifest.json`, flip `current.json`. Existing plain-build CLI path kept for `variant`/`conflict` profiles (registry targets the `main` production profile only in A; profiles remain explicit).

**Config (`src/nga/config.py`)** — add `index_registry_dir`, `chunker_kind` (`header` default), `chunk_size` (800), `chunk_overlap` (120), `metadata_schema_version` (2). Env: `INDEX_REGISTRY_DIR`, `NGA_CHUNKER`, `NGA_CHUNK_SIZE`, `NGA_CHUNK_OVERLAP`, `NGA_METADATA_SCHEMA_VERSION`.

**Modify `src/nga/cache/layers.py::get_corpus_version`** — prefer registry manifest `corpus_version` for the main store when a committed generation exists; keep mtime fallback (unit tests, no registry).

**Modify `src/nga/tools/retrieval_tool.py`** — populate result dicts and the formatted block with `section_path` (fall back to `section`), fixing the dead `section` metadata bug.

### Error handling

- Chunker: unreadable file / empty text ⇒ `[]` chunks (existing skip-empty behavior preserved).
- `make_chunk_id` raises on empty text (programmer error, fail loud).
- Registry publish: atomic rename failure ⇒ previous `current.json` untouched (temp-file + replace); manifest write precedes pointer flip.
- Build failure mid-generation ⇒ orphan `generations/<g>/` dir tolerated (never pointed to); `next_generation` skips ahead.

### Testing strategy

- `tests/test_chunking.py` (new): section boundaries respected; heading chain recorded; no chunk crosses a `##` boundary; determinism (same text ⇒ same boundaries & ids); overlap margin; hard-split fallback; title heading handling; `estimate_tokens`.
- `tests/test_metadata.py` (new): `chunk_id` formula, slugification, empty-text rejection, `doc_hash` stability.
- `tests/test_registry.py` (new): manifest round-trip; generation monotonicity; atomic pointer flip leaves a valid `current.json`; corpus version content-hash changes when a file's `doc_hash` changes and is stable otherwise.
- Extend `tests/test_ingestion.py`: existing suites keep passing (discovery counts, RBAC, cross-source `chunk_id` uniqueness); new assertions — every doc has the full §4.1 key set; `section_path` populated for headed docs; rebuild idempotence (run `build_documents` twice ⇒ identical `chunk_id` sets).
- `tests/test_retrieval_formatting.py`: section path appears in rendered context.

### Seams / deferred

- Streaming ingestion bus, object-store-backed registry, DB-backed catalog: **deferred** (interface is `registry.py`).
- Chunker parameters remain env-tunable; no online chunker hot-swap (config_hash forces rebuild).

## 6. Workstream B — KG Construction v2 (spec-ready)

**Goal:** 3-step extraction (lightweight candidates → few-shot LLM refinement → entity alignment) with provenance, confidence, community summaries, and delta ingestion gates. Replaces the single-pass per-chunk LLM extraction in `build_graph.py`.

### Components

**`src/nga/graphkg/extract.py`** (new package `nga.graphkg`):
- `CandidateExtractor` seam → reference impl `NgaRuleExtractor`:
  - High-throughput, no LLM. Regex/dictionary over NGA entity vocabularies: SOP/TEC/FAP/ESC/QCR/WO/SCAR/AUD/APP/TR IDs, torque tools (`TQ-6012`…), machine/robot ids, P-codes/E-codes, Class A systems (brake/steering/…), supplier names, criterion codes (C1–C5).
  - Emits candidate spans `{text, entity_type, span}` plus shallow relation *cues* (verb adjacency to typed entities, e.g., `requires`, `escalates_to`).
  - **Seam/deferred:** GLiNER/spaCy transformer extractor as the full-fidelity target; interface is the same, runtime is a model service.
- `TripleRefiner` seam → reference impl `LlamaRefiner` (LLM few-shot over OpenRouter tier-1/2 or Ollama):
  - Input: candidates + their source chunk(s), **not** raw corpus dumps. LLM validates each candidate triple (accept/reject), prunes duplicates, assigns `confidence ∈ [0,1]`, enriches attributes (e.g., numeric thresholds, units), and returns strict JSON.
  - Batched (≤ 20 candidate sets per call) to bound token cost; cost scales with extraction *candidates*, which is ≪ corpus size — this is the token-optimization lever vs. today's open-extraction-per-chunk.
  - Call budget/retry policy: single retry on malformed JSON; on repeated failure the candidate set is dropped and logged (never crashes the build).
- `EntityAligner` seam → reference impl `CooccurrenceAligner`:
  - Maps candidate mentions → canonical global entity ID. Signals: exact ID match; alias table (persisted on node, e.g., `"TQ-6012"` ↔ `"TorqMaster TF-6000"`); normalized string similarity (token/abbreviation); document-level co-occurrence (two mentions in the same doc/section with no distinct definition are candidates for the same entity).
  - Emits `alignment` records `{mention, resolved_global_id, confidence, signal}` for the quality metric (§6.4).
  - **Seam/deferred:** embedding-similarity or LLM-confirmed alignment as the full-fidelity target.

**Modify `src/nga/ingestion/build_graph.py`** → orchestrates the 3 steps per chunk set; node/edge payload per §4.6. Community detection (existing Louvain/greedy) retained, and `src/nga/graphrag/search.py` community output is filled: a `CommunitySummarizer` (LLM over member nodes, RBAC-pruned, ≤ 500 tokens each) writes `communities.json` at build time (query path only reads it — summaries are *not* generated per query).

### Graph element model

Node: `{id, type, name, description, aliases[], confidence, source_chunks[], level_rank, extraction_model, first_seen, last_updated, version, status}`.
Edge: `{source, target, relation, description, confidence, source_chunks[], level_rank, status}`.
`status ∈ {committed, pending_review, superseded}` — `superseded` elements are excluded from retrieval but retained for audit/rollback.

### Delta ingestion (incremental updates)

Full pipeline in §10. Workstream-B-specific rules:
- Only changed chunks feed Step 1 → Step 2.
- Affected elements = those whose `source_chunks[]` intersect changed chunks; re-evaluated against remaining supporting chunks (keep w/ reweighted confidence, or `superseded`).
- New/ambiguous elements and contradiction cases route to the **HITL ingestion review queue** (`status=pending_review`), reusing the existing approval pattern (`nga.hitl` + `decisions_log` + UI queue extended). Nothing unreviewed is published to the served generation.

### Quality metrics (fed to §8 observability)

`extraction_precision` and `relation_accuracy` on a golden subset (documents annotated with expected triples); `alignment_success_rate` (mentions resolved without creating spurious new nodes); `coverage` (fraction of docs with ≥ 1 sourced node); `quarantine_rate` (delta elements sent to review); `graph_staleness` (days since last committed generation).

### Error handling & testing

- Every step degrades to "no output for this batch + logged" — a broken refiner call never fails the whole build; alignment failure keeps the mention as its own node with `confidence=0` and routes to review.
- Tests: rule extractor recall on planted corpora (SOP files contain known entities); refiner prompt→strict-JSON contract (golden payloads, malformed-JSON retry); aligner merges `"TQ-6012"`/`"TorqMaster TF-6000"` mentions; provenance fields present on every element; delta path produces expected `superseded` set for an edited doc; quarantine gate routes low-confidence elements to `pending_review`.
- **Seams/deferred:** transformer NER service, embedding/LLM alignment service.

## 7. Workstream C — Retrieval Stack v2 (spec-ready)

**Goal:** intent-routed tri-hybrid retrieval with RRF fusion, a reranker seam, and token-budgeted structured context assembly. Dense-only today ⇒ latency-capped routing + quality levers.

### 7.1 IntentRouter

`src/nga/rag_agent/intent.py` — `classify_intent(query, *, user_level) -> Intent` with `Intent = dataclass(kind ∈ {lookup, hierarchy, multi_hop, sql}, strategy, params)`.

| Intent | Signals (heuristic, sub-5 ms — extends `classifier.py` cues) | Retrieval strategy | Latency budget |
|---|---|---|---|
| `lookup` | "what is", "how many", single known ID, exact code/torque/term | dense + BM25 only | < 50 ms retrieval |
| `hierarchy` | "reporting tree", "organization", "escalates to", "who reports to", workflow-matrix docs | graph traversal (+ dense fallback) | < 100 ms |
| `multi_hop` | "root cause", "impact of", "if … then …", 2+ entity IDs, cross-category | tri-hybrid (dense + BM25 + graph 1–3 hop) | budgeted per tier |
| `sql` | numeric aggregates, "how many NCRs", requires_sql flag | unchanged SQL tool path | n/a |

- Reference impl: deterministic keyword/entity-cue classifier (no LLM on the hot path). Ambiguous → `multi_hop` (conservative). `requires_sql` from question metadata and numeric/aggregate cues route `sql`.
- **Seam/deferred:** learned/LLM router as the full-fidelity target; same interface.
- Decoupling note: the existing complexity classifier keeps deciding the *model tier*; the intent router decides the *retrieval strategy*. Two orthogonal axes, two cheap classifiers, zero LLM.

### 7.2 Tri-hybrid retrievers

`Retriever` seam in `src/nga/tools/retrievers/`:
- `DenseRetriever` — existing Chroma per-category search behind the seam (RBAC filter unchanged).
- `Bm25Retriever` — new: inverted index over chunk corpus built at ingestion into the generation (`bm25/` artifacts; `rank_bm25` or a compact custom index). Query-time: tokenize (case-insensitive), score, filter by `level_rank`, return chunk ids. Corpus is small (hundreds of chunks) — a local index is honest and fast; **scale vector DB is deferred**.
- `GraphRetriever` — upgraded `graphrag/search.py`: seed selection by embedding cosine + exact-ID match; BFS 1–3 hops (per-tier `max_hops` from existing `Settings`); RBAC-pruned; returns both **entities and traversed paths** (`[A] -requires-> [B] -escalates_to-> [C]`) plus matching community summaries.
- Execution: retrievers run in parallel (`ThreadPoolExecutor`, as today); each returns candidate chunk ids with source tag (`dense|bm25|graph`).

### 7.3 Fusion & rerank

- **RRF** (`src/nga/rag_agent/fusion.py`): `reciprocal_rank_fusion(lists: dict[str, list[str]], k=60) -> list[str]` — merge candidate chunk-id lists across retrievers; deterministic tie-break by chunk_id. No score calibration needed across heterogeneous retrievers.
- **Reranker seam** (`src/nga/rag_agent/rerank.py`): `Reranker.rerank(query, candidates: list[ChunkCandidate], *, user_level, top_n) -> list[ChunkCandidate]`.
  - Reference stand-in: `RrfOnlyReranker` (identity, returns RRF order) and `LlmReranker` (listwise LLM prompt over ≤ 15 candidates: "score relevance + logical continuity, cite ids"; OpenRouter/Ollama).
  - **Deferred:** dedicated cross-encoder service (BGE-reranker class / Cohere rerank). Interface unchanged.
  - Cut: `top_n` from tier context budget — tier1 3, tier2 4, tier3 5 (configurable).

### 7.4 Context assembly & token budget

`src/nga/rag_agent/context.py`: `assemble_context(*, query, reranked_chunks, graph_paths, community_summaries, tier) -> str`:
- Serialize graph traversal paths as compact logical chains: `[Entity A] -requires-> [Entity B] -escalates_to-> [Entity C] (source: SOP-…:chunk)`.
- Chunk blocks carry `[category][doc_id] §section_path` prefixes (existing format, now with real section paths).
- Token budget per tier enforced (tier windows from `Settings`); truncation policy: drop lowest-ranked chunks first, then community summaries, never the top-3 chunks or RBAC guarantees.
- Output goes through the existing tool-result path so LangGraph, cache, and logs are unchanged.

### 7.5 RBAC & cache

- RBAC enforced at dense filter, BM25 filter, graph BFS, reranker candidate set, and assembly — a chunk/node invisible at any earlier stage never appears later.
- L1 retrieval cache keys extended to include `intent` + strategy; cache scope per §4.5.

### Error handling & testing

- Retriever failure ⇒ that retriever contributes no candidates (logged); RRF over surviving lists; total failure ⇒ existing loud dimension-mismatch diagnostics.
- Intent classifier never raises (regex-only); unknown cues default `multi_hop`.
- Tests: intent table unit cases (incl. hierarchy vs lookup vs multi-hop adversarial examples); BM25 returns exact-code hits dense misses on planted queries (`QCR-501`, `SOP-OPR-101`); RRF merge order + determinism; RBAC leak tests at each stage (a level-4 chunk never appears for `user_level=1` in any path); reranker cut to `top_n`; context assembly token cap; graph path serialization format.

### Seams / deferred

Cross-encoder service, scale vector DB (Qdrant-class), learned router. All behind the seams above.

## 8. Workstream D — Eval v2 & Stepped Rollout (spec-ready)

**Goal:** tier the benchmark, measure retrieval independently of the LLM answer, and gate every prompt/embedding/reranker/graph change on a stepped L1→L4 protocol.

### 8.1 Tier crosswalk

Add `tier ∈ {L1, L2, L3, L4}` to `eval-questions/questions.json` entries (secondary axis; existing `category` retained). Mapping rule applied to the current 85:
- L1 single-hop fact: all `retrieval` questions whose answer lives in one `source_doc` section (30).
- L2 2-hop: `multi-hop` questions resolvable within one doc family or two linked docs (subset of 12).
- L3 multi-document reasoning: remaining `multi-hop`, `scenario`, `escalation-recall`, and `sql`-adjacent scenario questions requiring ≥ 2 documents/DB+doc synthesis.
- L4 cross-domain adversarial: curated subset incl. conflict-detection (variant corpus), stress/ambiguity, and cross-policy compliance questions.
- `difficulty` field retained. Tier labels are reviewed against the `expected_answer`/`source_docs` ground truth during implementation; any question that cannot be labeled honestly is flagged (not force-labeled).

### 8.2 Evidence-hit harness

For questions with `source_docs` (doc IDs), the retrieval layer must surface chunks from those docs. Harness runs each question's query through the **retrieval pipeline only** (no LLM synthesis) and computes:
- `evidence_hit_rate@k` = fraction of required doc IDs present in the top-k chunk set (per tier and category);
- `rerank_lift` = hit-rate before vs after rerank;
- `strategy_share` = which retrievers contributed the hits (dense/BM25/graph).
Required set resolution: `source_docs` doc IDs → chunk IDs via the registry manifest `files[].chunk_ids` (from Workstream A). This is what makes "benchmark every prompt/embedding/reranker change" concrete and cheap (no LLM judge needed for retrieval changes).

### 8.3 Stepped rollout protocol

A change to chunker/embedding/graph/rerank/context assembly must:
1. Run the evidence-hit harness + end-to-end eval on **L1** (fast) — any regression aborts.
2. Promote to **L2**, then **L3**, then **L4** — regression at any tier aborts and triggers rollback (registry pointer or config revert).
3. CI records tier-level pass-rate deltas (existing `ci_tracker` + comparison studio extended with tier axis and harness metrics).
Acceptance: no tier regresses below the current CI threshold (≥ 70% overall pass, existing) *and* no evidence-hit-rate regression on any tier.

### 8.4 Tests & files

- `eval-questions/questions.json`: `tier` field added (curated mapping committed with rationale note).
- `src/nga/evaluation/evidence_harness.py` (new): pure retrieval harness + report schema.
- `src/nga/evaluation/report.py`, `scoring.py`, `ci_tracker.py`: tier-aware aggregation; harness metrics in run records; comparison emits `tier_deltas`.
- Tests: tier completeness (every question labeled L1–L4, no overlap ambiguity); harness hit-rate correctness on a planted query set; CI delta rendering with tiers.

## 9. Workstream E — Agentic Hardening & MCP Seam (spec-ready)

**Goal:** formalize the tool seam as an MCP gateway, add claim-level citation enforcement, extend HITL audit to ingestion review, without changing the LangGraph core (already stateful: loops, branches, shared state).

### 9.1 Tool seam / MCP

`src/nga/tools/seam.py`:
- `ToolDescriptor = dataclass(name, description, args_schema, rbac_level, category)` — MCP-tool-compatible shape.
- `MCPGateway` seam: `register_descriptor`, `call(name, args, *, user_level) -> Any`, `list_tools(user_level) -> list[ToolDescriptor]`.
- Reference implementation: existing `search_sop_documents` and `query_nga_database` wrapped as descriptors behind the gateway (no behavior change; RBAC enforced inside tools as today).
- **Deferred:** real MCP server adapters (external DBs, issue trackers) — connect via a standard MCP client when the deployment has servers; gateway interface is the contract.
- LangGraph tool node binds the gateway's surfaced tools (existing `ToolNode` + tool-loop caps unchanged).

### 9.2 Claim-level citation guardrail

- Synthesis policy (§ existing `_build_response_policy` prompt) hardened: every `findings`/`evidence` item must cite a `[chunk_id]`/`[node_id]`/SQL-table that appears in the provided context; explicit instruction to state "not found in provided sources" when information is absent rather than assuming.
- Structural post-check in `synthesis` node: `FinalAnswer.evidence` citations are validated against the set of context source ids passed to the model; ungrounded evidence is dropped and flagged (`ungrounded_citations` recorded) before persistence. Extends existing `_is_ungrounded` (which only checks whether any tool evidence ran).
- Audit: each persisted answer stores the claim→source map (extension of `decisions_log` / answer records) → "glass-box" trail.

### 9.3 HITL extension for ingestion

- `pending_review` graph elements (Workstream B) surface in the existing approval queue UI + API (`app_state`-backed), with reviewer actions `approve | reject | edit`; actions logged with approver identity into `decisions_log` (new `action_type="ingestion_review"`).

### 9.4 Tests

- Gateway descriptor registry + RBAC surfacing; citation post-check rejects fabricated `chunk_id`s and accepts provided ones; UI queue serves ingestion-review items; existing orchestrator/UI/CI tests keep passing (gateway is behavior-preserving).

## 10. Incremental Ingestion (delta pipeline)

Trigger: batch `nga.ingestion.delta_ingest` entrypoint (mirrors existing mains) run on corpus change (CI hook or file-watch stub; streaming bus deferred). The pipeline operates strictly on generations; served state is never mutated in place.

1. **Scan & diff (no LLM):** recompute per-file `doc_hash`; diff against the current manifest's `files[]`. Output `added / modified / removed`. Unchanged files cost one hash read.
2. **Chunk-level re-index:** re-chunk only touched files (header-aware splitter, §5). Content-addressed `chunk_id`s yield vanished/new chunk sets at chunk granularity. Chroma: delete vanished ids, upsert new (no full rebuild). BM25 index updated identically.
3. **Localized KG delta (B):** only changed chunks feed candidates → refinement → alignment (§6). Token cost scales with Δ-documents, never corpus size.
4. **Affected-element supersession:** elements whose `source_chunks[]` intersect the change are kept (reweighted), `superseded` (only supported by removed text), or re-derived.
5. **Confidence gate + circuit breaker:** new elements with `confidence < θ` or contradicting committed data → `status=pending_review`, HITL review queue (§6/§9.3). If the delta's sampled audit precision falls below a floor, the *entire* batch is quarantined (no partial merge).
6. **Publish:** draft generation N+1 (vector store copy, BM25, graph, communities, manifest with changelog) → atomic `current.json` flip (§4.4). In-flight queries finish on N; new queries read N+1 (per-request registry read ⇒ lock-free swap).
7. **Cache:** new `corpus_version` ⇒ old cache keys unaddressable by construction (§4.5). Rollback = pointer flip to N (immutable snapshots retained).
8. **Refused deltas:** `config_hash` mismatch (chunker/embedding/schema/extraction-model change) forces full re-ingest — delta path aborts with a clear message.

Quality metric feeds: `quarantine_rate`, review-queue age, delta latency, `extraction_precision` per delta (sampled audit).

## 11. Sequencing, Dependencies, Rollback

Recommended order (each is its own implementation plan, gated by §8 harness where relevant):

1. **A — Anchored ingestion v2** (contracts + registry; everything depends on the metadata schema and chunk identity).
2. **B — KG construction v2** (needs A's chunk provenance; independent of C).
3. **D — Eval v2 harness** (before C so C changes are measured; needs A's manifest for evidence-hit resolution).
4. **C — Retrieval stack v2** (benchmarked by D; graph-path serialization can ship against B's graph or degrade to existing graph output).
5. **E — Agentic hardening & MCP seam** (orthogonal polish; can start after A).

Dependency graph: `A → {B, D}`; `B/D → C`; `E ∥ rest` (after A).

Rollback: per workstream via registry pointer flip (A+), config revert (pre-registry), and git history. Cache invalidation automatic via corpus version. CI stays green at every step (each plan ends with full `uv run pytest` + lint + a smoke eval run).

## 12. Deferred / Seam Summary

| Capability | Seam location | Full-fidelity target | Reference impl (ships) |
|---|---|---|---|
| NER/RE candidate extraction | `graphkg.extract.CandidateExtractor` | GLiNER/spaCy model service | `NgaRuleExtractor` (regex/dictionary) |
| Triple refinement | `graphkg.extract.TripleRefiner` | fine-tuned small LLM | LLM few-shot (OpenRouter/Ollama) |
| Entity alignment | `graphkg.extract.EntityAligner` | embedding + LLM-confirm service | `CooccurrenceAligner` (string + co-occurrence) |
| Reranker | `rag_agent.rerank.Reranker` | cross-encoder service (BGE/Cohere) | RRF order / `LlmReranker` listwise |
| Intent router | `rag_agent.intent.classify_intent` | learned/LLM router | heuristic cue classifier |
| MCP servers | `tools.seam.MCPGateway` | standard MCP client to real servers | existing tools as descriptors |
| IndexRegistry storage | `ingestion.registry.IndexRegistry` | object store + catalog | filesystem generations + `current.json` |
| Scale vector DB | `Retriever` seam | Qdrant/Weaviate | Chroma |
| Streaming ingestion | delta pipeline trigger | message bus / file watcher | batch `delta_ingest` entrypoint |

## 13. Acceptance Criteria (per workstream, mapped to eval)

- **A:** Full metadata contract on every chunk; deterministic rebuild (identical `chunk_id` set for unchanged corpus); registry generation published with content-derived corpus version; all existing tests + new chunking/registry/metadata tests green.
- **B:** Graph elements carry provenance + confidence; golden-subset precision/accuracy reported; delta edit produces correct `superseded` set; low-confidence elements land in `pending_review`; community summaries populated and RBAC-pruned.
- **C:** Intent routing classifies the planted hierarchy/lookup/multi-hop cases correctly; BM25 recovers exact-code queries dense misses; RBAC leak tests pass at every stage; evidence-hit rate on L1/L2 does not regress (measured by D).
- **D:** Every question labeled L1–L4; harness metrics recorded in CI; stepped protocol documented and enforced for retrieval-affecting changes.
- **E:** Gateway surfaces exactly the RBAC-visible tools; fabricated-citation rejection verified; ingestion-review items flow through the UI queue with audit log.

Overall: repo green (`uv run pytest`, `ruff`), CI eval above the ≥ 70% threshold, and each workstream's docs/plans committed alongside code.
