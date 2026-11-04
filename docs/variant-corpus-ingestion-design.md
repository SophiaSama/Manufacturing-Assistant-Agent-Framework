# Design & Implementation Plan — Variant Corpus Ingestion

**Status:** Proposed
**Owners:** Platform/Agent team
**Related:** `TEST_REVIEW_SUMMARY.md` item 3, `docs/cache-design.md`, `src/nga/ingestion/`
**Problem:** `tests/integration/test_conflict_detection.py` cannot detect planted
inconsistencies because `variant-corpus/` is **not ingested** — the agent only sees
canonical values, so "conflict detection" tests silently pass via the canonical fallback.

---

## 1. Context

The ingestion pipeline (`src/nga/ingestion/build_vector_store.py`) scans a fixed set of
canonical corpus roots and writes a single Chroma collection:

```
discover_documents(base_dir)
  └─ corpus_roots = operator-sops, technician-sops, machine-details,
                    failure-analysis, recall-quality, additional-docs
  └─ skips EXCLUDED_FILES (ground-truth.md, planted-inconsistencies.md, ...)
      └─ build_documents() → chunks with RBAC metadata
          └─ Chroma.from_documents(collection="nga_reference_docs")
```

`variant-corpus/` mirrors the same folder layout but is **not** in `corpus_roots`,
so none of its 8 planted inconsistencies reach the vector store. The conflict test
docstring already documents the intent:

> "To run conflict detection, build the vector store with BOTH the canonical corpus
> AND the variant corpus. The agent should then surface conflicts."

`build_graph.py` reads whatever collection is current, so GraphRAG follows the same
profile automatically.

---

## 2. Goals and Non-Goals

### Goals
1. Ingest `variant-corpus/` into Chroma so conflict-detection tests exercise real
   contradictory evidence.
2. Keep the **canonical** pipeline byte-for-byte unchanged (default profile).
3. Avoid retrieval noise: only *differing* variant files enter the conflict store.
4. Make the corpus selection explicit and testable (`CORPUS_PROFILE`).
5. Preserve RBAC metadata and doc IDs for variant docs.

### Non-Goals
1. Auto-diffing documents at query time (the diff happens at ingestion).
2. Shipping the variant corpus to production stores (it's a test fixture corpus).
3. Changing retrieval ranking/reranking logic.

---

## 3. Design

### 3.1 Corpus profiles

Three profiles, selected by one env var:

| Profile | Ingested | Collection / store dir | Use |
|---|---|---|---|
| `main` (default) | Canonical corpus only | `nga_reference_docs` / `data/vector_store` | Production, benchmark |
| `variant` | Variant corpus only | `nga_variant_docs` / `data/variant_vector_store` | Isolation testing of variant behavior |
| `conflict` | Canonical + **only differing** variant files | `nga_conflict_docs` / `data/conflict_vector_store` | Conflict-detection integration tests |

### 3.2 Content-hash diff (the key mechanism)

For the `conflict` profile, each variant file is compared to its canonical counterpart
by **content hash**:

```
canonical_hash(f) = sha256(canonical_text(f.read_text()))
variant  file     = variant-corpus/<relpath>

include variant file iff sha256(variant) != sha256(canonical counterpart)
```

- The 8 planted files differ → included, tagged `corpus_source="variant"`.
- The 8 unchanged copies are byte-identical → **excluded** → zero noise on
  non-conflicting topics (retrieval on "brake fluid" returns only the canonical doc).
- Files with no canonical counterpart (none today) → included as `variant` (fail-open).

### 3.3 Metadata additions

| Field | main | variant | conflict |
|---|---|---|---|
| `corpus_source` | `canonical` | `variant` | `canonical` or `variant` |
| `chunk_id` | `{doc_id}:c{i}` | `{doc_id}:c{i}:v` | `{doc_id}:c{i}:c` / `{doc_id}:c{i}:v` |

`chunk_id` must be globally unique — the source suffix prevents collisions between
the two copies of `SOP-OPR-101` (same `doc_id` from filename).

### 3.4 Retrieval surface

`retrieval_tool._search_one_category` already returns a result dict per chunk; add
`corpus_source` to it, and let `format_retrieval_results` render it:

```
[QCR] [VARIANT] [QCR-501] §3 ...   ← variant evidence is visibly distinct
[QCR] [QCR-501] §3 ...             ← canonical evidence
```

The LLM then *sees* both 105 Nm and 108 Nm with provenance tags, which is exactly
what makes conflict flagging possible. The payload `references` list also gains
`corpus_source`.

### 3.5 Config

| Env var | Default | Meaning |
|---|---|---|
| `CORPUS_PROFILE` | `main` | `main` \| `variant` \| `conflict` |
| `VARIANT_CORPUS_DIR` | `variant-corpus` | Location of variant corpus |
| `VARIANT_VECTOR_STORE_DIR` | `data/variant_vector_store` | Override for `variant` profile |
| `CONFLICT_VECTOR_STORE_DIR` | `data/conflict_vector_store` | Override for `conflict` profile |

`Settings` gains: `corpus_profile`, `variant_corpus_dir`, `variant_vector_store_dir`,
`conflict_vector_store_dir`. Resolution:
`store_dir = getattr(settings, f"{profile}_vector_store_dir")` for variant/conflict,
else `vector_store_dir`.

### 3.6 GraphRAG

`build_graph.py` reads the Chroma collection named by the current profile. It needs a
`--profile` CLI arg (or reads `Settings.corpus_profile`) and uses the profile's store
dir + collection name. No graph-algorithm changes.

### 3.7 Why a separate collection instead of merging into the canonical one?

- **No risk of polluting production retrieval** with variant copies.
- Profile switching is a store-dir swap, not a destructive rebuild of the canonical store.
- CI can build the conflict store once and reuse it; the canonical store is untouched.

### 3.8 Verified assumption: RBAC works for variant paths

`rbac.FOLDER_LEVEL_MAP` keys use trailing-slash substrings (e.g. `"operator-sops/"`,
`"recall-quality/"`). `level_from_path` matches `prefix in path`, so
`variant-corpus/operator-sops/...` resolves the same levels as the canonical paths.
**No changes required in `rbac.py`** — verified against the current implementation.

---

## 4. Implementation Plan

### Phase 1 — Refactor discovery (no behavior change)

**`src/nga/ingestion/build_vector_store.py`**
1. Add `CORPUS_PROFILES` + profile→collection-name map.
2. Extract the root list into `canonical_roots()` and `variant_roots()`.
3. Add `profile_roots(profile) -> list[Path]`.
4. Add `_variant_diff_map(base_dir, variant_dir) -> dict[relpath, bool]` (content-hash diff).
5. Extend `discover_documents(base_dir, profile, variant_dir)`:
   - `main` → canonical roots only
   - `variant` → variant roots only
   - `conflict` → canonical roots + differing variant files
6. `build_documents(..., profile)` sets `corpus_source` + suffixed `chunk_id`.
7. `build_vector_store(settings, base_dir, profile=None)` resolves collection +
   store dir from profile (default from `Settings`).
8. `main()` gains `--profile` CLI arg.

**Tests (unit):** `tests/test_ingestion.py`
- main profile: unchanged behavior (16 docs, `corpus_source=canonical`).
- conflict profile: exactly 8 extra chunksets with `corpus_source=variant`; the 8
  identical copies excluded; `chunk_id`s unique.
- RBAC metadata preserved on variant docs.

### Phase 2 — Retrieval surface

**`src/nga/tools/retrieval_tool.py`**
- Add `corpus_source` to `_search_one_category` results and `build_retrieval_payload`
  references.
- `format_retrieval_results`: render `[VARIANT]` tag when `corpus_source == "variant"`.

**Tests:** `tests/test_retrieval_formatting.py` — tag appears for variant results,
absent for canonical.

### Phase 3 — Graph + CLI

**`src/nga/ingestion/build_graph.py`**
- `--profile` arg; read the profile's collection/store dir.

**`src/nga/cli.py` / `src/nga/ui/server.py`**
- Load the store per `Settings.corpus_profile` (prod stays `main`).

### Phase 4 — Test harness wiring

**`tests/conftest.py`**
- New session fixture `conflict_agent_graph(settings, tmp_path_factory)` that:
  1. Builds/loads the `conflict` profile store (lazy, one-time).
  2. Builds the orchestrator exactly like `agent_graph` but with the conflict store.
- `tests/integration/test_conflict_detection.py` switches its fixture to
  `conflict_agent_graph`.

**Build command for CI:**
```bash
CORPUS_PROFILE=conflict uv run python -m nga.ingestion.build_vector_store --profile conflict
uv run pytest tests/integration/test_conflict_detection.py -v -s
```

---

## 5. Test Plan

| Test | Assertion |
|---|---|
| Unit: `discover_documents(main)` | Same 16-doc set as today |
| Unit: `discover_documents(conflict)` | 16 canonical + 8 variant files |
| Unit: diff map | Exactly the 8 planted files differ; hash-stable |
| Unit: chunk_id uniqueness | No collisions across sources |
| Integration (conflict store + strong model) | Agent flags ≥ 1 conflict keyword OR returns canonical value for all 8 scenarios |

## 6. Acceptance Criteria

1. `CORPUS_PROFILE=main` build → byte-identical corpus to today (verify chunk count).
2. `CORPUS_PROFILE=conflict` build → canonical + exactly 8 variant sources; chunk count
   = canonical + sum(8 variant files' chunks).
3. `planted-inconsistencies.md` never ingested in any profile.
4. Conflict tests use the conflict store (fixture swap), no longer the canonical store.
5. Full unit suite green; canonical prod path unchanged.

## 7. Decisions Log

| # | Decision | Rationale |
|---|---|---|
| D1 | Profiles via env var, default `main` | Zero change for prod; explicit test opt-in |
| D2 | Content-hash diff excludes identical variant copies | Conflict retrieval stays clean; only planted contradictions add evidence |
| D3 | Separate collection per profile | No pollution of canonical store; swap = directory change |
| D4 | `corpus_source` + suffixed `chunk_id` metadata | Provenance visible to LLM; uniqueness guaranteed |
| D5 | Graph follows profile | One code path; GraphRAG evidence stays consistent with store |
