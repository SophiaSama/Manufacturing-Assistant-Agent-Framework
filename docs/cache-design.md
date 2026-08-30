# Cache Module Design — NGA Manufacturing Assistant

**Status:** Proposed
**Author:** Platform/Agent team
**Applies to:** `src/nga/**`, `tests/**`
**Related:** `database/nga.db`, `data/vector_store`, `data/app_state.db`, `eval-questions/*`

---

## 1. Context

The NGA Manufacturing Assistant is a LangGraph RAG agent (`src/nga/graph/orchestrator.py`):

```
prepare → agent ⇆ tools → synthesis → hitl → END
```

The `agent` node calls two tools:

| Tool | Implementation | Cost driver |
|---|---|---|
| `search_sop_documents` | Hybrid Chroma retrieval + GraphRAG evidence (`tools/retrieval_tool.py`) | Embedding API calls (query embedding + any re-embedding), vector search over 8 document categories, graph walk |
| `query_nga_database` | Validated read-only text-to-SQL (`tools/sql_tool.py`) | SQLite query execution + repeated LLM tool-call rounds when the same data is re-queried |

The evaluation harness (`evaluation/eval_runner.py`) runs the same graph per question with a fresh `thread_id` and scores each answer. `data/app_state.db` already stores decisions; the checkpointer (`memory/checkpointer.py`) persists graph state via `langgraph-checkpoint-sqlite`.

**Problem:** repeated queries (real users asking the same thing across shifts; eval suites re-running similar questions) recompute embeddings, vector searches, graph walks, and SQL plans from scratch. This costs money (embedding/LLM API), latency, and provider rate-limit headroom.

**Out of scope:** LLM prompt caching is handled by the provider (OpenRouter/Anthropic cache the system-prompt + history prefix server-side). This module caches **deterministic, corpus/data-dependent pipeline results** only.

---

## 2. Goals and Non-Goals

### Goals
1. Cache embedding results (text → vector) so duplicate text is never re-embedded.
2. Cache retrieval results (query → RBAC-filtered chunks + graph evidence) keyed so identical/near-identical queries skip vector + graph search.
3. Cache SQL query results for identical validated SQL statements.
4. Keep **eval runs cache-cold by default** so scores measure the true pipeline, with an explicit `--cache-hot` mode for realistic repeated-query measurements.
5. Make invalidation deterministic: any document re-ingest or DB data change invalidates affected layers.
6. Provide hit/miss/staleness metrics to observe cost and latency savings.

### Non-Goals
1. Caching LLM completions (responses) — risks serving stale/unsafe answers (torque specs, recall criteria). Explicitly **not** implemented; see §7.5.
2. Distributed cache (Redis etc.) — single-instance deployment; SQLite is consistent with the existing stack.
3. Cross-session conversational memory — that's the checkpointer's job, not this cache.
4. Near-duplicate fuzzy matching — exact (canonicalized) key matching only; semantic dedup is a future enhancement (§9).

---

## 3. Cache Layers

Four layers, from cheapest to most expensive to recompute. Only layers whose outputs depend on **rarely-changing inputs** are cached.

```
        ┌─────────────────────────── pipeline ───────────────────────────┐
        │  L0 embedding   L1 retrieval   L2 SQL   L3 response (NOT cached)│
 query ─► [embed text] ─► [chroma + graph] ─► [sqlite select] ─► [llm] ─► answer
              ▲                ▲                  ▲               ▲
         text→vector     query→chunks       sql→rows      (excluded)
         deterministic   corpus+RBAC dep    DB-data dep    nondeterministic
```

| Layer | Caches | Key depends on | Invalidate on | Env |
|---|---|---|---|---|
| **L0 embedding** | `text → vector` | text, embedding model | model change | shared-safe (but keep namespaced) |
| **L1 retrieval** | `query → chunk refs + graph evidence` | canonical query, categories, `user_level`, k, corpus version | corpus re-ingest | eval/prod separated |
| **L2 SQL** | `normalized sql → rows` | normalized SQL, DB etag | DB etag change | eval/prod separated |
| **L3 response** | `prompt → answer` | full prompt | — | **not implemented** |

### 3.1 L0 — Embedding cache
- Where: `providers/factory.py` `make_embeddings()` wrapper; used by both ingestion (`ingestion/build_vector_store.py`) and query-time embedding.
- Key: `sha256(canonical_text)` + embedding model id.
- Canonicalization: strip trailing whitespace, normalize newlines.
- This is the *only* layer that may be shared between eval and prod (§5) because `text → vector` is deterministic given the same model — but we still keep namespaces separate for simplicity and to isolate corpus versions.

### 3.2 L1 — Retrieval cache
- Where: inside `tools/retrieval_tool.py` — wraps `retrieve_documents()` and `retrieve_graph_evidence()`.
- Key inputs: canonical query text, normalized category list, `k`, `user_level`, corpus version.
- **RBAC is part of the key.** A Level 4 query returns higher-clearance chunks (`level_rank` filter); serving those to a Level 1 user is a security bug. Keys must include `user_level` so levels never share entries.
- Value: the JSON-serializable retrieval payload (`build_retrieval_payload`) — chunk references (doc_id, chunk_id, category, section) + text + graph evidence.
- Size bound: `k × len(categories) × avg_chunk_bytes`; default k=5 × 8 categories ≈ 40 chunks max ≈ tens of KB. Acceptable.

### 3.3 L2 — SQL cache
- Where: inside `tools/sql_tool.py` `run_query()`.
- Key inputs: **normalized SQL** (via `sqlglot` AST → canonical string, so whitespace/quote variants collide) + **DB etag**.
- DB etag: a cheap fingerprint of the data the query touches. Computed as `hash(COUNT(*) per referenced table + MAX(ts) per referenced table)`, itself cached with a short TTL (e.g., 30 s) so it isn't computed per query.
- In the current seeded deployment the DB is static (eval) or changes rarely (prod); the etag makes staleness impossible rather than time-based.
- Security: SQL cache never bypasses `validate_select_only` — caching wraps an already-validated read-only query only.

### 3.4 L3 — Response cache
**Not implemented.** Answers depend on the LLM, injected evidence, role prompt, and model — nondeterministic and safety-relevant. Caching them risks serving a stale torque spec or recall decision. Rejected in §7.5.

---

## 4. Key Design

```
key = "{env}:{corpus_version}:{layer}:{rbac}:{sha256(canonical_input)}"
```

| Component | Example | Notes |
|---|---|---|
| `env` | `prod`, `eval` | Hard partition between environments (§5) |
| `corpus_version` | `cv-2025-04-10-a3f9` | Hash of `(file path, mtime, size)` over `documents_dir`; stored in a `meta` table at ingestion |
| `layer` | `emb`, `retr`, `sql` | Layer discriminator |
| `rbac` | `lvl4` | Required for `retr`; `lvl0` for `emb`; for `sql` this encodes the *caller's* clearance only if the tool will filter rows (currently it does not) |
| `payload hash` | `sha256(...)` | Canonicalized input (query text / SQL AST / text) |

`corpus_version` is central to invalidation: bumping it on re-ingest instantly orphans every `retr`/`emb` entry from the old corpus without a delete sweep.

---

## 5. Eval vs Production Separation

**Same cache implementation, separate instances, eval cache-cold by default.**

### 5.1 Storage
```
data/cache/
├── prod_cache.db     # env=prod  — served by the UI/orchestrator, cache ON
└── eval_cache.db     # env=eval  — used ONLY by explicit --cache-hot eval runs
```

Two files (or one file with a `namespace` column — two files is simpler to reason about and to delete). Config drives the path (§6).

### 5.2 Eval modes
| Mode | Behavior | Use case |
|---|---|---|
| `cold` (default) | Cache bypassed entirely — every question exercises embeddings, retrieval, SQL, LLM | Headline quality metric; regressions in `evaluation/compare_evaluation_runs` |
| `hot` | Cache enabled inside the eval DB | Measures realistic repeated-query latency/cost; answers must still be scored identically |

### 5.3 Why separate instances
1. **No score inflation:** with a shared cache, eval question #2 can be warmed by question #1's identical chunks → retrieval no longer measures the pipeline.
2. **No contamination:** eval-only queries/answers never pollute the prod cache (and vice versa).
3. **Clean corpus isolation:** eval may run against a different document snapshot than prod; keys embed `corpus_version` so even a shared store would not collide — but isolation also removes the blast radius of accidental invalidation.

---

## 6. Configuration

New `Settings` fields (`src/nga/config.py`) with `.env` defaults:

```python
cache_enabled: bool            # master switch; default False for Local, True for Cloud
cache_mode: str                # "cold" | "hot"; eval runner uses this
cache_db_path: str             # default "data/cache/nga_cache.db"  (prod)
cache_eval_db_path: str        # default "data/cache/eval_cache.db" (eval)
cache_ttl_emb_s: float         # default 86400 (1 day)
cache_ttl_retr_s: float        # default 3600
cache_ttl_sql_s: float         # default 300
cache_max_entries: int         # default 100_000 (eviction: LRU)
cache_db_etag_ttl_s: float     # default 30
```

Env vars: `CACHE_ENABLED`, `CACHE_MODE`, `CACHE_DB_PATH`, `CACHE_EVAL_DB_PATH`, `CACHE_TTL_*`, `CACHE_MAX_ENTRIES`.

---

## 7. Component Design

### 7.1 New package: `src/nga/cache/`

```
src/nga/cache/
├── __init__.py      # public API: NgaCache, make_key, CacheMode, cache_stats
├── store.py         # SQLite-backed KV with TTL + LRU eviction
├── keys.py          # canonicalization + make_key()
├── layers.py        # embedding / retrieval / sql cache adapters
└── metrics.py       # hit/miss counters (in-memory + persisted snapshots)
```

**`store.py` — SQLite KV:**
```python
@dataclass
class CacheEntry:
    key: str
    value: bytes          # json.dumps(...).encode()
    created_at: float
    expires_at: float | None
    size_bytes: int

class SqliteCacheStore:
    def __init__(self, db_path: str, *, max_entries: int = 100_000): ...
    def get(self, key: str) -> bytes | None      # deletes expired on read
    def set(self, key: str, value: bytes, ttl: float | None = None) -> None
    def delete_prefix(self, prefix: str) -> int  # invalidation sweep
    def size(self) -> int
    def clear(self) -> None
    def close(self) -> None

    # schema
    # CREATE TABLE cache(k TEXT PRIMARY KEY, v BLOB NOT NULL, exp REAL,
    #                    created REAL NOT NULL, size INTEGER NOT NULL);
    # CREATE INDEX idx_cache_exp ON cache(exp);
```

- WAL mode (consistent with `data/app_state.db`).
- LRU eviction: on insert when `size() >= max_entries`, delete oldest `created` rows in a batch.
- Reads are O(1) PK lookups; no in-process cache above SQLite (single instance, local disk — avoids dual-cache consistency problems).

**`keys.py` — canonicalization:**
```python
def canonical_text(s: str) -> str            # collapse whitespace, lowercase? (NO — retrieval is case-sensitive to doc text; only trim + normalize newlines)
def canonical_sql(ast) -> str                # sqlglot AST → canonical string
def make_key(*, env, corpus_version, layer, rbac_level, canonical: str) -> str
```

**`layers.py` — adapters (integration points in existing code):**
```python
def cached_embeddings(cache, model_id, corpus_version):
    """wrap embed_query/embed_documents to check cache first"""

def cached_retrieval(cache, corpus_version, user_level, *, categories, k):
    """decorator around retrieve_documents / retrieve_graph_evidence"""

def cached_sql(cache, db_etag_fn):
    """decorator around run_query: normalize SQL, check etag, cache rows"""
```

### 7.2 Integration points (no behavior change when cache off)

| File | Change |
|---|---|
| `src/nga/providers/factory.py` | `make_embeddings()` gains optional `cache`; `embed_query`/`embed_documents` check L0 |
| `src/nga/tools/retrieval_tool.py` | `retrieve_documents()` / `retrieve_graph_evidence()` accept optional `cache` + `corpus_version` |
| `src/nga/tools/sql_tool.py` | `run_query()` accepts optional `cache` + `db_etag` provider |
| `src/nga/tools/tool_factory.py` | constructs tools with cache when `settings.cache_enabled` |
| `src/nga/ingestion/build_vector_store.py` | writes/reads `corpus_version` into cache `meta` table; on re-ingest bumps version |
| `src/nga/evaluation/eval_runner.py` | `run_single_eval_question(..., cache_mode="cold"|"hot")`; default cold; per-layer timers feed the `cache_improvement` block |
| `src/nga/evaluation/report.py` / `ci_tracker.py` | `cache_mode` in reports; `cache_improvement` block + CI gate (§7.4.1) |
| `src/nga/cli.py` | eval CLI flag `--cache-mode cold|hot` |

Design rule: **cache is an optional decorator parameter, default off.** Every path works identically without the cache; tests can toggle it freely.

### 7.3 Invalidation
- **Documents:** ingestion writes `corpus_version = hash(sorted(file relpath, mtime, size))` to the cache `meta` table. Any change → new version → old keys become unaddressable. Old rows are swept lazily by `delete_prefix(f"{env}:{old_version}")` after ingestion.
- **SQL DB:** per-query `db_etag` (see §3.3) with a 30 s TTL on the etag computation. No manual invalidation needed.
- **Manual:** `python -m nga.cache.cli clear --env prod --layer sql` (ops escape hatch).
- **Embedding model change:** model id is part of the key → automatic partition.

### 7.4 Metrics
Per layer, per env, exposed via the existing app (`ui/server.py` health endpoint):
```json
{ "layer": "retr", "env": "prod", "hits": 4821, "misses": 913,
  "hit_rate": 0.84, "saved_ms_avg": 141, "entries": 924, "size_kb": 318 }
```
- `saved_ms_avg` measured from a one-time side-by-side latency sample per unique key shape.
- Eval reports (`evaluation/report.py`) get a `cache_mode` field so cold vs hot runs are distinguishable in `compare_evaluation_runs`.

### 7.4.1 Before/After Improvement Metric (paired cold/hot protocol)

A cache's value can only be measured against a baseline. The canonical protocol:

1. **Baseline run** — full eval suite with cache disabled: `--cache-mode cold`.
2. **Candidate run** — identical suite, same question order, cache enabled: `--cache-mode hot`.
3. `compare_evaluation_runs(baseline, candidate)` emits a new **`cache_improvement`** block:

```json
{
  "run_a": { "label": "baseline_cold", "cache_mode": "cold" },
  "run_b": { "label": "cached_hot", "cache_mode": "hot" },
  "cache_improvement": {
    "latency_reduction_pct": 38.4,
    "layer_latency_reduction_pct": { "emb": 61.2, "retr": 44.8, "sql": 12.3 },
    "cost_reduction_pct": 22.1,
    "quality_parity": { "score_delta": 0.0, "pass_delta": 0 },
    "hit_rate": { "emb": 0.91, "retr": 0.78, "sql": 0.35 },
    "warmup_effect_s": 2.1
  }
}
```

**Metric definitions (headline = latency_reduction_pct):**

| Metric | Formula | Notes |
|---|---|---|
| `latency_reduction_pct` | `(1 − t_hot/t_cold) × 100` over the same question set | Headline improvement; report p50/p95 too |
| `layer_latency_reduction_pct` | `(1 − layer_time_hot/layer_time_cold) × 100` per layer | Requires per-layer timing instrumentation (L0/L1/L2 timers around tool execution) |
| `cost_reduction_pct` | `(1 − cost_hot/cost_cold) × 100` | Cost = embedding API calls (per-call $) + LLM tokens; LLM tokens shrink only when fewer tool rounds occur — measure, don't assume |
| `hit_rate` | `hits / (hits + misses)` per layer | Weighted efficacy = Σ(hit_i × layer cost share) |
| `quality_parity` | `score_hot − score_cold`, `pass_hot − pass_cold` | Must be ≥ 0 (or within noise); answers must not degrade |
| `warmup_effect_s` | latency of the first hot-run question vs its cold latency | Quantifies the cold-start cost per session (cache is per-process) |

**Production (no eval runs): always-on baseline sampling.** Serve ~5% of prod queries with cache bypassed ("canary misses", keyed in metrics as `baseline`). A rolling window (last 1,000 queries) continuously reports the same `latency_reduction_pct` and `cost_reduction_pct` against the cached majority. This detects cache degradation (e.g., invalidation storms dropping hit rate) without separate runs.

**CI gate (success criterion for the cache implementation):**
- `latency_reduction_pct ≥ 20%` (p50, full suite)
- `quality_parity ≥ 0` (no score/pass regressions vs cold)
- per-layer `hit_rate > 0.5` on the eval suite for `emb` and `retr`

Failing any gate fails the cache-related CI check (extend `evaluation/ci_tracker.py`).

### 7.5 Rejected: L3 response cache (rationale)
- Answers embed nondeterministic LLM behavior, injected evidence, and role prompts.
- A stale cached answer on a recall/Class-A question is a safety hazard, not just a quality issue (QCR-501, ESC-402 semantics).
- The agent already has an internal "do not re-query facts" guard (`orchestrator._summarize_tool_results`) — that's the right mechanism for repeated data within a turn, not a cross-turn answer cache.
- **Decision:** L3 is out of scope; revisit only with explicit per-question TTL, citation hash, and a human-approval gate.

---

## 8. Testing Plan

### Unit (`tests/test_cache.py`)
- Key canonicalization: whitespace variants collide; `user_level` differences do not.
- Store: set/get/expiry/eviction/LRU ordering/delete_prefix.
- L0: same text embedded once (mock embedder count).
- L1: same query + level → 1 retrieval call; different level → 2 calls (RBAC isolation test).
- L2: SQL whitespace/case variants hit same entry; etag bump → miss; `validate_select_only` still enforced before cache lookup.

### Integration
- **Cold-eval parity:** run the full suite with cache disabled before/after the change → scores identical (no behavior drift). This is the acceptance gate for merging.
- **Hot-eval check:** `--cache-mode hot` → same `passed`/`score` as cold for all 75 questions (answers must not change, only latency), except where a question legitimately repeats another's exact query (document as known behavior).
- **RBAC leak test:** Level 1 query must never return Level 4-cached chunks (key includes `lvl`).
- **Invalidation test:** re-ingest one document → retrieval cache for the old `corpus_version` misses; fresh queries hit the new version.
- **Stress:** existing `tests/integration/test_stress.py` run with hot cache → assert p95 latency drops while pass rate is unchanged.

### Acceptance criteria
1. `pytest` green; cold-eval parity: all existing eval scores unchanged (diff = 0 regressions in `compare_evaluation_runs`).
2. Hot-eval: hit_rate > 0 for repeated queries; identical answers to cold run.
3. RBAC isolation: no cross-level cache entries observable.
4. No new failure mode when `nga.db` or corpus changes mid-run (etag/corpus_version handle it).
5. **Improvement gate (§7.4.1):** paired cold/hot suite shows `latency_reduction_pct ≥ 20%`, `quality_parity ≥ 0`, and `emb`/`retr` hit rates > 0.5.

---

## 9. Future Enhancements
- **Semantic near-duplicate keys** (embedding-similarity prefix trees) to catch paraphrased repeat queries — heavier, needs its own eval gate.
- **Shared L0 embedding cache** across eval/prod once corpus snapshots are version-locked.
- **Tier-aware retrieval caching** keyed by `rag_tierX_max_hops` (GraphRAG hop depth affects cost most).
- Distributed store (Redis) if multi-instance deployment appears.

---

## 10. Decisions Log

| # | Decision | Rationale |
|---|---|---|
| D1 | SQLite-backed store, not Redis | Single instance; consistent with `app_state.db` / checkpoint-sqlite; zero infra |
| D2 | Eval cache-cold by default | Scores must measure the pipeline, not cache reuse |
| D3 | Separate eval/prod DBs | Prevents contamination + score inflation |
| D4 | RBAC level in every retrieval key | Prevents privilege escalation via cache |
| D5 | `corpus_version` / `db_etag` in keys, not TTL alone | Deterministic invalidation; TTL is a backstop, not the mechanism |
| D6 | No L3 response cache | Safety hazard for recall/Class-A content |
| D7 | Cache is an optional decorator, default off | Zero behavior change when disabled; easy cold/hot toggling |
| D8 | Improvement is measured by paired cold/hot runs (eval) + ~5% canary-miss sampling (prod), gated in CI | A cache without a baseline protocol is unverifiable; paired runs reuse `compare_evaluation_runs` |
