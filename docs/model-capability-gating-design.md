# Design & Implementation Plan — Model Capability Gating for Integration Tests

**Status:** Proposed
**Owners:** Platform/Agent team
**Related:** `TEST_REVIEW_SUMMARY.md` item 4, `tests/conftest.py`, `tests/integration/`
**Problem:** integration tests (benchmark 85 tests, conflict detection 8 tests) run
against whatever model is configured. With `llama3.2:3b` (local) they hallucinate
column names, time out, and fail — 7/8 conflict scenarios failed and benchmark
produced noise. CI needs to **skip** capability-sensitive suites on models that
cannot execute them, with a clear reason.

---

## 1. Context

- Model selection is environment-driven (`src/nga/config.py`):
  - `Local` → `OLLAMA_CHAT_MODEL` (e.g., `llama3.2:3b`)
  - `Cloud` → `OPENROUTER_MODEL` / tier models (`anthropic/claude-*`)
- Integration suites require:
  - **Benchmark** (`tests/integration/test_benchmark.py`): multi-hop reasoning, tool
    use, structured `FinalAnswer` JSON, SQL schema awareness → needs a **strong** model.
  - **Conflict detection** (`tests/integration/test_conflict_detection.py`): cross-doc
    contradiction reasoning, provenance-aware answers → needs the **strongest** tier.
- `pytest` already has a `--judge` gate for the LLM-as-judge scorer (conftest).

---

## 2. Goals and Non-Goals

### Goals
1. Define a small capability tier model (not a per-model allowlist sprawl).
2. Auto-detect the active model's tier from `Settings` (name-based heuristic) with an
   explicit override.
3. Auto-skip integration suites whose required tier exceeds the active model's tier,
   with a descriptive skip reason.
4. Keep unit tests and cache tests unaffected (they never require an LLM).
5. Provide an escape hatch to force-run (CI with real quota, manual runs).

### Non-Goals
1. Benchmarking model quality (that's the eval suite's job).
2. Router/tiering changes to the agent (`RAG_TIER*` routing is orthogonal).
3. Auto-downloading models.

---

## 3. Design

### 3.1 Capability tiers

| Tier | Meaning | Required for |
|---|---|---|
| `basic` | Single-hop lookup, no strict structured output | — |
| `standard` | Multi-hop retrieval + tool use + structured `FinalAnswer` JSON | Benchmark |
| `strong` | + cross-document contradiction reasoning, schema-aware SQL | Conflict detection |

Monotonic: `basic < standard < strong`.

### 3.2 Tier resolution

Pure function in a reusable module: `src/nga/evaluation/capability.py`

```python
def resolve_tier(model_id: str | None, *, provider: str) -> str:
    # 1. explicit override
    # 2. known cloud allowlist (substring match):
    #    claude-haiku → standard; claude-sonnet/opus, gpt-4o/4.1, gemini-1.5/2.5-pro → strong
    #    default cloud unknown → standard (conservative pass)
    # 3. local: parse parameter size from name regex (\\d+(?:\\.\\d+)?[bB])
    #    < 4B → basic | 4B..7B → standard | ≥ 7B → strong
    #    (evidence: llama3.2:3b at 3.21B fails integration, so <4B is gated out)
    # 4. unknown → basic (conservative skip)
```

- **Cloud unknown → `standard`** (fail-open: benchmark runs, conflict gates on `strong`).
- **Local unknown size → `basic`** (fail-closed: integration auto-skips).

Runtime probe (optional): when `--probe-capability` is set, run one cheap
self-check (e.g., `"Return JSON {\"ok\": true}"`) and upgrade/downgrade the tier.
Name parsing is the default — deterministic and free.

### 3.3 Pytest integration

New helper in `src/nga/evaluation/capability.py`:

```python
def require_tier(required: str) -> pytest.MarkDecorator
# skipif(active_tier < required, reason="...")
```

`tests/conftest.py`:
- `active_capability` fixture → `resolve_tier(settings provider/model, override)`.
- Each integration module declares its requirement:

```python
# tests/integration/test_benchmark.py
pytestmark = [pytest.mark.integration, require_tier("standard")]

# tests/integration/test_conflict_detection.py
pytestmark = [pytest.mark.integration, require_tier("strong")]
```

Skip output example:
`SKIPPED [85] — model llama3.2:3b resolves to tier 'basic'; benchmark requires 'standard'`

### 3.4 Config / CLI

| Knob | Mechanism | Default |
|---|---|---|
| `MODEL_CAPABILITY_OVERRIDE` | env var (e.g., `strong`) | unset → auto-detect |
| `--probe-capability` | pytest flag → runtime probe | off |
| Tier constants | module constants `TIER_RANK` | — |

### 3.5 Why name-based + override (not only probing)

- Probing costs a model call per session and adds flakiness to CI.
- Name heuristics are deterministic, free, and cover the two real cases (known Cloud
  allowlist, local param-size parsing).
- Override handles the rest (custom providers, fine-tunes).

---

## 4. Implementation Plan

### Phase 1 — Capability module

**New file `src/nga/evaluation/capability.py`**
- `TIERS = ("basic", "standard", "strong")`, `TIER_RANK` map.
- `parse_local_params(model_id) -> float | None` (regex).
- `cloud_allowlist_tier(model_id) -> str | None`.
- `resolve_tier(model_id, provider, override=None) -> str`.
- `require_tier(required) -> pytest.MarkDecorator` (skipif with reason).
- `probe_capability(settings) -> str` (optional runtime check).

**Unit tests `tests/test_capability.py`**
- `llama3.2:3b` → `basic`; `llama3.1:8b` → `strong`; `qwen2.5:7b` → `strong`.
- `anthropic/claude-haiku-4-5` → `standard`; `claude-sonnet-4-5`/`opus` → `strong`;
  `gpt-4o` → `strong`; unknown cloud → `standard`; unknown local → `basic`.
- Override wins over detection.
- Skip marker reason string is descriptive.

### Phase 2 — Conftest wiring

**`tests/conftest.py`**
- `active_capability` session fixture.
- Add `--probe-capability` option.

### Phase 3 — Module markers

- `tests/integration/test_benchmark.py`: `require_tier("standard")` + `--judge` gate
  (existing) — combined markers.
- `tests/integration/test_conflict_detection.py`: `require_tier("strong")`.

### Phase 4 — CI workflow

**`.github/workflows/ci.yml`**
- Add a job step: `MODEL_CAPABILITY_OVERRIDE` not set; run integration only when the
  job's model is Cloud (or an explicit `RUN_INTEGRATION=true` input).
- Keep unit tests unconditional.

---

## 5. Test Plan

| Test | Assertion |
|---|---|
| `resolve_tier` unit cases | Table-driven (see Phase 1) |
| Skip behavior | `pytest --collect-only` with `OLLAMA_CHAT_MODEL=llama3.2:3b` shows benchmark/conflict as skipped with reason |
| Forced run | `MODEL_CAPABILITY_OVERRIDE=strong` runs integration even on local model |
| Probe (optional) | `--probe-capability` upgrades/downgrades consistently |

## 6. Acceptance Criteria

1. `pytest tests/ -q` with default `.env` (Cloud, no key) → integration suites show
   `SKIPPED (requires tier ...)` instead of failing on API errors.
2. With `EXECUTION_ENVIRONMENT=Local`, `OLLAMA_CHAT_MODEL=llama3.2:3b` → benchmark +
   conflict auto-skip with reason mentioning the resolved tier.
3. Unit tests (incl. cache) unaffected — still green unconditionally.
4. `MODEL_CAPABILITY_OVERRIDE=strong` + working API → integration runs.
5. No change to the agent runtime; gating lives entirely in test configuration.

## 7. Decisions Log

| # | Decision | Rationale |
|---|---|---|
| D1 | Three tiers (`basic/standard/strong`) tied to suite requirements | Avoids per-model sprawl; suite declares what it needs |
| D2 | Name-based detection + explicit override; probe optional | Deterministic and free; probe only when wanted |
| D3 | Cloud unknown → `standard`, local unknown → `basic` | Fail-open for real cloud providers, fail-closed for weak local models |
| D4 | Gating in test config only | Runtime/agent code untouched; zero prod risk |
| D5 | Combined with existing `--judge` gate for benchmark | One clear mechanism per axis (judge vs capability) |
