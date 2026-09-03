# Real-Model Eval Review — 2026-09-03

**Project:** `/Users/ruiping/projects/manufacturing_agent`
**Model:** `deepseek/deepseek-v4-flash-0731` (OpenRouter)
**Embeddings:** `qwen/qwen3-embedding-4b` (2560-dim) — stores rebuilt to match
**Commit:** `82a5fc3` (+ conflict-store commits `6b8f242`, `02c9381`)

---

## 1. Executive Summary

The full pipeline (conflict corpus profiles → provenance retrieval → capability
gating → real LLM) now runs end-to-end with a real key. **Conflict-detection suite:
8/8 PASSED.** A bounded 10-question curated eval surfaced **3 real bugs** (all fixed)
before OpenRouter's weekly key limit interrupted it. A clean full rerun is needed
once the weekly budget resets.

## 2. Environment Notes (important)

| Item | Value |
|---|---|
| `.env` embedding model | `qwen/qwen3-embedding-8b` (4096-dim) |
| Stores built with | **4b (2560-dim)** — main & conflict rebuilt |
| ⚠️ Consistency rule | Store dimension MUST equal runtime `EMBEDDING_MODEL`. Any mismatch → silent retrieval failure (fixed to log loudly). |
| OpenRouter batch | qwen3-embedding has **no `:batch` endpoint** → must use `EMBEDDING_STRATEGY=sequential` |
| Weekly key limit | **Exhausted mid-run** (403) — full rerun blocked until reset |

**Recommendation:** standardize `.env` on `EMBEDDING_MODEL=qwen/qwen3-embedding-4b`
(works, fast, cheap) OR rebuild stores with 8b after the weekly reset. Mixed models
are the #1 failure source.

## 3. Results: Conflict Detection — 8/8 PASSED ✅

Runtime: 11m39s | model: deepseek-v4-flash | store: conflict profile (canonical + 8 variant sources)

| Scenario | Planted conflict | Result |
|---|---|---|
| CONFLICT-1 | Wheel torque 108 vs 105 Nm | ✅ |
| CONFLICT-2 | Electrode cap 5,000 vs 4,000 welds | ✅ |
| CONFLICT-3 | Calibration 6 vs 3 months | ✅ |
| CONFLICT-4 | Class A notify 4h vs 1h | ✅ |
| CONFLICT-5 | Defect threshold 1.0% vs 0.5% | ✅ |
| CONFLICT-6 | Rear mount 50 vs 45 Nm | ✅ |
| CONFLICT-7 | Bleed sequence FR→FL vs RR→RL | ✅ |
| CONFLICT-8 | Cure time 2h vs 4h | ✅ |

Pass criterion: agent flags the conflict **or** returns the canonical value with
evidence. These validate item 3 (corpus profiles) + item 4 (capability gate lets a
`strong` model run).

## 4. Results: Curated Eval (10 questions) — PARTIAL (blocked)

Report: `reports/eval/curated_real_10q.md` / `.json`

| ID | Category | Score | Latency | Verdict | Cause |
|---|---|---|---|---|---|
| R1 | retrieval | 0.67 | 175s | tools+evidence ✅, answer lost | HITL EOF crash *(fixed)* |
| R12 | retrieval | 0.67 | 119s | same | HITL EOF crash *(fixed)* |
| M1 | multi-hop | 0.67 | 158s | same | HITL EOF crash *(fixed)* |
| M3 | multi-hop | 0.67 | 175s | same | HITL EOF crash *(fixed)* |
| S1 | scenario | 0.67 | 307s | same | HITL EOF crash *(fixed)* |
| S5 | scenario | 0.00 | 0.4s | no run | 403 weekly limit |
| E1, E6 | escalation | 0.00 | 0.1s | no run | 403 weekly limit |
| SQL2, SQL10 | sql | 0.00 | 0.1s | no run | 403 weekly limit |

**Interpretation:** the 5 completed questions demonstrate correct pipeline behavior
(correct tools called: `query_nga_database` + `search_sop_documents`; document
sources found; real 119–307s latencies). Scores capped at 0.67 because the HITL
crash discarded the answer (`non_empty_answer: false`). **Scores will rise once the
rerun uses the fixed approver.** The 0.00 rows are API-budget failures, not agent
failures.

## 5. Bugs Found & Fixed (real value of this run)

| # | Bug | Impact | Fix (commit) |
|---|---|---|---|
| 1 | **Embedding dimension mismatch** (store 4b/2560 vs runtime 8b/4096) | Every query embedding failed → silent empty retrieval → agent answered blind | Loud ERROR diagnostics (`6b8f242`); stores rebuilt consistently |
| 2 | **HITL `input()` in non-TTY** | Every recommendation crashed CLI/eval AND UI server (EOFError) — answers lost | Pluggable approver; server uses safety-conservative `pending`, eval uses auto-approve (`82a5fc3`) |
| 3 | **Silent all-category retrieval failure** | Empty context returned with no error | Loud log + dimension-mismatch hint (`6b8f242`) |
| 4 | `.env` embedding model drifted from stores | Main store was 1024-dim (stale model) | Rebuilt stores; consistency rule documented |

## 6. Reviewable Artifacts

| Path | What |
|---|---|
| `reports/eval/curated_real_10q.md` | Official eval report (contaminated — see §4 caveats) |
| `reports/eval/curated_real_10q.json` | Machine-readable per-question data |
| `docs/variant-corpus-ingestion-design.md` | Item 3 design (implemented, conflict 8/8) |
| `docs/model-capability-gating-design.md` | Item 4 design (implemented, gate verified) |
| `TEST_REVIEW_SUMMARY.md` | Prior session summary |
| `curated_eval.py` | Driver for bounded real-model runs (10 curated questions) |

## 7. To Get a Clean Full Eval

1. Wait for OpenRouter weekly reset **or** use another key.
2. Optionally standardize: set `.env` `EMBEDDING_MODEL=qwen/qwen3-embedding-4b`
   (stores already match it).
3. Run:
   ```bash
   EMBEDDING_MODEL=qwen/qwen3-embedding-4b python curated_eval.py     # 10 curated
   EMBEDDING_MODEL=qwen/qwen3-embedding-4b python -m pytest tests/integration/test_benchmark.py -s  # full 75
   ```
   (HITL now auto-approves in eval; no more EOF crashes.)
4. Compare runs via the UI (`compare_evaluation_runs`) for regressions.

---

*Environment: Cloud | cache: cold | capability gate: active (deepseek-v4 → strong)*
