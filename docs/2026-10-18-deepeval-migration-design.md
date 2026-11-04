# DeepEval Migration & TypeSafe Jev Evaluation Architecture

## 1. Overview & Objective

This document outlines the migration of the benchmark and evaluation harness to **DeepEval (`>=4.3.1`)** using **TypeSafe Jev (`jev-1.13.0`)** as the judge for both standard RAG metrics and domain-specific criteria across all vertical domain packs.

The canonical evaluation implementation is housed in `src/enterprise_agent/evaluation/` (`judge.py`, `scoring.py`, `deepeval_adapter.py`, `deepeval_metrics.py`, `side_by_side.py`, `grounding.py`). All legacy modules under `src/nga/evaluation/` are preserved as **backward-compatibility shims** that re-export from `enterprise_agent.evaluation`, ensuring seamless interop with existing tooling.

The migration preserves the legacy evaluation harness (`src/enterprise_agent/evaluation/judge.py`, `src/enterprise_agent/evaluation/scoring.py` and their `src/nga/evaluation/` shims) alongside the new DeepEval implementation, enabling side-by-side comparison of verdict agreement, score shifts, confidence, latency, and costs before retiring legacy components.

---

## 2. Core Decisions & Constraints

### 2.1 Model Pinning: `jev-1.13.0`
- To prevent silent upstream model updates from altering benchmark baselines, all evaluation paths pin `system_one_model = "jev-1.13.0"` instead of `"jev-latest"`.
- Pinned across:
  - Built-in DeepEval RAG metrics (`FaithfulnessMetric`, `AnswerRelevancyMetric`, `ContextualPrecisionMetric`, `ContextualRecallMetric`)
  - Domain `JevEval` metric (`make_domain_jev_eval_metric`, e.g. `make_manufacturing_jev_eval_metric`)
  - Cross-document contradiction `JevEval` metric (`make_contradiction_jev_eval_metric`)
  - Legacy Jev judge (`make_jev_judge`, `evaluate_with_jev`)

### 2.2 Execution Modes: `system_one` vs. `hybrid`
- **`system_one` (Default)**:
  - 100% Jev-only execution in a single request per metric.
  - Zero generative LLM calls, zero output token billing ($T_{\text{out}} = 0$).
  - Deterministic reasons and sub-50ms evaluation speed.
- **`hybrid` (Opt-in)**:
  - An LLM extracts statements/claims and writes narrative reasons while Jev decides every verdict in between.
  - Useful when human-readable natural language justifications are required.
  - Token and latency costs reflect both LLM and Jev operations.

---

## 3. Architecture & Components

```
                                ┌──────────────────────────────────────────────┐
                                │        Benchmark Question / Trajectory       │
                                └──────────────────────┬───────────────────────┘
                                                       │
                                                       ▼
                                ┌──────────────────────────────────────────────┐
                                │   DeepEval Adapter (deepeval_adapter.py)    │
                                │   - Maps Question -> input                   │
                                │   - Generated Answer -> actual_output        │
                                │   - Golden Answer -> expected_output         │
                                │   - SOP / KG / SQL -> retrieval_context      │
                                │   - Tool Telemetry -> ToolCall instances     │
                                └──────────────────────┬───────────────────────┘
                                                       │
                        ┌──────────────────────────────┴──────────────────────────────┐
                        ▼                                                             ▼
         ┌─────────────────────────────┐                               ┌─────────────────────────────┐
         │     Built-in RAG Metrics    │                               │     Custom JevEval Metrics  │
         │  (Faithfulness, Relevancy,  │                               │  - Manufacturing Rubric     │
         │    Precision, Recall)       │                               │  - Contradiction Handling   │
         │   [Pinned: jev-1.13.0]      │                               │    [Pinned: jev-1.13.0]     │
         └──────────────┬──────────────┘                               └──────────────┬──────────────┘
                        │                                                             │
                        └──────────────────────────────┬──────────────────────────────┘
                                                       ▼
                                ┌──────────────────────────────────────────────┐
                                │   DeepEval Execution Engine & Safety Gating  │
                                │   - Extracts sub-criterion probabilities     │
                                │   - Clamps on severe hallucination / specs   │
                                │   - Emits OpenTelemetry spans & attributes   │
                                └──────────────────────────────────────────────┘
```

### 3.1 Test Case Conversion (`src/enterprise_agent/evaluation/deepeval_adapter.py`)
- Canonical adapter residing in `src/enterprise_agent/evaluation/deepeval_adapter.py` (re-exported by `src/nga/evaluation/deepeval_adapter.py`).
- Maps benchmark rows (`eval-questions/questions.json` or `domains/<name>/data/eval/questions.json`) to DeepEval `LLMTestCase`.
- Extracts rich context via `extract_retrieval_context`:
  - Document retrieval tools (`search_sop_documents`, `search_compliance_documents`, or generic retrieval): formats document passages, doc_ids, categories, and knowledge-graph evidence (`entities`, `relations`, `community_summaries`).
  - SQL database tools (`query_nga_database`, `query_banking_database`, or generic SQL): serializes SQL queries, returned rows, and table schemas into structured `[SQL RESULT]` blocks.
- Maps cross-document contradiction scenarios (`CONFLICT_SCENARIOS`) to condition test cases (`conflict_flagged`, `blind_variant`, `hallucinated_middle`).

### 3.2 DeepEval RAG & Custom Metrics (`src/enterprise_agent/evaluation/deepeval_metrics.py`)
- Canonical metric factory residing in `src/enterprise_agent/evaluation/deepeval_metrics.py` (re-exported by `src/nga/evaluation/deepeval_metrics.py`).
- **Built-in RAG Metrics**: `FaithfulnessMetric`, `AnswerRelevancyMetric`, `ContextualPrecisionMetric`, `ContextualRecallMetric`.
- **Domain Quality `JevEval` (`make_domain_jev_eval_metric`)**:
  - `Score` (weight 2.0): 6-level domain rubric (0–5 mapped to `[0, 1]`).
  - `Noul` (weight 1.5): Zero hallucinated specs, values, or fake document references (`is_faithful`).
  - `Noul` (weight 1.5): Exact numerical figures, limits, tolerances, or transaction values (`specs_accurate`).
  - `Noul` (weight 1.0): Coverage of required operational or compliance procedures (`completeness`).
  - `Choice` (weight 1.0): Defect classification (`none`, `missing_procedure`, `inaccurate_specs`, `hallucinated_reference`, `intent_mismatch`).
- **Contradiction Handling `JevEval`**:
  - Evaluates explicit conflict flagging using `Choice` with `"no_contradiction_in_expected": None` dropout so normal questions are not penalized.

### 3.3 OpenTelemetry Tracing Integration
- Evaluated cases emit OpenTelemetry spans named `deepeval.evaluate_test_case`.
- Recorded attributes include:
  - `question_id`, `category`, `eval_mode`, `system_one_model`
  - `faithfulness`, `answer_relevancy`, `contextual_precision`, `contextual_recall`
  - `domain_jev_score` / `manufacturing_jev_score`, `rubric_score_0_to_5`, `is_faithful`, `specs_accurate`, `completeness`, `failure_mode`
  - `min_confidence`, `overall_score`, `passed`, `latency_ms`, token usage, and cost

### 3.4 Vertical Domain Pack Portability (`make_domain_jev_eval_metric` & `to_deep_eval_test_case`)
Any vertical domain pack (e.g. `domains/banking/` or future custom domains) integrates with the evaluation subsystem without custom metric code:
1. **Dynamic Test Case Conversion (`to_deep_eval_test_case`)**:
   ```python
   from enterprise_agent.evaluation.deepeval_adapter import to_deep_eval_test_case

   test_case = to_deep_eval_test_case(
       question_data=question_row,
       actual_output=agent_answer,
       tool_outputs=[retrieval_output_json, sql_output_json],
       tools_called=["search_compliance_documents", "query_banking_database"],
   )
   ```
   - Automatically parses `results` containing `doc_id`, `chunk_id`, `category`, `corpus_source`, and `text`.
   - Parses SQL rows, column names, and row counts into formatted `[SQL RESULT]` chunks.
   - Populates `test_case.retrieval_context` and creates typed DeepEval `ToolCall` instances.
2. **Domain Evaluation Metric (`make_domain_jev_eval_metric`)**:
   ```python
   from enterprise_agent.evaluation.deepeval_metrics import (
       make_domain_jev_eval_metric,
       evaluate_test_case_with_deepeval,
   )

   domain_metric = make_domain_jev_eval_metric(
       threshold=config.evaluation.pass_threshold,
       system_one_model=config.evaluation.system_one_model, # "jev-1.13.0"
       include_reason=True,
   )
   result = evaluate_test_case_with_deepeval(test_case, custom_metrics=[domain_metric])
   ```
   - Evaluates multi-dimensional domain quality in sub-50ms using TypeSafe AI Jev System One without generative LLM billing.

---

## 4. Side-by-Side Comparison Harness

The canonical side-by-side comparison engine (`src/enterprise_agent/evaluation/side_by_side.py`, re-exported via `src/nga/evaluation/side_by_side.py`) evaluates benchmark cases on both harnesses concurrently and records:
- **Per-Criterion Verdict Agreement**:
  - `is_faithful` agreement (Legacy Jev vs DeepEval `JevEval` Noul)
  - `faithfulness` agreement (Legacy Jev vs DeepEval `FaithfulnessMetric`)
  - `specs_accurate` agreement (Legacy Jev vs DeepEval `JevEval` Noul)
  - `completeness` agreement (Legacy Jev vs DeepEval `JevEval` Noul)
  - `contextual_recall` agreement (Legacy Jev vs DeepEval `ContextualRecallMetric`)
  - Rubric agreement within ±1 level
  - Overall pass/fail verdict agreement
- **Score Shift Analysis**: Distribution of `deepeval_overall_score - legacy_normalized_score`.
- **Artifacts Saved**: Markdown and JSON comparison ledgers under `reports/eval/` and `reports/judge/{YYYY-MM-DD}/`.

---

## 5. Running Evaluations

```bash
# Run side-by-side evaluation with pure Jev (default, zero LLM calls)
uv run python scripts/side_by_side_eval.py --limit 6

# Run side-by-side evaluation with hybrid mode (LLM reasons + Jev verdicts)
uv run python scripts/side_by_side_eval.py --eval-mode hybrid --limit 6

# Run full unit test suite
uv run pytest tests/unit/test_deepeval_adapter.py tests/unit/test_deepeval_metrics.py tests/unit/test_side_by_side.py -v
```
