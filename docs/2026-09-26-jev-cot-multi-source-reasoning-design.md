# Design Specification — Multi-Source Chain of Thought & Reasoning Orchestration via TypeSafe Jev

**Status:** Proposed  
**Date:** 2026-09-26  
**Owners:** Agent / Orchestrator / Platform Architecture Team  
**Drivers (ranked):**  
1. **A — Factual grounding & zero-hallucination guarantee** (preventing ungrounded deductions in multi-source manufacturing operations)  
2. **B — Latency and cost optimization** (semantic early exit reducing unnecessary multi-round LLM tool loops)  
3. **C — Resilient defense-in-depth** (combining deterministic mechanical circuit breakers with calibrated semantic gates)  
**Related docs:** `docs/2026-09-20-jev-model-vercel-ai-gateway-design.md`, `docs/2026-09-07-graph-rag-reference-architecture-design.md`, `docs/model-capability-gating-design.md`

---

## 1. Executive Summary & Problem Statement

The NGA Manufacturing Assistant synthesizes intelligence across disparate, heterogeneous knowledge sources:
* **Relational SQL Database (`database/nga.db`):** Live build records, station logs, defect event tables, and shift metrics.
* **Vector Document Corpus (ChromaDB):** Standard Operating Procedures (SOPs), machine specs, torque tolerances, and safety standards (`TEC`, `OPR`, `FA`).
* **Maintenance Work Orders (`additional-docs/maintenance-work-orders/`):** Real-world unscheduled maintenance logs, station repairs, and recalibration reports (`WO-2025-*`).
* **Supplier Quality Records (`additional-docs/supplier-quality/`):** Supplier corrective action requests (`SCAR-2025-*`), audit findings (`AUD-2025-*`), and interim containment deviations.
* **Subsystem Knowledge Graph:** Entity-relation linkages connecting stations, failure modes, tools, and components.

### 1.1 Existing Limitations in Autoregressive CoT
In complex multi-source inquiries (e.g. root-cause failure analysis, Class A safety investigations, or conflicting torque specifications), standard autoregressive Chain-of-Thought (CoT) exhibits severe failure modes:
1. **Tool-Loop Thrashing:** The agent iteratively invokes tools until hitting hardcoded loop ceilings (`MAX_TOOL_CALL_ROUNDS = 8`), even when sufficient evidence was collected in round 1 or 2, causing 4–10 seconds of needless latency.
2. **Blindness to Cross-Source Contradictions:** When a standard SOP specifies $45\,\text{Nm}$ nominal torque, but an active SCAR specifies an interim containment override of $52\,\text{Nm}$, generative LLMs tend to blend contradictory numbers, pick one arbitrarily, or produce misleading compromises.
3. **Compounding Step Errors:** In a multi-hop reasoning chain, if Step 1 misinterprets a column or unit, subsequent steps build on that hallucinated deduction without verification.
4. **Context Bloating & Lost-in-the-Middle:** Loading raw SQL result dumps, multiple SOP sections, and maintenance tickets into the reasoning LLM context dilutes attention.

### 1.2 The Solution: System One + System Two Cognitive Architecture
While **Pattern 5 (Context Compression / Jev Reranker)** is implemented in `src/enterprise_agent/retrieval/reranker.py`, the active reasoning loop remains unguided.

This specification introduces a hybrid **System One (TypeSafe Jev) + System Two (Generative LLM)** architecture:
* **Generative LLMs (System Two)** perform expressive tasks: natural language synthesis, hypothesis generation, and SQL statement construction.
* **TypeSafe Jev (System One)** provides sub-30ms, non-generative, calibrated judgments (`Choice`, `Score`, `Noul`) to steer, verify, prune, and gate the reasoning lifecycle.
* **Mechanical Circuit Breakers** remain intact as non-bypassable fail-safes to ensure rock-solid stability.

---

## 2. Target Orchestration Topology

```
                         ┌────────────────────────────────────────────────────────┐
                         │                 User Manufacturing Query               │
                         │             (Question, Role, Active Station)           │
                         └───────────────────────────┬────────────────────────────┘
                                                     │
                                                     ▼
                         ┌────────────────────────────────────────────────────────┐
                         │ Node: prepare                                          │
                         │ ├─ Jev Complexity Tier Route (existing classifier)     │
                         │ └─ [Pattern 1] Jev Speculative Multi-Source Fan-out    │
                         │    {needs_sql, needs_sop, needs_scar, needs_wo}        │
                         └───────────────────────────┬────────────────────────────┘
                                                     │
                                                     ▼
                         ┌────────────────────────────────────────────────────────┐
                         │ Concurrent Multi-Source Retrieval                      │
                         │ ├─ SQL query / schema inspection                       │
                         │ ├─ Vector similarity search (SOPs, WOs, SCARs)         │
                         │ └─ [Pattern 5] Jev Candidate Reranker (reranker.py)    │
                         └───────────────────────────┬────────────────────────────┘
                                                     │
                                                     ▼
                         ┌────────────────────────────────────────────────────────┐
                         │ Node: agent (System Two LLM CoT Generation)            │
                         │ Generates intermediate reasoning & proposed tool calls │
                         └───────────────────────────┬────────────────────────────┘
                                                     │
                                                     ▼
                         ┌────────────────────────────────────────────────────────┐
                         │ Conditional Edge: _route_after_agent                   │
                         │ ├─ [Layer 1] Mechanical Circuit Breaker (Hard Ceiling) │
                         │ │    rounds >= 8 OR repeats >= 3 -> Force synthesis    │
                         │ └─ [Layer 2] [Pattern 4] Jev Semantic Sufficiency Gate │
                         │      is_complete >= 0.80 & confidence >= 0.80          │
                         │      -> Early Exit to synthesis                        │
                         └─────────────┬────────────────────────────┬─────────────┘
                                       │                            │
                              [More Evidence Needed]        [Sufficient / Stop]
                                       │                            │
                                       ▼                            ▼
                         ┌───────────────────────────┐  ┌─────────────────────────────────────────────────┐
                         │ Node: tools               │  │ [Pattern 2] Cross-Source Contradiction Pre-Pass │
                         │ Execute requested queries │  │ Jev evaluates SQL vs SOP vs SCAR consistency    │
                         └─────────────┬─────────────┘  └───────────────────────┬─────────────────────────┘
                                       │                                        │
                                       └────────────────►                       ▼
                                                        ┌─────────────────────────────────────────────────┐
                                                        │ Node: synthesis                                 │
                                                        │ FinalAnswer structured output generation        │
                                                        └───────────────────────┬─────────────────────────┘
                                                                                │
                                                                                ▼
                                                        ┌─────────────────────────────────────────────────┐
                                                        │ [Pattern 3] Jev Fact Verification & Judge       │
                                                        │ {is_faithful: Noul, groundedness: Score 1-5}   │
                                                        └───────────────────────┬─────────────────────────┘
                                                                                │
                                                                ┌───────────────┴───────────────┐
                                                          [Score >= 4]                    [Groundedness < 4]
                                                                │                               │
                                                                ▼                               ▼
                                                        ┌───────────────┐               ┌───────────────┐
                                                        │ Node: hitl    │               │ Fallback /    │
                                                        │ Approval Gate │               │ Refusal State │
                                                        └───────┬───────┘               └───────────────┘
                                                                │
                                                                ▼
                                                               END
```

---

## 3. Detailed Specifications for Missing Reasoning Patterns

### 3.1 Pattern 1: Speculative Multi-Source Fan-out (`prepare` node)
* **Objective:** Determine up-front which of the multiple data stores must be queried before the LLM begins multi-round serial deliberation.
* **Placement:** Executed within `_make_prepare_node` in `src/nga/graph/orchestrator.py`.
* **Latency:** Single batch request to Jev (~25ms).
* **Jev Schema:**
  * **State:** `{"query": str, "user_role": str, "active_station": str}`
  * **Questions:**
    ```python
    questions = {
        "needs_sql_db": Noul(
            instructions=(
                "Does answering this question require quantitative database records, "
                "such as serial numbers, station cycle times, build counts, or defect logs?"
            )
        ),
        "needs_sop_procedures": Noul(
            instructions=(
                "Does answering this question require standard operating procedures, "
                "nominal torque specifications, tool calibration limits, or assembly steps?"
            )
        ),
        "needs_supplier_quality": Noul(
            instructions=(
                "Does this inquiry relate to incoming component defects, supplier containment, "
                "SCAR reports, or vendor material certifications?"
            )
        ),
        "needs_maintenance_logs": Noul(
            instructions=(
                "Does this inquiry reference past equipment breakdown, emergency work orders, "
                "or physical machine repairs?"
            )
        ),
        "cross_source_depth": Choice(
            instructions="Assess the degree of multi-source cross-referencing required.",
            criteria={
                "single_source": "Query can be answered completely from one document or table.",
                "dual_source": "Requires comparing two sources (e.g., build count vs SOP spec).",
                "triangulation": "Root-cause diagnosis requiring synthesis across database, SOP, and quality reports."
            }
        )
    }
    ```
* **Execution Behavior:** The orchestrator pre-fetches relevant schema descriptions or dispatches vector search across the predicted domains asynchronously, priming the state before the first agent turn.

---

### 3.2 Pattern 2: Cross-Source Contradiction & Precedence Gate
* **Objective:** Explicitly detect when evidence retrieved from Source A contradicts Source B (e.g. standard SOP vs temporary engineering deviation), and resolve precedence using deterministic domain hierarchy.
* **Placement:** Executed prior to `synthesis` in `src/nga/graph/orchestrator.py`.
* **Manufacturing Precedence Policy:**
  1. Priority 1 (Highest): Active Interim Containment / SCAR / Stop-Ship Notice
  2. Priority 2: Approved Maintenance Work Order Modification (`WO`)
  3. Priority 3: Released Standard Operating Procedure (`SOP` / `TEC`)
  4. Priority 4: General Documentation / Draft Specs
* **Jev Schema:**
  * **State:**
    ```json
    {
      "source_a": {"type": "SOP", "citation": "SOP-CHASSIS-04", "content": "Nominal torque: 45 Nm."},
      "source_b": {"type": "SCAR", "citation": "SCAR-2025-007", "content": "Interim containment: tighten to 52 Nm until further notice."}
    }
    ```
  * **Questions:**
    ```python
    questions = {
        "has_contradiction": Noul(
            instructions=(
                "Do source_a and source_b contain conflicting procedural requirements, "
                "incompatible numeric tolerances, or contradictory instructions?"
            )
        ),
        "conflict_nature": Choice(
            instructions="Identify the nature of the contradiction.",
            criteria={
                "none": "No conflict exists; sources are mutually compatible.",
                "numeric_tolerance": "Conflicting numeric torque, pressure, or dimension values.",
                "procedural_deviation": "Conflicting instructions or routing steps.",
                "status_discrepancy": "Discrepancy in component approval, pass/fail, or supplier status."
            }
        )
    }
    ```
* **Synthesis Guidance:** If `has_contradiction.probability >= 0.70`, the orchestrator injects an authoritative resolution block into the synthesis prompt:
  ```markdown
  [DETECTED SPECIFICATION CONFLICT]
  SOP-CHASSIS-04 (45 Nm) conflicts with SCAR-2025-007 (52 Nm).
  Rule: SCAR interim containment takes precedence over standard SOP. 
  Highlight this override explicitly in findings and recommendations.
  ```

---

### 3.3 Pattern 3: Step-by-Step Fact Verification & Grounding Judge
* **Objective:** Upgrade the primitive `_is_ungrounded` list check in `synthesis` to an exact, calibrated semantic verification of each claim against retrieved evidence.
* **Placement:** Immediately after answer synthesis, before `hitl`.
* **Jev Schema:**
  * **State:**
    ```json
    {
      "query": "What is the caliper bolt torque and are there deviations?",
      "source_context": "[Collected multi-source text from SQL and documents]",
      "generated_answer": "[Candidate text from LLM synthesis]"
    }
    ```
  * **Questions:**
    ```python
    questions = {
        "is_faithful": Noul(
            instructions=(
                "Is every factual claim, numerical specification, part ID, and procedure "
                "in generated_answer fully supported by source_context with zero hallucinated details?"
            )
        ),
        "groundedness": Score(
            instructions="Rate the groundedness of generated_answer on a 1-5 scale.",
            criteria={
                "1": "Severe hallucination; invents specifications or procedures not in source.",
                "2": "Major unsupported claims or incorrect numbers.",
                "3": "Mostly supported, but contains minor ungrounded peripheral statements.",
                "4": "High fidelity; all core technical claims match evidence exactly.",
                "5": "Flawlessly grounded; strict adherence to source citations."
            }
        ),
        "unsupported_claim_type": Choice(
            instructions="Categorize any ungrounded assertions.",
            criteria={
                "none": "All assertions are fully grounded.",
                "invented_numeric_spec": "Fabricated torque, temperature, dimension, or limit.",
                "invented_citation": "Cited a document, SOP ID, or database table that was not retrieved.",
                "unsupported_safety_assertion": "Unverified claim regarding Class A safety or recall status."
            }
        )
    }
    ```
* **Gating Policy:**
  * **Pass:** `is_faithful.probability >= 0.85` AND `groundedness.score >= 4`.
  * **Fail / Refusal:** If `is_faithful.probability < 0.50` or `groundedness.score < 3`, automatically quarantine the response, substitute a grounded refusal notice, and log a high-priority audit event.

---

### 3.4 Pattern 4: Adaptive Sufficiency Gate & Defense-in-Depth Circuit Breaker
* **Objective:** Prevent redundant tool loops while guaranteeing safety via a non-bypassable mechanical ceiling.
* **Placement:** In `_route_after_agent` in `src/nga/graph/orchestrator.py`.

#### Layered Architecture:
1. **Layer 1: Mechanical Circuit Breaker (Inviolable Fail-Safe)**
   * **Rule 1A:** `tool_rounds >= MAX_TOOL_CALL_ROUNDS (8)` $\rightarrow$ Trip circuit breaker.
   * **Rule 1B:** `identical_call_repeats >= MAX_IDENTICAL_TOOL_CALL_REPEATS (3)` $\rightarrow$ Trip circuit breaker.
   * **Action:** Immediately force routing to `synthesis`. No network or model call can override this.
2. **Layer 2: Semantic Sufficiency Gate (Jev System One Early Exit)**
   * **Trigger Condition:** Evaluated after $\ge 1$ round of tool evidence collection.
   * **Jev Schema:**
     * **State:** `{"original_query": str, "question_parts": list[str], "accumulated_evidence": str}`
     * **Questions:**
       ```python
       questions = {
           "is_sufficient": Noul(
               instructions=(
                   "Does accumulated_evidence contain all specific facts, tables, and document "
                   "clauses necessary to provide a complete, verified answer to original_query?"
               )
           ),
           "evidence_completeness": Score(
               instructions="Score the completeness of evidence across all question parts.",
               criteria={
                   "1": "Insufficient: Key question parts have zero evidence.",
                   "2": "Partial: Evidence gathered for some parts, but primary question unanswered.",
                   "3": "Adequate: Core question can be fully answered; minor details omitted.",
                   "4": "Exhaustive: Complete evidentiary support for every sub-part."
               }
           ),
           "next_action": Choice(
               instructions="Determine the optimal next step.",
               criteria={
                   "proceed_to_synthesis": "Evidence is complete; further tool calls would be redundant.",
                   "query_database": "Still missing structured database facts.",
                   "search_documents": "Still missing SOP or specification documents.",
                   "conclude_unanswerable": "Sources have been queried and data does not exist."
               }
           )
       }
       ```
   * **Early Exit Rule:** If `is_sufficient.probability >= 0.80` AND `evidence_completeness.score >= 3`, trigger **immediate early exit** to `synthesis`.

---

## 4. LangGraph AgentState Enhancements

To support these patterns, `AgentState` in `src/nga/graph/state.py` will be extended:

```python
class AgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]
    question_parts: list[str]
    model_route: dict[str, Any]
    user_role: str
    
    # ── Multi-Source Reasoning Additions ──────────────────────────────────────
    predicted_sources: dict[str, float]      # Pattern 1: Output of speculative fan-out
    evidence_sufficiency: dict[str, Any]     # Pattern 4: Current sufficiency score & next_action
    detected_conflicts: list[dict[str, Any]] # Pattern 2: Detected cross-source discrepancies
    verification_result: dict[str, Any]      # Pattern 3: Groundedness & faithfulness audit
    
    # ── Existing Fields ───────────────────────────────────────────────────────
    pending_recommendation: str | None
    answered_parts: list[str]
    unanswered_parts: list[str]
    final_answer: dict[str, Any] | None
```

---

## 5. Resilience Engineering & Failure Modes

| Failure Scenario | Immediate System Behavior | Fallback / Recovery |
| :--- | :--- | :--- |
| **TypeSafe API Timeout / Network Drop** | Jev client raises `TimeoutException` after 2 retries | Log warning; fall back seamlessly to mechanical circuit breaker (Layer 1) |
| **Misclassified Sufficiency (False Positive)** | Jev prematurely reports evidence complete | Pattern 3 (Fact Verification) in `synthesis` catches missing facts and refuses or flags unanswered parts |
| **Misclassified Sufficiency (False Negative)** | Jev reports evidence incomplete despite having facts | Layer 1 mechanical circuit breaker limits execution to max 8 rounds; prevents infinite loop |
| **Direct Contradiction in Evidence** | Conflict gate detects discrepancy ($p > 0.70$) | Injects hierarchical precedence prompt; forces explicit disclosure in final answer |
| **Severe Hallucination in Synthesis** | Fact verification score $< 3$ | Replaces synthesis with fail-closed refusal; alerts operator to review source documents directly |

---

## 6. Implementation & Rollout Roadmap

1. **Phase 1: Dual-Layer Stop Gate (Pattern 4 + Circuit Breaker)**
   * Add `evaluate_evidence_sufficiency()` utility wrapping Jev.
   * Update `_route_after_agent()` in `src/nga/graph/orchestrator.py` to check Layer 1 (circuit breaker) then Layer 2 (Jev early exit).
   * Benchmark against existing multi-hop test queries to verify reduction in redundant tool calls.
2. **Phase 2: Fact Verification Judge (Pattern 3)**
   * Replace naive `_is_ungrounded()` in `_make_synthesis_node()` with calibrated Jev `is_faithful` and `groundedness` scoring.
   * Add automated tests in `tests/test_fact_verification.py`.
3. **Phase 3: Speculative Multi-Source Fan-out (Pattern 1)**
   * Update `_make_prepare_node()` to predict domain source requirements and warm tool caches.
4. **Phase 4: Cross-Source Contradiction Resolution (Pattern 2)**
   * Implement pairwise conflict detection for multi-document retrieval responses.

---

## 7. Key Success Metrics

| Metric | Baseline / Fallback State | Target with Jev Multi-Source Orchestration | Improvement Delta |
| :--- | :--- | :--- | :--- |
| **Average Multi-Hop Tool Rounds** | 4.2 rounds / query | **1.8 rounds / query** (via semantic early exit) | **-57% fewer tool turns** |
| **Unnecessary Loop Over-runs** | 18% of complex queries hit round 8 | **< 1%** (tripped only when truly unanswerable) | **-94% loop runaway reduction** |
| **Decision Output Tokens Generated** | 200–600 tokens per evaluation | **0 tokens** ($0.00 output cost) | **100% output token elimination** |
| **Factual Hallucination Rate** | ~3.8% on edge cases | **< 0.2%** (fail-closed fact verification gate) | **-95% reduction** |
| **Conflict Resolution Accuracy** | Inconsistent (LLM blends specs) | **100% deterministic precedence** (SCAR/WO > SOP) | **Deterministic compliance** |
| **Routing & Verification Overhead** | 1,200ms–2,500ms (LLM self-reflection) | **< 35ms** per forward evaluation pass | **~50× latency reduction** |
| **Cost per 1,000 Complex Queries** | ~$4.80 – $18.50 (depending on base model) | **~$1.40 – $4.90** | **~70% cost reduction** |

---

## 8. Token Consumption & Cost Tracking KPI Framework

To quantitatively compare the operational efficiency of **Jev-accelerated reasoning** versus **fallback/baseline configurations** (e.g. unaccelerated LLM tool loops, generative LLM-as-a-judge, and heuristic fallbacks), the framework establishes dedicated token tracking KPIs.

### 8.1 KPI Definitions & Mathematical Formulas

1. **Token Consumption Efficiency Ratio (TCER)**:
   Measures total tokens consumed per resolved query under Jev vs. fallback execution:
   $$\text{TCER} = \frac{T_{\text{in}}^{\text{Jev}} + T_{\text{out}}^{\text{Jev}}}{T_{\text{in}}^{\text{Fallback}} + T_{\text{out}}^{\text{Fallback}}}$$
   * Target: $\text{TCER} \le 0.45$ (meaning Jev uses less than 45% of the tokens consumed by the fallback baseline).

2. **Output Token Elimination Rate (OTER)**:
   Measures the share of auxiliary decision tokens completely eliminated by Jev's non-generative forward pass:
   $$\text{OTER} = \frac{T_{\text{out}}^{\text{Fallback\_Aux}} - T_{\text{out}}^{\text{Jev\_Aux}}}{T_{\text{out}}^{\text{Fallback\_Aux}}} \times 100\% = 100\%$$
   * Because Jev outputs typed decisions without generating tokens, auxiliary decision output token cost is **$0.00**.

3. **Cost per 1,000 Queries ($C_{1\text{k}}$)**:
   Calculated across all query steps:
   $$C_{1\text{k}} = 1000 \times \left( \frac{T_{\text{in}}}{10^6} \times P_{\text{in}} + \frac{T_{\text{out}}}{10^6} \times P_{\text{out}} \right)$$
   * Where $P_{\text{in}}$ and $P_{\text{out}}$ are the pricing tiers per million tokens for each involved model.

4. **Cost Delta & ROI ($\Delta C$)**:
   $$\Delta C = \frac{C_{1\text{k}}^{\text{Fallback}} - C_{1\text{k}}^{\text{Jev}}}{C_{1\text{k}}^{\text{Fallback}}} \times 100\%$$

---

### 8.2 Component-by-Component Token & Cost Comparison

| Reasoning Lifecycle Phase | Fallback / Baseline Implementation | Jev System One Implementation | Token Delta (Per Decision) | Cost Delta ($/1M ops) |
| :--- | :--- | :--- | :--- | :--- |
| **Model Tier & Source Routing** | Generative LLM prompt-and-parse (~400 in, 120 out) | `Choice` + `Noul` (~350 in, **0 out**) | **-120 output tokens** | **$0.015 vs $0.96** (-98%) |
| **Document Reranking** | Dedicated GPU worker OR LLM listwise rank (~2,000 in, 250 out) | Jev parallel `Choice` (~1,800 in, **0 out**) | **-250 output tokens** | **$0.076 vs $4.35** (-98%) |
| **Tool Loop Stop Gate** | Heuristic misses $\rightarrow$ 3 extra LLM rounds (~4,500 in, 900 out) | Jev `Score` early exit at round 1 or 2 (~600 in, **0 out**) | **-3,900 in, -900 out** | **$0.025 vs $27.00** (-99%) |
| **Fact Verification / Grounding** | LLM self-reflection / judge prompt (~1,500 in, 350 out) | Jev `Noul` + `Score` (~1,400 in, **0 out**) | **-350 output tokens** | **$0.059 vs $5.70** (-99%) |
| **Total Query Decision Overhead** | **~8,400 input tokens, ~1,620 output tokens** | **~4,150 input tokens, 0 auxiliary output tokens** | **-4,250 in, -1,620 out** | **~$0.17 vs ~$38.00** (-99.5%) |

---

### 8.3 Telemetry Data Model & Integration

To log and report these KPIs in CI and production runs, the orchestrator logs token telemetry into `AgentState["token_usage"]` and persists it to `reports/eval/history.json`:

```json
{
  "query_id": "eval-q-042",
  "routing_source": "typesafe_jev",
  "tool_rounds_executed": 2,
  "circuit_breaker_tripped": false,
  "early_exit_triggered": true,
  "token_consumption": {
    "system_two_llm": {
      "model": "anthropic/claude-3.5-sonnet",
      "prompt_tokens": 2840,
      "completion_tokens": 420,
      "cost_usd": 0.01482
    },
    "system_one_jev": {
      "model": "typesafe-ai/jev",
      "eval_calls": 4,
      "input_tokens": 1650,
      "output_tokens": 0,
      "cost_usd": 0.000069
    },
    "totals": {
      "total_tokens": 4910,
      "total_cost_usd": 0.014889
    }
  },
  "counterfactual_fallback_estimate": {
    "estimated_rounds_without_jev": 5,
    "estimated_fallback_tokens": 9800,
    "estimated_fallback_cost_usd": 0.03840,
    "token_savings_pct": 49.9,
    "cost_savings_pct": 61.2
  }
}
```

### 8.4 Verification in Automated Evaluation Suite
The token consumption KPI will be automatically computed and asserted in:
* `tests/test_jev_reasoning.py`: Validates that Jev evaluations return `output_tokens == 0`.
* `src/nga/evaluation/multi_model_eval.py`: Includes Jev in the Pareto efficiency frontier plot, visualizing cost savings against frontier baselines.
* `src/nga/evaluation/ci_tracker.py`: Tracks rolling token consumption trends and flags cost regressions if average query tokens exceed the baseline threshold.
