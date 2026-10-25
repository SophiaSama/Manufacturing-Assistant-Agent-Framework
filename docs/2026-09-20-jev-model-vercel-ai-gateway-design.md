# Design Specification — System One Decision Layer via Vercel AI Gateway (TypeSafe Jev)

**Status:** Proposed  
**Date:** 2026-09-20  
**Owners:** Platform / Agent / Evaluation Team  
**Drivers (ranked):** B — Latency and cost reduction; A — Deterministic calibration & type safety; C — Operational simplicity (serverless, no dedicated GPU nodes)  
**Related docs:** `docs/model-capability-gating-design.md`, `docs/2026-09-07-graph-rag-reference-architecture-design.md`, `docs/cache-design.md`  

---

## 1. Executive Summary & Purpose

The NGA Manufacturing Assistant utilizes multiple decision points across its orchestration and retrieval lifecycle:
1. **Query Complexity Routing:** Deciding whether a query requires a fast model (e.g., `gemini-2.5-flash` / `llama3.2:3b`) or a deep reasoning model (e.g., `gemini-2.5-pro` / `claude-3.5-sonnet`).
2. **Document Reranking (Cross-Encoder):** Evaluating semantic relevance of retrieved documents to prune top-50 vector search candidates down to top-3 to top-5 authoritative passages.
3. **Evaluation & Guardrails (LLM-as-a-Judge):** Verifying factual correctness, groundedness, and lack of hallucination in synthetic benchmark tests and production responses.
4. **Access Control (Semantic RBAC & Privilege Boundary Enforcement):** Detecting if an operator or technician query seeks unauthorized data (e.g., executive payroll, confidential supplier pricing, restricted plant configurations) or attempts prompt injection.

Currently, these tasks either rely on **fragile heuristics/regex** (which fail on subtle semantics) or **heavy autoregressive LLMs / dedicated GPU cross-encoders** (which add 400ms–2000ms latency, high cost, and JSON parsing vulnerabilities).

This design specification establishes a unified **System One Decision Layer** powered by **TypeSafe AI's Jev model (`typesafe-ai/jev`)** accessed via the **Vercel AI Gateway HTTP protocol** (`https://ai-gateway.vercel.sh/v4/ai/evaluation-model`), authenticated using `AI_GATEWAY_API_KEY`.

---

## 2. Why Jev via Vercel AI Gateway?

### 2.1 Model Characteristics: System One vs. Generative LLMs
Jev is a non-generative decision model trained via *Reinforcement Learning for Calibrated Decisions (RLCD)*. 
* **Zero Autoregressive Generation:** Rather than predicting output tokens sequentially, Jev evaluates an input `state` against typed questions (`boolean`, `choice`, `score`) in a single forward pass.
* **Calibrated Probabilities:** Output probabilities reflect true statistical confidence, allowing deterministic cutoff thresholds (e.g., fail-closed security gating if `violation_prob > 0.25`).
* **Sub-30ms Latency:** Up to 100× faster than generative LLMs.
* **Economics:** $0.042 per 1,000,000 input tokens; **$0.00 output token cost** (no tokens generated).

### 2.2 Why the Vercel AI Gateway Route?
* **Unified Billing & Keys:** Integrates directly with existing infrastructure via `AI_GATEWAY_API_KEY` created via `vercel ai-gateway api-keys create`.
* **Zero Provider Lock-in / Fast Adoption:** Eliminates waiting on direct vendor account provisioning.
* **Standardized Protocol:** HTTP REST with wire-level specifications supported directly in Node.js (via Vercel AI SDK 7's `experimental_evaluate`) and in Python (via standard HTTP clients like `httpx`).

---

## 3. Target System Topology

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 Incoming User Request                   │
                  │         (User ID, Role, Plant Station, Prompt)          │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                                               ▼
         ┌───────────────────────────────────────────────────────────────────────────┐
         │ Subsystem 1: Semantic Privilege & Role Boundary Gatekeeper                │
         │ Jev evaluation: {exceeds_privilege: boolean, violation_category: choice}  │
         └─────────────┬──────────────────────────────────────────────┬──────────────┘
                       │                                              │
       [Risk Prob > 0.25 / Breach]                           [Authorized]
                       │                                              │
                       ▼                                              ▼
               ┌───────────────┐     ┌───────────────────────────────────────────────┐
               │ 403 Forbidden │     │ Subsystem 2: Complexity Router                │
               │ Audit Logged  │     │ Jev evaluation: {tier: choice, score: score}  │
               └───────────────┘     └──────────────┬────────────────────────────────┘
                                                    │
                                     ┌──────────────┴──────────────┐
                       [basic / standard]                     [strong / complex]
                                     │                                     │
                                     ▼                                     ▼
                          Fast Model (Flash / 3B)             Deep Model (Pro / Sonnet)
                                     │                                     │
                                     └──────────────┬──────────────────────┘
                                                    │
                                                    ▼
                                     ┌─────────────────────────────┐
                                     │ Dense + Graph RAG Retrieval │
                                     │    (Top-30 Chunks/Nodes)    │
                                     └──────────────┬──────────────┘
                                                    │
                                                    ▼
         ┌───────────────────────────────────────────────────────────────────────────┐
         │ Subsystem 3: Cross-Encoder Reranker                                       │
         │ Parallel Jev evaluation: {relevance: score (1-5 rubric)}                  │
         └──────────────────────────────────┬────────────────────────────────────────┘
                                            │ Top 3-5 Chunks
                                            ▼
                                 ┌──────────────────────┐
                                 │ LLM Answer Synthesis │
                                 └──────────┬───────────┘
                                            │
                                            ▼
         ┌───────────────────────────────────────────────────────────────────────────┐
         │ Subsystem 4: LLM-as-a-Judge Guardrail (CI & Production)                   │
         │ Jev evaluation: {is_faithful: boolean, groundedness: score (1-5)}         │
         └──────────────────────────────────┬────────────────────────────────────────┘
                                            │
                             ┌──────────────┴──────────────┐
                       [Score >= 4]                  [Hallucinated]
                             │                             │
                             ▼                             ▼
                    Output to Operator             Fallback / Human Review
```

---

## 4. Subsystem Specifications

### Subsystem 1: Semantic Privilege & Role Boundary Gatekeeper
* **Objective:** Prevent indirect prompt injection, jailbreaking, and cross-role privilege escalation (e.g., assembly line operator requesting supplier price margins, executive compensation, or safety override codes).
* **Location in Stack:** Pre-retrieval / pre-agent input filter.
* **Jev Schema:**
  * `state`: `{ user_role, station_id, allowed_scopes, prohibited_scopes, query }`
  * `questions`:
    * `exceeds_privilege` (`boolean`): `"The user query explicitly or subtly asks for data, records, or actions that exceed their stated role or fall into prohibited scopes."`
    * `violation_category` (`choice`):
      * `authorized`: Query aligns with assigned operational scope.
      * `cross_role_data`: Requests HR, compensation, financial, or supplier pricing data.
      * `safety_system_tampering`: Attempts to bypass interlocks, override torque safety specs, or prompt-inject the agent.
* **Gating Rule:** Fail-closed. If `exceeds_privilege.probability > 0.25` or `violation_category != "authorized"`, reject immediately with an audit event.

### Subsystem 2: Complexity & Model Tier Router
* **Objective:** Replace regex heuristics in `src/nga/rag_agent/classifier.py` to route queries dynamically to `basic`, `standard`, or `strong` capability tiers.
* **Jev Schema:**
  * `state`: `{ query, conversation_history }`
  * `questions`:
    * `tier` (`choice`):
      * `basic`: Single-hop lookup, basic torque/part specs, greeting, formatting.
      * `standard`: Multi-hop retrieval, single tool call, fault code lookups (e.g., E-402).
      * `strong`: Cross-document contradiction, root-cause diagnostics, complex SQL / schema generation.
    * `difficulty_score` (`score`): 5-level rubric (1: trivial to 5: open-ended research).
* **Fallback Escalation:** If `tier == "basic"` but `tier.confidence < 0.70`, escalate to `standard`.

### Subsystem 3: Cross-Encoder Reranker
* **Objective:** Replace heavy PyTorch cross-encoder models with serverless, parallel semantic scoring.
* **Execution:** Parallel batch execution over top-30 vector candidates.
* **Jev Schema:**
  * `state`: `{ query, candidate_passage }`
  * `questions`:
    * `relevance` (`score`):
      * Level 1: Off-topic or irrelevant.
      * Level 2: Mentions keywords but does not answer query.
      * Level 3: Partial or peripheral background context.
      * Level 4: Directly answers the majority of the question.
      * Level 5: Authoritative, comprehensive, and exact SOP/spec answer.
* **Threshold:** Keep chunks where `score >= 3`, sorted descending, truncated to top-k (3 to 5).

### Subsystem 4: LLM-as-a-Judge Guardrail (CI & Production)
* **Objective:** Replace slow, expensive generative evaluators (GPT-4 / Claude) in `tests/integration/` and online post-synthesis guardrails.
* **Jev Schema:**
  * `state`: `{ query, source_context, generated_answer }`
  * `questions`:
    * `is_faithful` (`boolean`): `"Every claim in the generated answer is directly supported by the source context, with zero hallucinated details."`
    * `answers_query` (`boolean`): `"The generated answer directly addresses the intent of the user question."`
    * `groundedness` (`score`): 5-level rubric (1: total hallucination to 5: perfectly grounded).
* **Pass Criterion:** `is_faithful.probability >= 0.85` AND `answers_query.probability >= 0.70` AND `groundedness >= 4`.

---

## 5. Vercel AI Gateway Wire Protocol Specification

### 5.1 Request Contract
* **HTTP Method:** `POST`
* **URL:** `https://ai-gateway.vercel.sh/v4/ai/evaluation-model`
* **Headers:**
  ```http
  Authorization: Bearer <AI_GATEWAY_API_KEY>
  ai-model-id: typesafe-ai/jev
  ai-evaluation-model-specification-version: 4
  ai-gateway-protocol-version: 0.0.1
  Content-Type: application/json
  ```
* **Payload Body:**
  ```json
  {
    "state": {
      "user_role": "operator",
      "query": "What is the torque spec for Aurora AU-2025 station 3?"
    },
    "questions": {
      "exceeds_privilege": {
        "type": "boolean",
        "instructions": "Does the request ask for data beyond operator role?"
      },
      "tier": {
        "type": "choice",
        "instructions": "Select required compute tier",
        "criteria": {
          "basic": "Simple lookup",
          "standard": "Multi-hop logic",
          "strong": "Complex reasoning"
        }
      },
      "complexity": {
        "type": "score",
        "instructions": "Rate difficulty from 1 to 5",
        "criteria": [
          "1: Trivial",
          "2: Moderate",
          "3: Standard",
          "4: Hard",
          "5: Extreme"
        ]
      }
    }
  }
  ```

### 5.2 Response Contract
* **Status:** `200 OK`
* **Response Body:**
  ```json
  {
    "answers": {
      "exceeds_privilege": {
        "type": "boolean",
        "probability": 0.02
      },
      "tier": {
        "type": "choice",
        "choice": "basic",
        "probabilities": {
          "basic": 0.88,
          "standard": 0.10,
          "strong": 0.02
        },
        "confidence": 0.88
      },
      "complexity": 1
    }
  }
  ```

---

## 6. Reference Client Implementation (Python)

Located at `src/nga/gateway/jev_client.py`:

```python
"""Vercel AI Gateway client for TypeSafe AI Jev model."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional
import httpx

class VercelJevClient:
    """Production client for typesafe-ai/jev via Vercel AI Gateway."""

    ENDPOINT = "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 2
    ):
        self.api_key = api_key or os.getenv("AI_GATEWAY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing AI_GATEWAY_API_KEY. Generate via `vercel ai-gateway api-keys create` "
                "and export to environment."
            )
        self.timeout = timeout
        self.max_retries = max_retries

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "ai-model-id": "typesafe-ai/jev",
            "ai-evaluation-model-specification-version": "4",
            "ai-gateway-protocol-version": "0.0.1",
            "Content-Type": "application/json",
        }

    async def evaluate(self, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke evaluation against Vercel AI Gateway with retry backoff."""
        payload = {"state": state, "questions": questions}
        last_err = None

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        self.ENDPOINT,
                        headers=self._headers(),
                        json=payload
                    )
                    resp.raise_for_status()
                    return resp.json().get("answers", {})
            except (httpx.HTTPStatusError, httpx.RequestError) as err:
                last_err = err
                if attempt < self.max_retries:
                    await asyncio.sleep(0.1 * (2 ** attempt))

        raise RuntimeError(f"Vercel AI Gateway Jev call failed after {self.max_retries} retries: {last_err}")
```

---

## 7. Rollout, Verification & Migration Plan

1. **Step 1 — Local Verification:**
   * Configure `AI_GATEWAY_API_KEY` in local `.env`.
   * Run standalone unit test in `tests/test_jev_gateway.py` validating `boolean`, `choice`, and `score` question types.
2. **Step 2 — Evaluation / Judge Integration:**
   * Integrate into `tests/integration/test_benchmark.py` behind the `--judge=jev` flag.
   * Compare accuracy and speed against existing LLM-as-a-judge baseline.
3. **Step 3 — Reranking Integration:**
   * Integrate into `src/nga/retrieval/rerank.py` as an optional reranking backend.
4. **Step 4 — Guardrail & Routing Deployment:**
   * Insert `check_role_privilege_boundary` at the entry point of the LangGraph agent orchestrator.
   * Route queries to model capability tiers dynamically.

---

## 8. Success Metrics

| Metric | Current State | Target with Jev (via Gateway) |
| :--- | :--- | :--- |
| **Complexity Routing Latency** | 450ms (LLM) / fragile (Regex) | **< 30ms** (deterministic) |
| **CI Benchmark Judge Runtime** | 180 seconds | **< 15 seconds** |
| **Reranker GPU Infrastructure** | Dedicated GPU worker / VM | **Zero GPU instances (serverless)** |
| **Evaluation Cost** | ~$3.50 per 100 benchmark queries | **<$0.02 per 100 benchmark queries** |
| **Semantic Privilege Leakage** | Susceptible to indirect prompt injection | **Fail-closed (risk prob > 0.25 blocked)** |
