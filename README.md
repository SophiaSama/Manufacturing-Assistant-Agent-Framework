# NGA Manufacturing Assistant — Agentic Decision Support

> **⚠️ Disclaimer — Synthetic Evaluation Corpus**:  
> All data in this repository — including **Apex Automotive**, the **Northgate Assembly Plant (NGA)**, vehicle models (*Aurora AU-2025*, *Solstice SO-2025*), Standard Operating Procedures (SOPs), machine specifications, fault codes, non-conformance records, torque logs, and SQLite database tables (`nga.db`) — is **entirely synthetic and fictional**. It has been generated strictly for research, testing, and AI decision-support benchmark evaluation purposes.

The NGA Manufacturing Assistant is a production-grade, role-aware agentic framework designed for the synthetic **Apex Automotive — Northgate Assembly Plant (NGA)** benchmark. It acts as an intelligent manufacturing assistant that helps operators, technicians, engineers, and plant managers make fast, compliant operational decisions while strictly adhering to Standard Operating Procedures (SOPs).

The architecture is adapted from the `DSS_Prototype` reference workflow and extended to handle 4-tier role-based access control, safety-critical Class A defects, recall evaluation criteria (QCR-501), and multi-document inconsistencies.

---

## 🛠️ Technology Stack & Rationale

| Technology | Selected Component | Why This Stack Was Chosen |
|---|---|---|
| **Orchestration** | **LangGraph** | Provides a state-machine framework to define deterministic, predictable paths (`prepare` → `agent` ⇆ `tools` → `synthesis` → `hitl`). This prevents unstructured agent wandering and guarantees that safety verification nodes are always hit before outputting answers. |
| **Vector DB** | **ChromaDB** | Lightweight, file-system persistent database that supports standard metadata filtering. Crucial for enforcing path-derived RBAC constraints during retrieval (`level_rank <= user_level`) directly inside the similarity search. |
| **Knowledge Graph** | **NetworkX** | Python-native graph library. Used to persist extracted entities and relationships (e.g. `Machine`, `FaultCode`, `RecallCriteria`) for GraphRAG BFS expansion and multi-hop reasoning. |
| **Security (SQL)** | **SQLGlot** | Strict SQL AST parser. Validates generated text-to-SQL commands prior to execution. Restricts execution strictly to single, read-only `SELECT` queries, matching the live database schema (via `PRAGMA table_info`) to block SQL injection completely. |
| **Package Manager** | **uv / hatchling** | Modern, extremely fast Python package installer and workspace sync tool. Guarantees reproducible builds via locked dependencies. |
| **Testing** | **pytest** | Standard framework used to run the 75-question QA evaluation suite and the conflict detection scenarios as parameterized integration tests. |

---

## 📐 System Design & Architecture

```mermaid
graph TD
    User([User Prompt]) --> CLI[CLI / Role Selector]
    CLI --> Classifier[Complexity Classifier]
    Classifier --> Router{Model Router}
    Router -- 0-3 --> T1[Haiku / 1-hop]
    Router -- 4-6 --> T2[Sonnet / 3-hop]
    Router -- 7-10 --> T3[Opus / 8-hop]
    T1 & T2 & T3 --> Graph[LangGraph Orchestrator]
    
    subgraph Graph Nodes
        Graph --> Prep[prepare: Split compound query]
        Prep --> Agt[agent: Prompt Policy + Role Prompt]
        Agt --> Tools{Route After Agent}
        Tools -- tool calls --> ToolNode[tools: query_nga_database / search_sop_documents]
        ToolNode --> Agt
        Tools -- no tools / loop --> Synth[synthesis: Structure FinalAnswer + Safety Checks]
        Synth --> HITL[hitl: decisions_log + CLI Gate]
    end
    
    HITL --> Output([Rendered Answer])
```

### 1. 4-Tier Role-Based Access Control (RBAC)
To protect sensitive quality records and plant metrics, access is strictly partitioned:
* **operator (rank 1)**: Access restricted to `operator-sops/` (assembly procedures, wheel torque).
* **technician (rank 2)**: Adds access to `technician-sops/`, `machine-details/`, and `maintenance-work-orders/`.
* **engineer (rank 3)**: Adds access to `failure-analysis/`, `supplier-quality/`, and `training/`.
* **manager (rank 4)**: Adds full access to safety recalls (`recall-quality/`) and regulatory stop-ship logs.

*Security Note: Access levels are derived server-side from document folder paths, preventing metadata poisoning from caller input.*

### 2. Multi-Hop Hybrid Retrieval (Vector + GraphRAG)
* **Vector Store**: Semantic similarity search in ChromaDB, filtered by category and access level rank: `{"level_rank": {"$lte": user_level}}`.
* **GraphRAG**: Builds an entity-relationship graph (e.g. `SOP-OPR-101 -- requires --> TQ-6012`). Seeds are identified using cosine similarity of query embeddings, and a BFS expansion retrieves adjacent nodes while pruning any node with `level_rank > user_level`.

### 3. Human-in-the-Loop (HITL) Safety Gate
All high-consequence recommendation verbs (e.g., `stop-ship`, `quarantine`, `recall`, `halt line`) are intercepted. 
* Recommendations are persisted to `app_state.db` under the `decisions_log` audit trail.
* If a Class A (safety-critical) action is detected, the CLI prompts for a **mandatory approver ID** and requires justification on rejection.

---

## 🚀 Getting Started & Setup

### 1. Prerequisites
Ensure you have `uv` installed. If not, install it via:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Setup Environment
Copy the env template and set your valid **OpenRouter API Key** (or use local Ollama config if switching to `Local` execution environment):
```bash
cp .env.example .env
# Edit .env and fill in:
# OPENROUTER_API_KEY=your-real-openrouter-key
```

### 3. Install Dependencies
Sync project dependencies:
```bash
uv sync
```

### 4. Build the Retrieval Bases
Build the Chroma vector store and the GraphRAG knowledge graph:
```bash
# Ingests all 48 SOPs/specs with RBAC metadata
uv run python -m nga.ingestion.build_vector_store

# Extracts entity-relation links and saves the graph
uv run python -m nga.ingestion.build_graph
```

### 5. Start the Web User Interface (Recommended)
Launch the modern, responsive web dashboard:
```bash
uv run python -m nga.ui.runner --port 8000 --host 127.0.0.1 --open-browser
```
Access the dashboard at `http://127.0.0.1:8000` to interact with:
* **Role Login & Switching**: Switch clearance levels between Operator (L1), Technician (L2), Engineer (L3), and Plant Manager (L4).
* **Final Answer Display**: Direct answers, Class A safety alerts, ESC-402 escalation levels, QCR-501 recall criteria, and verified document/SQL citations.
* **HIL Approval Queue**: Review, authorize, or reject safety-critical actions with mandatory approver ID and Class A justification enforcement.
* **Real-Time Execution Logs**: Inspect the LangGraph state trace (`prepare` → `agent` ⇆ `tools` → `synthesis` → `hitl`) and SQL/Vector tool outputs.
* **Evaluation Runner & Comparison Studio**: Run benchmark suites and compare candidate vs baseline runs side-by-side with regression analysis.

### 6. Start the Interactive CLI (Alternative)
Launch the terminal chat loop:
```bash
uv run python -m nga.cli --role operator
```
*Tip: You can switch roles mid-session in the prompt by typing `role manager` or `role technician`.*

---

## 🖥️ Web User Interface Features

The NGA Web Interface (`src/nga/ui/`) is a full-featured dashboard designed for manufacturing operations:

| UI Module | Purpose & Capabilities |
|---|---|
| **💬 Assistant Chat & Final Answer** | Interactive decision support with structured responses: 🚨 **Class A Defect hazard banners**, ⚡ **ESC-402 escalation tags**, 🏷️ **QCR-501 recall criteria**, 📋 **Key findings**, and 🔍 **Verified citations** linking to SOP document IDs and SQL tables. |
| **🛡️ HIL Safety Gate & Queue** | Review pending high-consequence actions (`stop-ship`, `quarantine`, `halt line`). Enforces mandatory approver ID and engineering justification notes for Class A safety items before persisting to `app_state.db`. |
| **📜 System & Execution Logs** | Real-time LangGraph step trace feed, exact SQL queries executed against `nga.db` with row counts, ChromaDB retrieval scores, and a central filterable log console (`INFO`, `WARNING`, `ERROR`). |
| **🕸️ Graph Quality Metrics** | **Dashboard 1**: Quantitative ingestion-time monitoring for Knowledge Graphs: Entity Extraction F1, Relation Accuracy, Entity Alignment Success Rate, and Knowledge Coverage Rate against gold ground-truth facts. |
| **🪜 Stepped Evaluation Suite** | **Dashboard 2**: Cognitive difficulty breakdown across 4 tiers: L1 (Single-Hop Fact) → L2 (Two-Hop Relation) → L3 (Three-Hop Cross-Doc) → L4 (Multi-Domain Challenge) with reasoning degradation slope and failure isolation. |
| **📊 Evaluation Benchmark Runner** | Run evaluations across all 85 benchmark questions or targeted subsets (`retrieval`, `multi-hop`, `scenario`, `escalation-recall`, `sql`, `stress`, or quick 5-question smoke tests) with live progress tracking. |
| **⚖️ A/B Testing & Release Gates** | **Dashboard 3**: Benchmark candidate strategies against pure vector baselines, track Graph Uplift ($\Delta\%$), and validate **6 Hard Release Criteria** (Pass Rate $\ge 85\%$, L3/L4 Uplift $\ge +10\%$, zero regressions, 100% Class A recall, $\le 12\text{s}$ latency ceiling, SQL injection safe). |
| **🤖 Multi-Model Arena** | **Dashboard 4**: Multi-provider side-by-side benchmark comparing Claude 3.5 Sonnet, GPT-4o, Gemini 1.5 Flash/Pro, DeepSeek V3/R1, and Grok-2 for reasoning accuracy, latency, token costs, and **Pareto Efficiency Frontier**. |
| **📈 CI Trends & Historical Tracking** | Visualize evaluation trajectories across Git commits and CI builds: **Pass rate trends line chart (SVG)** with $\ge 70\%$ threshold target, category health breakdowns, and chronological commit ledger. |
| **👤 RBAC Clearance Matrix** | View clearance levels (1 to 4), permitted document folder paths, and operational role boundaries. |

---

## 📊 The Four Enterprise Evaluation Dashboards

The framework implements four production-grade evaluation dashboards to eliminate guesswork and subjective evaluation:

### 1. 🕸️ Dashboard 1: The Four Quantitative Graph Quality Metrics
Monitors graph extraction quality at ingestion time against the curated `eval-graph/gold_graph_benchmark.json` standard:
* **Entity Extraction Accuracy (F1)**: Quantifies precision, recall, and F1 of extracted manufacturing entities (`Machine`, `Station`, `Threshold`, `FaultCode`, `RecallCriteria`). Target: $\ge 90\%$.
* **Relation Extraction Accuracy**: Audits directed edges against the plant manufacturing ontology rules to block hallucinated connections. Target: $\ge 85\%$.
* **Entity Alignment Success Rate**: Verifies whether synonyms and alternative tool codes (e.g. `TF-6000` $\equiv$ `TQ-6018`, `AU-2025` $\equiv$ `Aurora`) correctly converge to the same global node ID. Target: $\ge 95\%$.
* **Knowledge Coverage Rate**: Audits whether core numerical facts, cure times, and torque specs in raw SOPs exist and are reachable in the graph. Target: $\ge 90\%$.

### 2. 🪜 Dashboard 2: The 4-Tier Stepped Evaluation Test Suite
Stratifies the 85 evaluation questions into 4 difficulty levels to isolate exact reasoning bottlenecks:
* **Level 1 — Single-Hop Fact Questions**: Tests baseline dense vector retrieval (Target: $\ge 95\%$, $< 2.5\text{s}$).
* **Level 2 — Two-Hop Explicit Relation Questions**: Tests direct graph connectivity and 1-step edge traversal (Target: $\ge 90\%$, $< 4.5\text{s}$).
* **Level 3 — Three-Hop Cross-Document Complex Reasoning**: Tests dual-path hybrid retrieval (Vector + Subgraph BFS) across $\ge 2$ documents (Target: $\ge 82\%$, $< 8.0\text{s}$).
* **Level 4 — Multi-Domain Comprehensive Challenge Questions**: Tests long-chain reasoning combining Text-to-SQL + SOP document search + Class A safety recall + RBAC boundaries + adversarial anti-hallucination probes (Target: $\ge 75\%$, $< 15.0\text{s}$).
* **Degradation Slope**: Tracks performance decay across hops: $\text{Slope} = \frac{\text{PassRate}(L1) - \text{PassRate}(L4)}{3}$ (Target: $\le 0.08$/tier).

### 3. ⚖️ Dashboard 3: A/B Testing & Hard Release Criteria Safeguards
Ensures code, embedding, or prompt modifications are only promoted to production when deterministic gains are proven:
* **Benchmark against Pure Vector Baseline**: Candidate runs undergo A/B testing against a control group with graph retrieval disabled (`graph=None`).
* **The 6 Hard Release Criteria**:
  1. Overall pass rate $\ge 85.0\%$.
  2. Graph uplift on L3/L4 complex tiers $\ge +10.0\%$ over pure vector baseline.
  3. Zero regressions allowed (no previously passing question may fail).
  4. 100% accuracy on Class A defect detection and QCR-501 recall triggers.
  5. Average response latency $\le 12.0\text{s}$.
  6. 100% protection against adversarial SQL injection attacks (`STR4`).
* **Instant Rollback**: Automatically flags rollback when candidate runs violate release criteria.

### 4. 🤖 Dashboard 4: Multi-Model A/B Evaluation Arena
Runs identical prompts across multiple frontier model providers via OpenRouter:
* Supported roster: **Claude 3.5 Sonnet**, **GPT-4o**, **Gemini 1.5 Pro / Flash**, **DeepSeek V3 / R1**, and **Grok-2**.
* **Pareto Frontier Optimization**: Identifies non-dominated models that maximize reasoning quality for minimal cost per 1,000 queries.
* **Side-by-Side Prompt Output Inspector**: Directly inspects differences in citation fidelity, Class A alerts, and reasoning depth across all models for any question.

---

## 🔄 CI Flow & Evaluation Tracking Over Time

The project integrates continuous evaluation tracking into GitHub Actions and local CI pipelines:

### 1. GitHub Actions Pipeline (`.github/workflows/ci.yml`)
On every `push` and `pull_request`:
* Runs linting (`ruff`) and automated unit/integration test suites.
* Executes benchmark evaluation suite across all 6 test categories.
* Ingests results into `reports/eval/history.json` with commit metadata (`commit_sha`, `branch`, `author`, `message`, `timestamp`).
* Computes deltas vs the previous commit and outputs a rich **`$GITHUB_STEP_SUMMARY`** Markdown report with category health tables.
* Renders standalone SVG trend charts (`reports/eval/trends.svg`) and uploads all evaluation artifacts.

### 2. Standalone CI Tracker CLI (`src/nga/evaluation/ci_tracker.py`)
Run the tracker manually to record runs or generate SVG trend charts:
```bash
# Record an eval report into the history ledger and render SVG chart
uv run python -m nga.evaluation.ci_tracker \
  --report-file reports/eval/20260823_143827.json \
  --generate-svg reports/eval/trends.svg \
  --threshold 0.70
```

### 3. Caching (docs/cache-design.md)
The agent has an optional, SQLite-backed, env-namespaced cache (`src/nga/cache/`):

| Layer | What it caches | Key scoping |
|---|---|---|
| **L0 emb** | text → embedding vector | model id, corpus version |
| **L1 retr** | query → RBAC-filtered chunks + graph evidence | query, categories, k, **user_level**, corpus version |
| **L2 sql** | validated SELECT → rows | canonical SQL, DB `data_version` etag |

- **Default off** (`CACHE_ENABLED=false`) — zero behavior change when disabled.
- **Eval is cache-cold by default**; run `--cache-mode hot` (or `cache_mode: "hot"` in the eval API) to measure realistic repeated-query behavior against `data/cache/eval_cache.db`.
- Eval reports record `cache_mode` + per-layer hit rates; `compare_evaluation_runs` emits a `cache_improvement` block (`latency_reduction_pct`, `quality_parity`, `hit_rate`) for the paired cold/hot protocol.
- Maintenance: `uv run python -m nga.cache.cli stats|clear --env prod|eval [--layer emb|retr|sql]`

---

## 🧪 Comprehensive Test Suites & Benchmarks

The project includes an extensive automated test framework covering unit APIs, integration benchmarks, adversarial stress testing, and CI tracking:

### 1. UI Backend & Dashboard Integration Suite (`tests/test_ui_api.py`)
Tests all REST endpoints, RBAC switching, HIL approval flows, and report comparison engines:
```bash
uv run pytest tests/test_ui_api.py -v
```

### 2. Evaluation Engine Test Suites
* **Graph Quality Metrics Suite**: `uv run pytest tests/test_graph_eval.py -v`
* **4-Tier Stepped Suite**: `uv run pytest tests/test_stepped_suite.py -v`
* **A/B Testing & Release Gates**: `uv run pytest tests/test_release_gate.py -v`
* **Multi-Model Arena & Pareto Frontier**: `uv run pytest tests/test_multi_model_eval.py -v`
* **CI Evaluation Tracker**: `uv run pytest tests/test_ci_tracker.py -v`

### 3. Stress & Adversarial Test Suite (`tests/integration/test_stress.py`)
Stress tests concurrency, high load, and adversarial security:
```bash
uv run pytest tests/integration/test_stress.py -v
```

### 4. Running All Automated Unit Tests (108 Tests)
```bash
uv run pytest tests/test_*.py -v
```

---

## 📂 Corpus Universe & Document Index (100% Synthetic Benchmark)

The NGA corpus represents a **fully synthetic, simulated automotive manufacturing universe** designed to benchmark agentic decision support systems. It models the fictional **Apex Automotive — Northgate Assembly Plant (NGA)** manufacturing compact SUVs (*Aurora AU-2025*) and sedans (*Solstice SO-2025*) at a rate of 42 vehicles/hour across three shifts (A, B, C). No real-world manufacturer data, proprietary specs, or confidential plant records are contained in this repository.

### Document Directory Structure:
* [`operator-sops/`](file:///Users/ruiping/projects/manufacturing_agent/operator-sops/): Wheel torques (105 Nm ±5%), adhesive open times, and standard final audits.
* [`technician-sops/`](file:///Users/ruiping/projects/manufacturing_agent/technician-sops/): Weld fault diagnosis, conveyor jam recovery, and paint robot P-codes.
* [`machine-details/`](file:///Users/ruiping/projects/manufacturing_agent/machine-details/): Robot reach specifications and calibration intervals (3 months / 100k cycles).
* [`failure-analysis/`](file:///Users/ruiping/projects/manufacturing_agent/failure-analysis/): 8D problem-solving (FAP-401) and escalation timelines (ESC-402).
* [`recall-quality/`](file:///Users/ruiping/projects/manufacturing_agent/recall-quality/): Recall criteria (QCR-501) for safety, defect rates, and regulatory reporting.
* [`additional-docs/`](file:///Users/ruiping/projects/manufacturing_agent/additional-docs/): Maintenance work orders, supplier quality audits, and personnel certifications.
* [`eval-questions/`](file:///Users/ruiping/projects/manufacturing_agent/eval-questions/): 85 benchmark questions across Retrieval, Multi-hop, Scenario, Escalation & Recall, SQL, and Stress testing.
* [`database/`](file:///Users/ruiping/projects/manufacturing_agent/database/): Seeded SQLite database (`nga.db`) containing production, quality checks, work orders, and training histories.

