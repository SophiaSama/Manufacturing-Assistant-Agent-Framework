# Evaluation Architecture & Dashboards Design Specification

**Status:** Approved & Implemented  
**Date:** 2026-09-08  
**Scope:** Offline Graph Quality Ingestion, 4-Tier Stepped Reasoning, A/B Testing & Release Gates, Multi-Model Pareto Arena  

---

## 1. Executive Summary

This document specifies the enterprise evaluation architecture and real-time dashboard suite for the **NGA Manufacturing Assistant**. It addresses the four core dimensions of knowledge-grounded agent evaluation:
1. **The Four Quantitative Graph Quality Metrics**: Ingestion-time monitoring of entity extraction accuracy, relation validity, alias alignment, and knowledge coverage to prevent silent graph degradation.
2. **The 4-Tier Stepped Evaluation Test Suite**: Cognitive stratification (L1 Single-Hop → L2 Two-Hop → L3 Three-Hop Cross-Doc → L4 Multi-Domain Complex) to isolate retrieval vs reasoning bottlenecks.
3. **A/B Testing & Regression Safeguards**: Benchmarking hybrid GraphRAG strategies against pure vector baselines with automated 6-point Hard Release Criteria and fast rollback mechanisms.
4. **Multi-Model A/B Evaluation**: Multi-provider side-by-side benchmarking across frontier models (Claude 3.5 Sonnet, GPT-4o, Gemini, DeepSeek, Grok) for quality, latency, token economics, and Pareto frontier optimization.

---

## 2. Dashboard 1: The Four Quantitative Graph Quality Metrics

### 2.1 Core Metrics & Mathematical Formulations

| Metric | Mathematical Definition | Target Threshold | Purpose |
| :--- | :--- | :--- | :--- |
| **Entity Extraction Accuracy (F1)** | Precision = |E_ext ∩ E_gold| / |E_ext|, Recall = |E_ext ∩ E_gold| / |E_gold|<br>F1 = 2 * (Precision * Recall) / (Precision + Recall) | ≥ 90.0% | Prevents entity omissions and hallucinated spurious entities. |
| **Relation Extraction Accuracy** | Rel Acc = (# Valid Triples Checked) / (# Triples Audited) | ≥ 85.0% | Verifies that directed edges adhere to the manufacturing ontology. |
| **Entity Alignment Success Rate** | Align Rate = (Σ I(canonical(a) == canonical(b))) / |S| | ≥ 95.0% | Ensures synonyms (e.g. TF-6000 and TQ-6018) converge to identical global node IDs. |
| **Knowledge Coverage Rate** | Coverage Rate = (Σ I(Graph contains f)) / |F_gold| | ≥ 90.0% | Audits whether core numerical thresholds and safety rules exist in the graph. |

### 2.2 Ground Truth Benchmark
Located at `eval-graph/gold_graph_benchmark.json`, containing human-verified annotations for 10 representative plant SOPs, including:
- Standard Operating Procedures (`SOP-OPR-101`, `SOP-OPR-114`, `SOP-OPR-142`, `SOP-OPR-158`)
- Technician Procedures & Machine Specs (`SOP-TEC-201`, `SOP-TEC-214`, `TEC-301`, `TEC-314`, `TEC-328`)
- Quality & Regulatory Standards (`QCR-501`, `ESC-402`, `FAP-401`)

### 2.3 Canonical Entity Registry, Sanitization & Hybrid GraphRAG Search
- **Ontology & Canonical Entity Registry (`src/nga/ingestion/normalize.py`)**: Enforces 15 entity types (`NGA_ENTITY_TYPES`), 12 directed relation rules (`NGA_ONTOLOGY_RULES`, including `owned_by` and `certified_for`), and `CANONICAL_ENTITY_REGISTRY` to deterministically resolve plant synonyms (`Aurora`/`AU25` $\rightarrow$ `aurora-au-2025`, `TorqMaster TF-6000` $\rightarrow$ `tq-6018`, `Class A` $\rightarrow$ `class-a-defect`).
- **Conflict-Safe Entity Alignment (`src/nga/ingestion/entity_alignment.py`)**: Uses alias-aware blocking, `_has_conflicting_codes()` equipment/SOP number conflict guards (`tq-6012` vs. `tq-6018`), and TypeSafe AI System One (`Score` + `Noul`) judgments.
- **Graph Sanitization (`sanitize_graph` in `src/nga/ingestion/build_graph.py`)**: Upgrades persisted graphs to directed `nx.MultiDiGraph`, auto-reverses inverted relations, attaches chunk provenance, prunes isolated noise nodes, and computes Louvain/modularity `community` IDs.
- **Hybrid Lexical + Semantic GraphRAG Retrieval (`src/nga/graphrag/search.py`)**: Combines canonical ID/alias resolution, lexical token overlap (`_lexical_node_score`), and vector cosine similarity for seed node selection before 2-hop RBAC-filtered BFS expansion.

### 2.4 Interactive Knowledge Graph Topology Visualizer (`GET /api/graph/topology`)
The **🕸️ Graph Quality** tab in the Web UI (`src/enterprise_agent/ui/static/index.html`, `app.js`) embeds an interactive force-directed HTML5 canvas backed by `GET /api/graph/topology`:
- **Filtering & Ego-Network Search**: Filter by RBAC clearance (`user_level=1..4`), entity type (`NGA_ENTITY_TYPES`), or search query (expands a configurable $k$-hop neighborhood around matching seed nodes).
- **Node & Provenance Inspector**: Clicking any node highlights its 1-hop neighborhood and displays its canonical ID, bound synonym aliases, RBAC rank, incoming/outgoing relation triples, and source document chunks.

---

## 3. Dashboard 2: 4-Tier Stepped Evaluation Test Suite (阶梯式评测集)

To diagnose degradation as reasoning chains grow longer, the 85-question benchmark is partitioned into four distinct tiers:

- **Level 1: Single-Hop Fact**: Direct semantic similarity & dense vector retrieval (Target: ≥ 95%, < 2.5s)
- **Level 2: Two-Hop Explicit Relation**: 1-step graph traversal (Tool → SOP → Spec, Code → Cause → Action) (Target: ≥ 90%, < 4.5s)
- **Level 3: Three-Hop Cross-Doc Reasoning**: Multi-document hybrid retrieval (Vector + Graph BFS ≥ 2 hops) (Target: ≥ 82%, < 8.0s)
- **Level 4: Multi-Domain Complex Challenge**: Complex state: Text-to-SQL + SOP + Safety Recall + RBAC + Anti-Hallucination (Target: ≥ 75%, < 15.0s)

### 3.1 Failure Mode Isolation
- **L1 Failure**: Chunk boundary fragmentation or embedding dimension/distance mismatch.
- **L2 Failure**: Missing relational triple or incorrect entity linking.
- **L3 Failure**: BFS search truncation, low cosine similarity seed cutoff, or incomplete cross-document linking.
- **L4 Failure**: Multi-tool state orchestration deadlock, SQL generation error, or RBAC clearance leak.

---

## 4. Dashboard 3: A/B Testing & Hard Release Criteria Safeguards

### 4.1 Control vs Treatment Protocol
- **Control Group (A - Pure Vector)**: ChromaDB dense similarity search with `graph=None`.
- **Treatment Group (B - Candidate)**: Full Tri-Hybrid Retrieval (Dense + NetworkX BFS Expansion + RRF Reranking).

### 4.2 The 6 Hard Release Criteria
Code and graph updates are strictly gated against automated regression checks:
1. **Overall Pass Rate**: PassRate_B ≥ 85.0%.
2. **Graph Uplift on L3/L4**: Δ_Score(L3, L4) ≥ +10.0% over Pure Vector.
3. **Zero Regressions**: No question that previously passed in Baseline A may fail in Candidate B (Regressions = 0).
4. **Class A Safety Invariant**: 100% detection of Class A defects and QCR-501 recall triggers.
5. **Latency Ceiling**: Average response latency ≤ 12.0s.
6. **SQL Injection Resilience**: 100% rejection rate on adversarial SQL injection probes.

---

## 5. Dashboard 4: Multi-Model A/B Evaluation

### 5.1 Multi-Provider Roster
- Anthropic Claude 3.5 Sonnet (`anthropic/claude-3.5-sonnet`)
- OpenAI GPT-4o (`openai/gpt-4o`)
- Google Gemini 1.5 Pro / Flash (`google/gemini-1.5-pro`, `google/gemini-1.5-flash`)
- DeepSeek V3 / R1 (`deepseek/deepseek-chat`, `deepseek/deepseek-r1`)
- xAI Grok-2 (`x-ai/grok-2`)

### 5.2 Pareto Optimization
Plots **Reasoning Score** against **Cost per 1,000 Questions ($)** to identify optimal production configurations (e.g. high-throughput triage vs high-consequence escalation).
