# OpenTelemetry Tracing & Observability Guide

The NGA Manufacturing Assistant uses **OpenTelemetry** (`src/nga/tracing.py`) for vendor-neutral distributed tracing, span hierarchy instrumentation, and System 1 / System 2 token and cost telemetry.

---

## 1. Quick Start: Configuring the Trace Provider

Tracing is controlled via environment variables in `.env` (loaded by `Settings.from_env()` in `src/nga/config.py` and initialized by `configure_tracing()` in `src/nga/tracing.py`).

### Environment Variables Reference

| Environment Variable | Default | Supported Values | Description |
| :--- | :--- | :--- | :--- |
| `OTEL_TRACING_ENABLED` | `false` | `true`, `false`, `1`, `0` | Master toggle to enable or disable OpenTelemetry span recording. |
| `OTEL_SERVICE_NAME` | `nga-manufacturing-assistant` | Any string | Sets the OpenTelemetry `service.name` resource attribute on all spans. |
| `OTEL_TRACES_EXPORTER` | `otlp` (when endpoint set) or `memory` | `otlp`, `console`, `memory`, `none` | Selects which span processor/exporter to attach to the `TracerProvider`. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `""` | e.g., `http://localhost:4317` | OTLP gRPC collector endpoint used when `OTEL_TRACES_EXPORTER=otlp`. |
| `OTEL_EXPORTER_OTLP_HEADERS` | *(unset)* | `key1=val1,key2=val2` | Optional authentication headers read automatically by the OTLP exporter for managed cloud backends. |

> **Note:** `configure_tracing()` explicitly sets `LANGCHAIN_TRACING_V2=false` and `LANGSMITH_TRACING=false` so LangChain and LangGraph remain completely decoupled from LangSmith.

---

## 2. Where to View Traces

Depending on your environment and `OTEL_TRACES_EXPORTER` setting, you can view traces in any of the following interfaces:

### Option A: Arize Phoenix (Recommended for LLM & Agent Workflows)
[Arize Phoenix](https://phoenix.arize.com/) natively understands `openinference.span.kind` (`CHAIN`, `TOOL`, `RETRIEVER`, `LLM`, `EMBEDDING`) and `gen_ai.usage.*` token attributes emitted by NGA.

1. Start Phoenix via Docker:
   ```bash
   docker run -d --name phoenix -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest
   ```
2. Configure `.env`:
   ```ini
   OTEL_TRACING_ENABLED=true
   OTEL_SERVICE_NAME=nga-manufacturing-assistant
   OTEL_TRACES_EXPORTER=otlp
   OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
   ```
3. View traces at **`http://localhost:6006`**.

### Option B: Jaeger (Standard Distributed Tracing UI)
1. Start Jaeger All-in-One via Docker:
   ```bash
   docker run -d --name jaeger -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one:latest
   ```
2. Configure `.env`:
   ```ini
   OTEL_TRACING_ENABLED=true
   OTEL_SERVICE_NAME=nga-manufacturing-assistant
   OTEL_TRACES_EXPORTER=otlp
   OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
   ```
3. Open **`http://localhost:16686`**, select service **`nga-manufacturing-assistant`**, and click **Find Traces**.

### Option C: Cloud OTLP Backends (Grafana Tempo, Honeycomb, Datadog, SigNoz)
Point `OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_EXPORTER_OTLP_HEADERS` to your cloud collector:
```ini
OTEL_TRACING_ENABLED=true
OTEL_SERVICE_NAME=nga-manufacturing-assistant
OTEL_TRACES_EXPORTER=otlp
OTEL_EXPORTER_OTLP_ENDPOINT=https://api.honeycomb.io:443
OTEL_EXPORTER_OTLP_HEADERS=x-honeycomb-team=your-api-key
```

### Option D: Built-in NGA Web UI Sidebar (No Collector Required)
When running the web server (`uv run python -m nga.ui.runner`), the right-hand **⚡ Execution Trace** sidebar at `http://127.0.0.1:8000` displays real-time orchestrator node transitions (`prepare`, `agent`, `tools`, `synthesis`, `hitl`) and tool outputs directly in the browser, even when `OTEL_TRACING_ENABLED=false`.

### Option E: Terminal Output (`console`) or Programmatic Inspection (`memory`)
- **Console output:** Set `OTEL_TRACES_EXPORTER=console` with `OTEL_TRACING_ENABLED=true` to print formatted JSON spans directly to `stdout`.
- **In-memory inspection:** Set `OTEL_TRACES_EXPORTER=memory` (or omit `OTEL_EXPORTER_OTLP_ENDPOINT`) and retrieve recorded spans in Python:
  ```python
  from nga.tracing import clear_in_memory_spans, get_in_memory_spans

  spans = get_in_memory_spans()
  for span in spans:
      print(span.name, dict(span.attributes or {}))
  clear_in_memory_spans()
  ```

---

## 3. Instrumented Span Hierarchy & Attributes

All functions decorated with `@traceable_if_enabled` emit OpenTelemetry spans when tracing is enabled:

| Span Name | `nga.run_type` | `openinference.span.kind` | Description |
| :--- | :--- | :--- | :--- |
| `orchestrator.prepare` | `chain` | `CHAIN` | Input safety guard, tier routing, speculative fan-out planning, episodic memory recall. |
| `orchestrator.guard_rejection` | `chain` | `CHAIN` | Adversarial / out-of-scope query rejection node. |
| `orchestrator.agent` | `chain` | `CHAIN` | Iterative tool-calling reasoning loop with Jev evidence sufficiency checks. |
| `orchestrator.synthesis` | `chain` | `CHAIN` | Final answer synthesis, Jev fact groundedness verification, contradiction resolution, and token telemetry attachment. |
| `orchestrator.hitl` | `chain` | `CHAIN` | Human-in-the-loop safety approval gate. |
| `tool:<tool_name>` | `tool` | `TOOL` | Wrapped LangChain tool invocation (`sql.run_query`, `sql.describe_schema`, retrieval tools). |
| `retrieval.retrieve_documents` | `retriever` | `RETRIEVER` | Vector store similarity search and System 1 Jev reranking. |
| `retrieval.retrieve_graph_evidence` | `retriever` | `RETRIEVER` | GraphRAG BFS multi-hop evidence traversal. |
| `jev.*` | `chain` | `CHAIN` | System 1 TypeSafe Jev evaluations (`check_input_safety`, `evaluate_evidence_sufficiency`, `evaluate_fact_groundedness`, `detect_cross_source_contradiction`, etc.). |
| `classifier.*` | `chain` | `CHAIN` | Query complexity scoring and tier model routing. |
| `cache.*` | `chain` | `CHAIN` | L0/L1/L2 cache lookup and store operations. |

### Token & Cost Telemetry Attributes (`orchestrator.synthesis`)

During `orchestrator.synthesis`, `attach_token_telemetry()` records the following attributes on the active span:

- `gen_ai.usage.input_tokens` / `tokens_llm_prompt`: System 2 LLM prompt tokens
- `gen_ai.usage.output_tokens` / `tokens_llm_completion`: System 2 LLM completion tokens
- `tokens_jev_input`: System 1 TypeSafe Jev input tokens
- `tokens_total`: Combined token count across System 1 and System 2
- `cost_total_usd`, `cost_llm_usd`, `cost_jev_usd`: USD cost breakdown
- `tcer`: Token Cost Efficiency Ratio
- `token_savings_pct`, `cost_savings_pct`: Savings achieved via Jev early exit and routing
- `early_exit_triggered`: Boolean flag indicating whether Jev sufficiency check avoided extra LLM rounds
- `tool_rounds_executed`: Number of tool rounds executed
- `nga.tags`: List of span tags (e.g., `["early-exit", "tcer:0.375"]`)
