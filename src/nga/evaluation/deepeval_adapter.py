"""DeepEval test case conversion layer for NGA Manufacturing Assistant benchmarks.

Maps NGA benchmark rows (`eval-questions/questions.json`), cross-document
contradiction scenarios (`CONFLICT_SCENARIOS`), and agent execution outputs
into DeepEval `LLMTestCase` and `ToolCall` instances:
- `question` -> `input`
- generated/candidate answer -> `actual_output`
- gold `expected_answer` -> `expected_output`
- retrieved chunks / SQL results / fallback context -> `retrieval_context`
- executed tool calls and payloads -> `tools_called` (`list[ToolCall]`)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from deepeval.test_case import LLMTestCase, ToolCall

logger = logging.getLogger("nga.evaluation.deepeval_adapter")


def _format_retrieval_chunk(item: dict[str, Any]) -> str:
    """Format a single retrieval result item into a readable context string."""
    doc_id = item.get("doc_id") or ""
    section = item.get("section") or ""
    category = item.get("category") or ""
    source = item.get("corpus_source", "canonical")
    text = (item.get("text") or "").strip()

    prefix_parts: list[str] = []
    if category:
        prefix_parts.append(f"[{category}]")
    if source == "variant":
        prefix_parts.append("[VARIANT]")
    if doc_id:
        prefix_parts.append(f"[{doc_id}]")
    if section:
        prefix_parts.append(f"§{section}")

    header = " ".join(prefix_parts)
    if header and text:
        return f"{header} {text}"
    return text or header


def _format_graph_evidence_chunks(graph_evidence: dict[str, Any]) -> list[str]:
    """Extract knowledge-graph entities, relations, and community summaries as context strings."""
    chunks: list[str] = []
    if not isinstance(graph_evidence, dict):
        return chunks

    entities = graph_evidence.get("entities") or []
    if isinstance(entities, list) and entities:
        ent_lines = []
        for ent in entities:
            if isinstance(ent, dict):
                name = ent.get("name") or ent.get("id") or ""
                etype = ent.get("type") or ""
                desc = ent.get("description") or ""
                source_docs = ent.get("source_docs") or []
                src_str = f" (sources: {', '.join(map(str, source_docs))})" if source_docs else ""
                if name:
                    ent_lines.append(f"- {name} [{etype}]: {desc}{src_str}".strip())
        if ent_lines:
            chunks.append("[GRAPH ENTITIES]\n" + "\n".join(ent_lines))

    relations = graph_evidence.get("relations") or []
    if isinstance(relations, list) and relations:
        rel_lines = []
        for rel in relations:
            if isinstance(rel, dict):
                src = rel.get("source") or ""
                tgt = rel.get("target") or ""
                rtype = rel.get("relation") or rel.get("type") or "RELATED_TO"
                desc = rel.get("description") or ""
                if src and tgt:
                    line = f"- {src} --[{rtype}]--> {tgt}"
                    if desc:
                        line += f": {desc}"
                    rel_lines.append(line)
        if rel_lines:
            chunks.append("[GRAPH RELATIONS]\n" + "\n".join(rel_lines))

    summaries = graph_evidence.get("community_summaries") or []
    if isinstance(summaries, list):
        for s in summaries:
            if isinstance(s, str) and s.strip():
                chunks.append(f"[GRAPH COMMUNITY SUMMARY] {s.strip()}")
            elif isinstance(s, dict) and s.get("summary"):
                chunks.append(f"[GRAPH COMMUNITY SUMMARY] {str(s['summary']).strip()}")

    return chunks


def _format_sql_payload_chunk(payload: dict[str, Any]) -> str:
    """Format a SQL tool output payload (`{query, rows, row_count, tables}`) into a context string."""
    query = payload.get("query", "")
    rows = payload.get("rows", [])
    row_count = payload.get("row_count", len(rows) if isinstance(rows, list) else 0)
    tables = payload.get("tables", [])

    header_parts = ["[SQL RESULT]"]
    if tables:
        header_parts.append(f"tables={','.join(map(str, tables))}")
    header_parts.append(f"row_count={row_count}")
    if query:
        header_parts.append(f"query={query}")

    rows_repr = json.dumps(rows, ensure_ascii=False, default=str)
    return f"{' | '.join(header_parts)}\nRows: {rows_repr}"


def extract_retrieval_context(
    tool_outputs: list[str] | None = None,
    question_data: dict[str, Any] | None = None,
    fallback_to_expected: bool = True,
) -> list[str]:
    """Extract normalized `retrieval_context` strings from tool outputs or static benchmark data.

    Handles:
    1. `search_sop_documents` JSON payloads (`results` passages + `graph_evidence`)
    2. `query_nga_database` JSON payloads (`query`, `rows`, `row_count`, `tables`)
    3. Raw text tool outputs (e.g., `retrieve_uncompressed_evidence` or error strings)
    4. Static fallback context when evaluating pre-generated/perturbed candidate answers
       without live tool execution (using `retrieval_context`, `context`, or `expected_answer`
       + `source_docs` from `question_data`).
    """
    contexts: list[str] = []

    for raw_output in tool_outputs or []:
        if not raw_output or not str(raw_output).strip():
            continue
        raw_str = str(raw_output).strip()
        try:
            payload = json.loads(raw_str)
        except (TypeError, ValueError):
            contexts.append(raw_str)
            continue

        if not isinstance(payload, dict):
            contexts.append(raw_str)
            continue

        # Case 1: Document retrieval tool payload
        if "results" in payload and isinstance(payload["results"], list):
            for item in payload["results"]:
                if isinstance(item, dict):
                    formatted = _format_retrieval_chunk(item)
                    if formatted:
                        contexts.append(formatted)
                elif isinstance(item, str) and item.strip():
                    contexts.append(item.strip())
            if payload.get("graph_evidence"):
                contexts.extend(_format_graph_evidence_chunks(payload["graph_evidence"]))
            continue

        # Case 2: SQL database tool payload
        if "query" in payload and "rows" in payload:
            contexts.append(_format_sql_payload_chunk(payload))
            continue

        # Case 3: Generic JSON dictionary output
        contexts.append(json.dumps(payload, ensure_ascii=False, default=str))

    # Fallback for static benchmark evaluations where no live tools were run
    if not contexts and question_data:
        explicit_ctx = question_data.get("retrieval_context") or question_data.get("context")
        if isinstance(explicit_ctx, list):
            contexts.extend(str(c) for c in explicit_ctx if str(c).strip())
        elif isinstance(explicit_ctx, str) and explicit_ctx.strip():
            contexts.append(explicit_ctx.strip())

        if not contexts and fallback_to_expected:
            expected = (
                question_data.get("expected_answer")
                or question_data.get("golden")
                or ""
            ).strip()
            source_docs = question_data.get("source_docs") or []
            if expected:
                if source_docs:
                    src_prefix = " ".join(f"[{doc}]" for doc in source_docs)
                    contexts.append(f"{src_prefix} {expected}")
                else:
                    contexts.append(expected)

    return contexts


def build_deepeval_tool_calls(
    tools_called: list[str] | None = None,
    tool_outputs: list[str] | None = None,
    tool_calls_raw: list[dict[str, Any]] | None = None,
) -> list[ToolCall]:
    """Convert NGA tool call names, raw metadata, and outputs into DeepEval `ToolCall` objects."""
    result: list[ToolCall] = []

    if tool_calls_raw:
        for idx, tc in enumerate(tool_calls_raw):
            name = tc.get("name") or (
                tools_called[idx] if tools_called and idx < len(tools_called) else "unknown_tool"
            )
            input_params = tc.get("args") or tc.get("input_parameters")
            if not isinstance(input_params, dict):
                input_params = {"input": input_params} if input_params is not None else None

            raw_out = tc.get("output")
            if raw_out is None and tool_outputs and idx < len(tool_outputs):
                raw_out = tool_outputs[idx]

            parsed_out: Any = raw_out
            if isinstance(raw_out, str):
                try:
                    parsed_out = json.loads(raw_out)
                except (TypeError, ValueError):
                    parsed_out = raw_out

            result.append(
                ToolCall(
                    name=str(name),
                    input_parameters=input_params,
                    output=parsed_out,
                )
            )
        return result

    names = list(tools_called or [])
    outputs = list(tool_outputs or [])

    if not names and outputs:
        # Infer tool names from payload shapes when only outputs were provided
        for out_str in outputs:
            inferred_name = "tool_execution"
            parsed_out: Any = out_str
            try:
                parsed = json.loads(out_str)
                parsed_out = parsed
                if isinstance(parsed, dict):
                    if "query" in parsed and "rows" in parsed:
                        inferred_name = "query_nga_database"
                    elif "results" in parsed or "references" in parsed:
                        inferred_name = "search_sop_documents"
            except (TypeError, ValueError):
                pass
            result.append(ToolCall(name=inferred_name, output=parsed_out))
        return result

    for idx, name in enumerate(names):
        raw_out = outputs[idx] if idx < len(outputs) else None
        parsed_out = raw_out
        input_params: dict[str, Any] | None = None
        if isinstance(raw_out, str):
            try:
                parsed_out = json.loads(raw_out)
                if isinstance(parsed_out, dict):
                    if "query" in parsed_out and "rows" in parsed_out:
                        input_params = {"sql": parsed_out["query"]}
                    elif "query" in parsed_out and "categories" in parsed_out:
                        input_params = {
                            "query": parsed_out["query"],
                            "categories": parsed_out.get("categories"),
                        }
            except (TypeError, ValueError):
                parsed_out = raw_out
        result.append(
            ToolCall(
                name=str(name),
                input_parameters=input_params,
                output=parsed_out,
            )
        )

    return result


def build_expected_tool_calls(question_data: dict[str, Any]) -> list[ToolCall]:
    """Derive expected DeepEval `ToolCall` entries from a benchmark question row."""
    expected: list[ToolCall] = []
    if question_data.get("requires_sql"):
        expected.append(ToolCall(name="query_nga_database"))
    if question_data.get("source_docs"):
        expected.append(ToolCall(name="search_sop_documents"))
    return expected


def to_deep_eval_test_case(
    question_data: dict[str, Any],
    actual_output: str,
    tool_outputs: list[str] | None = None,
    tools_called: list[str] | None = None,
    tool_calls_raw: list[dict[str, Any]] | None = None,
    retrieval_context: list[str] | None = None,
    latency_s: float | None = None,
    token_cost: float | None = None,
    fallback_to_expected_context: bool = True,
) -> LLMTestCase:
    """Convert an NGA benchmark question and candidate/agent output into a DeepEval `LLMTestCase`.

    Mapping:
    - `question` -> `input`
    - `actual_output` (generated answer) -> `actual_output`
    - `expected_answer` (or `golden`) -> `expected_output`
    - retrieved chunks / tool outputs -> `retrieval_context`
    - `tools_called` + `tool_outputs` -> `tools_called` (`list[ToolCall]`)
    """
    q_id = str(question_data.get("id", "UNKNOWN"))
    category = str(question_data.get("category", "general"))
    tier = str(question_data.get("tier", "L2"))
    question_text = str(
        question_data.get("question")
        or question_data.get("name")
        or question_data.get("input")
        or q_id
    )
    expected_output = str(
        question_data.get("expected_answer")
        or question_data.get("golden")
        or question_data.get("expected_output")
        or ""
    )

    effective_tool_outputs = tool_outputs
    if effective_tool_outputs is None and tool_calls_raw:
        effective_tool_outputs = [
            tc.get("output") if isinstance(tc.get("output"), str) else json.dumps(tc.get("output"))
            for tc in tool_calls_raw
            if tc.get("output") is not None
        ]

    resolved_context = (
        list(retrieval_context)
        if retrieval_context is not None
        else extract_retrieval_context(
            tool_outputs=effective_tool_outputs,
            question_data=question_data,
            fallback_to_expected=fallback_to_expected_context,
        )
    )

    deepeval_tools = build_deepeval_tool_calls(
        tools_called=tools_called,
        tool_outputs=effective_tool_outputs,
        tool_calls_raw=tool_calls_raw,
    )
    expected_tools = build_expected_tool_calls(question_data)

    metadata: dict[str, Any] = {
        "question_id": q_id,
        "category": category,
        "tier": tier,
        "source_docs": question_data.get("source_docs", []),
        "requires_sql": bool(question_data.get("requires_sql", False)),
    }
    for k in ("condition", "difficulty", "target_capability", "scenario_id", "scenario_name"):
        if k in question_data:
            metadata[k] = question_data[k]

    return LLMTestCase(
        name=q_id,
        input=question_text,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=resolved_context,
        tools_called=deepeval_tools,
        expected_tools=expected_tools,
        metadata=metadata,
        tags=[category, tier],
        completion_time=latency_s,
        token_cost=token_cost,
    )


def conflict_scenario_to_test_cases(
    scenario: dict[str, Any],
) -> dict[str, LLMTestCase]:
    """Convert a cross-document contradiction scenario (`CONFLICT_SCENARIOS` item)
    into a dictionary of condition -> `LLMTestCase` (`conflict_flagged`, `blind_variant`, `hallucinated_middle`).
    """
    sid = scenario.get("id", "CONFLICT-UNKNOWN")
    name = scenario.get("name", "")
    golden = scenario.get("golden", "")
    answers = scenario.get("answers", {})

    cases: dict[str, LLMTestCase] = {}
    for condition, candidate_answer in answers.items():
        q_row = {
            "id": f"{sid}:{condition}",
            "category": "contradiction",
            "tier": "L3",
            "question": f"Resolve specification inquiry for: {name}",
            "expected_answer": golden,
            "retrieval_context": [golden],
            "condition": condition,
            "scenario_id": sid,
            "scenario_name": name,
        }
        cases[condition] = to_deep_eval_test_case(
            question_data=q_row,
            actual_output=candidate_answer,
            retrieval_context=[golden],
        )
    return cases
