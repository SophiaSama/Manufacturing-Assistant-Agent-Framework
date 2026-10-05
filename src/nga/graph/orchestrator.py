"""LangGraph orchestrator for the NGA Manufacturing Assistant.

Workflow: prepare → agent ⇆ tools → synthesis → hitl → END

Roles feed role-specific system prompts and RBAC-filtered retrieval.
The synthesis node detects Class A defects and recall criteria automatically.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from nga.config import Settings
from nga.graph.nodes import (
    extract_question_parts,
    extract_recommendation,
    is_class_a_defect,
)
from nga.graph.state import AgentState
from nga.hitl.approval import request_approval
from nga.memory.decision_log import insert_recommendation, update_decision
from nga.models.answer_schema import (
    FinalAnswer,
    parse_final_answer,
    render_final_answer,
)
from nga.providers.factory import make_chat_model
from nga.rag_agent.classifier import classify_model_route
from nga.rag_agent.jev_reasoning import (
    FactGroundednessResult,
    calculate_reasoning_token_telemetry,
    check_input_safety,
    evaluate_evidence_sufficiency,
    evaluate_fact_groundedness,
    plan_speculative_fanout,
    screen_evidence_contradictions,
)
from nga.rag_agent.rbac import SYSTEM_PROMPTS
from nga.tools.sql_tool import describe_schema
from nga.tracing import (
    attach_token_telemetry,
    attach_trace_metadata,
    traceable_if_enabled,
)

logger = logging.getLogger("nga.orchestrator")

MAX_TOOL_CALL_ROUNDS = 8
MAX_IDENTICAL_TOOL_CALL_REPEATS = 3


def _truncate_for_log(value: Any, limit: int = 400) -> str:
    text = str(value)
    return text if len(text) <= limit else f"{text[:limit]}... [+{len(text)-limit} chars]"


def _wrap_tool_with_logging(tool_obj: BaseTool) -> BaseTool:
    @traceable_if_enabled(run_type="tool", name=f"tool:{tool_obj.name}")
    def _logged(**kwargs):
        logger.info("tool_call_start name=%s input=%s", tool_obj.name, _truncate_for_log(kwargs))
        try:
            result = tool_obj.invoke(kwargs)
        except Exception:
            logger.exception("tool_call_error name=%s", tool_obj.name)
            raise
        logger.info("tool_call_end name=%s output=%s", tool_obj.name, _truncate_for_log(result))
        return result

    return StructuredTool.from_function(
        func=_logged,
        name=tool_obj.name,
        description=tool_obj.description,
        args_schema=tool_obj.args_schema,
        return_direct=tool_obj.return_direct,
    )


# ── Node: prepare ─────────────────────────────────────────────────────────────

def _make_prepare_node(settings: Settings, episodic_memory=None):
    @traceable_if_enabled(run_type="chain", name="orchestrator.prepare")
    def _prepare_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        question = last.content if hasattr(last, "content") else str(last)

        guard = check_input_safety(question)
        logger.info(
            "input_guard_evaluated is_allowed=%s category=%s prob=%.2f reason=%s",
            guard.is_allowed,
            guard.intent_category,
            guard.is_malicious_prob,
            guard.rejection_reason,
        )

        route_info = classify_model_route(question, settings)
        logger.info(
            "model_route_selected choice=%s model=%s source=%s confidence=%s",
            route_info.get("choice"),
            route_info.get("model_name"),
            route_info.get("source"),
            route_info.get("confidence"),
        )
        fanout = plan_speculative_fanout(
            query=question,
            user_role=state.get("user_role", "operator"),
        )
        logger.info(
            "speculative_fanout_planned depth=%s sources=%s",
            fanout.cross_source_depth,
            fanout.recommended_sources,
        )

        episodic_memories = []
        if settings.headroom_memory_enabled and episodic_memory is not None:
            try:
                user_role = state.get("user_role", "operator")
                episodic_memories = episodic_memory.recall_similar_incidents(
                    question, user_id=user_role, top_k=settings.headroom_memory_top_k
                )
                logger.info("recalled_episodic_memories count=%d", len(episodic_memories))
            except Exception as exc:
                logger.warning("episodic_memory_recall_failed err=%s", exc)

        return {
            "question_parts": extract_question_parts(question),
            "model_route": route_info,
            "fanout_plan": asdict(fanout),
            "input_guard": asdict(guard),
            "episodic_memories": episodic_memories,
        }

    return _prepare_node


# ── Node: guard_rejection ───────────────────────────────────────────────────

@traceable_if_enabled(run_type="chain", name="orchestrator.route_after_prepare")
def _route_after_prepare(state: AgentState) -> str:
    guard = state.get("input_guard") or {}
    if guard.get("is_allowed") is False:
        return "guard_rejection"
    return "agent"


def _make_guard_rejection_node():
    @traceable_if_enabled(run_type="chain", name="orchestrator.guard_rejection")
    def _guard_rejection_node(state: AgentState) -> dict[str, Any]:
        guard = state.get("input_guard") or {}
        reason = guard.get("rejection_reason")
        parts = state.get("question_parts", [])

        if reason == "adversarial_blocked":
            direct_ans = (
                "⚠️ [Security Alert]: Request blocked by safety policy. "
                "The query was flagged as adversarial, manipulative, or attempting to bypass security boundaries."
            )
        else:
            direct_ans = (
                "ℹ️ [Out of Scope]: I am the Northgate Assembly Plant Assistant, "
                "specialized in vehicle assembly, technician maintenance, quality engineering, and plant operations. "
                "Please ask questions related to manufacturing specifications, SOPs, or production records."
            )

        final = FinalAnswer(
            direct_answer=direct_ans,
            findings=[],
            evidence=[],
            answered_questions=[],
            unanswered_questions=list(parts),
            recommendation=None,
            class_a_alert=False,
            escalation_level=None,
            recall_criteria_met=[],
        )
        rendered = render_final_answer(final)
        final_dump = final.model_dump()
        final_dump["input_guard"] = guard

        attach_trace_metadata(
            input_blocked=True,
            input_intent_category=guard.get("intent_category"),
            input_malicious_prob=guard.get("is_malicious_prob"),
        )

        return {
            "messages": [AIMessage(content=rendered)],
            "pending_recommendation": None,
            "answered_parts": [],
            "unanswered_parts": list(parts),
            "final_answer": final_dump,
        }

    return _guard_rejection_node


# ── Response policy prompt ─────────────────────────────────────────────────────

def _extract_recent_tool_error(messages: list[Any]) -> str | None:
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content.startswith("Query not allowed:"):
                return content
    return None


def _summarize_tool_results(messages: list[Any], per_result_limit: int = 2000) -> str:
    summaries: list[str] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        name = getattr(msg, "name", None) or "tool"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        formatted_text = ""
        if name == "search_sop_documents":
            try:
                data = json.loads(content)
                docs = data.get("results", []) or data.get("documents", [])
                doc_lines = []
                for d in docs:
                    if isinstance(d, dict):
                        doc_id = d.get("doc_id", "DOC")
                        txt = d.get("text", "") or d.get("content", "") or d.get("summary", "")
                        if txt:
                            doc_lines.append(f"[{doc_id}]: {txt.strip()}")
                if doc_lines:
                    formatted_text = "\n".join(doc_lines)
            except Exception:
                pass
        elif name == "query_nga_database":
            try:
                data = json.loads(content)
                sql_query = data.get("query", "")
                rows = data.get("rows", [])
                row_count = data.get("row_count", len(rows))
                tables = data.get("tables", [])
                lines = [f"[SQL QUERY RESULT] Query: {sql_query}"]
                for i, row in enumerate(rows[:20], 1):
                    if isinstance(row, dict):
                        pairs = ", ".join(f"{k}={v}" for k, v in row.items())
                        lines.append(f"  Row {i}: {pairs}")
                if row_count > 20:
                    lines.append(f"  ... ({row_count - 20} more rows)")
                lines.append(
                    f"({row_count} row(s) returned from tables: {', '.join(tables) if tables else 'N/A'})"
                )
                formatted_text = "\n".join(lines)
            except Exception:
                pass

        if not formatted_text:
            formatted_text = content

        summaries.append(
            f"[{len(summaries)+1}] {name}: "
            f"{_truncate_for_log(formatted_text, limit=per_result_limit)}"
        )
    return "\n".join(summaries)


def _get_current_turn_messages(messages: list[Any]) -> list[Any]:
    """Return only messages from the most recent user turn onward."""
    turn_messages: list[Any] = []
    for m in reversed(messages):
        turn_messages.append(m)
        if isinstance(m, HumanMessage) or getattr(m, "type", "") == "human":
            break
    return list(reversed(turn_messages))


def _count_tool_rounds(messages: list[Any]) -> int:
    turn_msgs = _get_current_turn_messages(messages)
    return sum(1 for m in turn_msgs if getattr(m, "tool_calls", None))


def _build_response_policy(state: AgentState, schema: str = "") -> str:
    question_parts = state.get("question_parts", [])
    user_role = state.get("user_role", "operator")
    role_prompt = SYSTEM_PROMPTS.get(user_role, SYSTEM_PROMPTS["operator"])
    current_msgs = _get_current_turn_messages(state.get("messages", []))
    recent_error = _extract_recent_tool_error(current_msgs)
    tool_rounds = _count_tool_rounds(current_msgs)
    collected = _summarize_tool_results(current_msgs)

    schema_block = f"\nNGA database schema:\n{schema}\n" if schema else ""
    error_block = (
        f"\nPrevious tool error: {recent_error}\n"
        "Correct using exact table/column names from the schema above.\n"
        if recent_error else ""
    )
    collected_block = (
        f"\nData already collected (do NOT re-query these facts):\n{collected}\n"
        if collected else ""
    )
    if tool_rounds >= 3:
        escalation_block = (
            f"\nCRITICAL: {tool_rounds} tool rounds used. "
            "Synthesize the final answer from the data above now.\n"
        )
    elif tool_rounds >= 2:
        escalation_block = (
            "\nYou have collected data above. Only call a tool if a "
            "question part still has zero supporting evidence.\n"
        )
    else:
        escalation_block = ""
    fanout = state.get("fanout_plan")
    fanout_block = (
        f"\n[SPECULATIVE FAN-OUT TARGET DOMAINS]:\n{fanout.get('guidance_prompt')}\n"
        if fanout and fanout.get("guidance_prompt") else ""
    )
    contradiction = state.get("contradiction_resolution")
    conflict_block = (
        f"\n{contradiction.get('resolution_guidance')}\n"
        if contradiction and contradiction.get("has_contradiction") else ""
    )

    episodic_memories = state.get("episodic_memories") or []
    memory_block = ""
    if episodic_memories:
        mem_lines = [
            f"- {m.get('content', '').strip()} (score: {m.get('score', 0):.2f})"
            for m in episodic_memories
            if m.get("content")
        ]
        if mem_lines:
            memory_block = (
                "\n[HISTORICAL RESOLUTION MEMORIES]:\n"
                + "\n".join(mem_lines)
                + "\n"
            )

    return (
        f"{role_prompt}\n\n"
        "Rules:\n"
        "1) GROUNDING: if the question asks about plant documents, SOPs, "
        "procedures, torques/specs/limits, machine details, failure analysis, "
        "recall criteria, or any fact documented in NGA references, call "
        "search_sop_documents BEFORE answering and answer strictly from the "
        "retrieved documents (cite doc IDs). Never answer such questions from "
        "general knowledge without tool evidence. If the question needs build/"
        "defect counts from the database, call query_nga_database instead.\n"
        "2) Check already-collected data before calling a tool.\n"
        "3) NEVER invent SQL rows, numbers, or document citations.\n"
        "4) Use only exact table/column names from the NGA schema.\n"
        "5) For safety decisions, always cite the specific SOP/document ID and "
        "clause.\n"
        "6) When the answer is complete, return a JSON object with: "
        "direct_answer, findings (list), evidence (list of {source_type, citation, supports}), "
        "answered_questions (list), unanswered_questions (list), recommendation (optional), "
        "class_a_alert (bool), escalation_level (string or null), "
        "recall_criteria_met (list of criteria codes like C1, C2 ...).\n"
        "7) VERBATIM EVIDENCE: If retrieved data contains markers like [Compressed ... | hash=abc123] "
        "and you require full uncompressed clauses, exact numerical tolerances, or Class A safety limits, "
        "call retrieve_uncompressed_evidence(content_hash='abc123') to verify original text.\n"
        f"{fanout_block}{conflict_block}{memory_block}{schema_block}{error_block}{collected_block}{escalation_block}"
        f"\nQuestion parts: {question_parts}"
    )


# ── Node: agent ──────────────────────────────────────────────────────────────

def _make_agent_node(
    settings: Settings,
    sql_tool,
    retrieval_tool,
    schema: str = "",
    headroom_mgr=None,
    retrieve_evidence_tool=None,
):
    tools = [sql_tool, retrieval_tool]
    if retrieve_evidence_tool is not None:
        tools.append(retrieve_evidence_tool)

    # Cache chat models per model identifier to avoid repeated client instantiation
    models_by_slug: dict[str, Any] = {}

    def _get_model_for_state(state: AgentState):
        route = state.get("model_route") or {}
        model_slug = route.get("model_name") or settings.openrouter_model
        if model_slug not in models_by_slug:
            models_by_slug[model_slug] = make_chat_model(
                settings, model_override=model_slug
            ).bind_tools(tools)
        return models_by_slug[model_slug]

    @traceable_if_enabled(run_type="chain", name="orchestrator.agent")
    def _agent_node(state: AgentState) -> dict[str, Any]:
        llm = _get_model_for_state(state)
        raw_messages = state["messages"]

        if settings.headroom_compression_enabled and headroom_mgr is not None:
            processed_messages = headroom_mgr.compress_message_history(
                raw_messages, protect_recent=2
            )
        else:
            processed_messages = raw_messages

        messages = [
            SystemMessage(content=_build_response_policy(state, schema)),
            *processed_messages,
        ]
        response = llm.invoke(messages)
        return {"messages": [response]}

    return _agent_node


# ── Loop detection ───────────────────────────────────────────────────────────

def _tool_call_signature(msg: Any) -> tuple | None:
    tc = getattr(msg, "tool_calls", None)
    if not tc:
        return None
    return tuple(
        (str(c.get("name", "")), _truncate_for_log(c.get("args", {}), 1000))
        for c in tc
    )


def _tool_loop_detected(messages: list[Any]) -> bool:
    current_msgs = _get_current_turn_messages(messages)
    if _count_tool_rounds(current_msgs) >= MAX_TOOL_CALL_ROUNDS:
        return True
    repeated = 0
    last_sig = None
    for msg in current_msgs:
        sig = _tool_call_signature(msg)
        if sig is None:
            continue
        if sig == last_sig:
            repeated += 1
        else:
            repeated = 1
            last_sig = sig
        if repeated >= MAX_IDENTICAL_TOOL_CALL_REPEATS:
            return True
    return False


@traceable_if_enabled(run_type="chain", name="orchestrator.route_after_agent")
def _route_after_agent(state: AgentState) -> str:
    messages = state["messages"]
    last = messages[-1]
    if not getattr(last, "tool_calls", None):
        return "synthesis"

    # LAYER 1: Hard Mechanical Circuit Breaker (Fail-Safe Ceiling)
    if _tool_loop_detected(messages):
        logger.warning(
            "circuit_breaker_tripped tool_rounds=%d repeats=%d",
            _count_tool_rounds(messages),
            MAX_IDENTICAL_TOOL_CALL_REPEATS,
        )
        return "synthesis"

    # LAYER 2: Semantic Sufficiency Gate (Jev System One Early Exit)
    if _tool_evidence_ran(messages):
        first_human = next(
            (m.content for m in messages if isinstance(m, HumanMessage) or getattr(m, "type", "") == "human"),
            "",
        )
        parts = state.get("question_parts", [])
        evidence_summary = _summarize_tool_results(messages)

        sufficiency = evaluate_evidence_sufficiency(
            query=first_human,
            question_parts=parts,
            evidence=evidence_summary,
        )

        if sufficiency.is_complete:
            logger.info(
                "semantic_early_exit_triggered round=%d prob=%.2f score=%d action=%s",
                _count_tool_rounds(messages),
                sufficiency.is_sufficient_prob,
                sufficiency.completeness_score,
                sufficiency.next_action,
            )
            return "synthesis"

    return "tools"


# ── Node: synthesis ──────────────────────────────────────────────────────────

def _tool_evidence_ran(messages: list[Any]) -> bool:
    return any(isinstance(m, ToolMessage) for m in messages)


def _is_ungrounded(state: AgentState, answer: FinalAnswer) -> bool:
    if _tool_evidence_ran(state.get("messages", [])):
        return False
    return not answer.evidence


def _is_negative_or_unanswered(answer: FinalAnswer) -> bool:
    """Return True if the answer states no records, defects, or evidence were found."""
    text = (answer.direct_answer or "").strip().lower()
    negative_phrases = (
        "no critical issues",
        "no issues found",
        "no issues occurred",
        "no defects found",
        "no records found",
        "no matching records",
        "0 matching records",
        "0 rows returned",
        "no relevant document",
        "could not gather supporting evidence",
        "could not find",
        "no incidents reported",
        "no incidents occurred",
        "no non-conformance",
        "no escalation",
        "no data found",
    )
    if any(phrase in text for phrase in negative_phrases):
        return True
    if answer.unanswered_questions and not answer.answered_questions and not answer.findings:
        return True
    return False


def _get_searched_categories(messages: list[Any]) -> str:
    seen: list[str] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        if (getattr(msg, "name", None) or "") != "search_sop_documents":
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        try:
            payload = json.loads(content)
            for cat in payload.get("categories", []) or []:
                if cat not in seen:
                    seen.append(cat)
        except (ValueError, TypeError):
            pass
    return ",".join(seen) if seen else "GENERAL"


def _make_synthesis_node(settings: Settings):
    # Cache structured models per model identifier to avoid repeated client instantiation
    structured_models_by_slug: dict[str, Any] = {}

    def _get_structured_model_for_state(state: AgentState):
        route = state.get("model_route") or {}
        model_slug = route.get("model_name") or settings.openrouter_model
        if model_slug not in structured_models_by_slug:
            try:
                base_model = make_chat_model(settings, model_override=model_slug)
                structured_models_by_slug[model_slug] = base_model.with_structured_output(FinalAnswer)
            except Exception as exc:
                logger.info("structured_output_unavailable model=%s reason=%s", model_slug, exc)
                structured_models_by_slug[model_slug] = None
        return structured_models_by_slug[model_slug]

    @traceable_if_enabled(run_type="chain", name="orchestrator.synthesis")
    def _synthesis_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]

        if getattr(last, "tool_calls", None) and _tool_loop_detected(state.get("messages", [])):
            final = FinalAnswer(
                direct_answer=(
                    "I stopped because the agent entered a repeated tool-calling loop. "
                    "Please retry with a more specific question."
                ),
                unanswered_questions=list(state.get("question_parts", [])),
            )
            return {
                "messages": [AIMessage(content=render_final_answer(final))],
                "pending_recommendation": None,
                "answered_parts": [],
                "unanswered_parts": final.unanswered_questions,
                "final_answer": final.model_dump(),
            }

        answer_text = last.content if hasattr(last, "content") else str(last)

        final = None
        structured_model = _get_structured_model_for_state(state)
        if structured_model is not None:
            try:
                prompt = (
                    "Create a structured final answer for a manufacturing decision-support query. "
                    "Use only grounded details from the provided answer. "
                    "For each evidence entry, source_type must strictly be either 'sql' (for database queries) "
                    "or 'document' (for SOPs and reference texts). "
                    "Flag class_a_alert if the text involves Class A safety-critical systems "
                    "(brakes, steering, airbags, seat belts, fuel, wheel retention, "
                    "engine mounts, windshield retention). "
                    "Set escalation_level if an ESC-402 level is mentioned. "
                    "Set recall_criteria_met with QCR-501 criteria codes (C1–C5) if met.\n\n"
                    f"Question parts: {state.get('question_parts', [])}\n"
                    f"Answer text:\n{answer_text}"
                )
                structured = structured_model.invoke([HumanMessage(content=prompt)])
                if isinstance(structured, FinalAnswer):
                    final = structured
                elif isinstance(structured, dict):
                    final = FinalAnswer.model_validate(structured)
            except Exception as exc:
                logger.warning("structured_output_parse_failed: %s; falling back to heuristic parsing", exc)

        if final is None:
            final = parse_final_answer(answer_text, state.get("question_parts", []))

        messages = state.get("messages", [])
        fact_grounding = None
        contradiction_dict = None

        if _is_ungrounded(state, final):
            final = FinalAnswer(
                direct_answer=(
                    "I could not gather supporting evidence from the NGA database or "
                    "reference documents. Please retry; verify that data tools are available."
                ),
                unanswered_questions=list(state.get("question_parts", [])),
            )
        elif _tool_evidence_ran(messages):
            # Check for cross-document / cross-source contradictions
            conflict_res = screen_evidence_contradictions(messages)
            if conflict_res and conflict_res.has_contradiction:
                logger.info(
                    "cross_source_contradiction_detected sources=%s nature=%s rule=%s",
                    conflict_res.conflicting_sources,
                    conflict_res.conflict_nature,
                    conflict_res.precedence_rule,
                )
                contradiction_dict = asdict(conflict_res)
                if not any("precedence" in f.lower() or "conflict" in f.lower() for f in final.findings):
                    final.findings.append(f"Precedence Resolution: {conflict_res.precedence_rule}")
                if not final.recommendation:
                    final.recommendation = conflict_res.precedence_rule

            current_human = next(
                (m.content for m in reversed(messages) if isinstance(m, HumanMessage) or getattr(m, "type", "") == "human"),
                "",
            )
            # Escape fallback/negative answers (no records/issues found) from hallucination quarantine
            if _is_negative_or_unanswered(final):
                logger.info("negative_or_fallback_answer_escaped_from_grounding_check")
                grounding_result = FactGroundednessResult(
                    is_faithful=True,
                    is_faithful_prob=1.0,
                    groundedness_score=5,
                    unsupported_claim_type="none",
                    unsupported_claim_conf=0.0,
                    is_grounded=True,
                    source="negative_result_bypass",
                )
            else:
                evidence_summary = _summarize_tool_results(_get_current_turn_messages(messages))
                grounding_result = evaluate_fact_groundedness(
                    query=current_human,
                    evidence=evidence_summary,
                    generated_answer=final.direct_answer,
                )
            fact_grounding = asdict(grounding_result)

            if not grounding_result.is_grounded:
                logger.warning(
                    "hallucination_detected_by_jev score=%d faith_prob=%.2f claim_type=%s conf=%.2f action=quarantine",
                    grounding_result.groundedness_score,
                    grounding_result.is_faithful_prob,
                    grounding_result.unsupported_claim_type,
                    grounding_result.unsupported_claim_conf,
                )
                quarantine_notice = (
                    f"⚠️ **Grounding Verification Alert**: The generated response contained ungrounded assertions "
                    f"({grounding_result.unsupported_claim_type.replace('_', ' ')}) that could not be verified against "
                    f"the retrieved manufacturing records and specifications.\n\n"
                    f"To maintain manufacturing safety standards, unverified claims have been quarantined. "
                    f"Please review the verified source evidence or refine your query."
                )
                final = FinalAnswer(
                    direct_answer=quarantine_notice,
                    findings=[f"Quarantined ungrounded assertion: {grounding_result.unsupported_claim_type}"],
                    evidence=final.evidence,
                    answered_questions=final.answered_questions,
                    unanswered_questions=list(state.get("question_parts", [])),
                    recommendation="Review raw source documents directly; do not proceed on unverified specifications.",
                )

        # Auto-detect Class A if not already flagged
        if not final.class_a_alert and is_class_a_defect(final.direct_answer):
            final = final.model_copy(update={"class_a_alert": True})

        rendered = render_final_answer(final)
        recommendation = final.recommendation or extract_recommendation(rendered)

        # Calculate Token & Cost Telemetry KPI
        tool_rounds = _count_tool_rounds(messages)
        prompt_tokens = sum(
            len(getattr(m, "content", "")) // 4
            for m in messages
            if not isinstance(m, AIMessage)
        )
        completion_tokens = sum(
            len(getattr(m, "content", "")) // 4
            for m in messages
            if isinstance(m, AIMessage)
        )
        prompt_tokens = max(1200, prompt_tokens)
        completion_tokens = max(250, completion_tokens + len(rendered) // 4)

        early_exit = tool_rounds < MAX_TOOL_CALL_ROUNDS and not _tool_loop_detected(messages)
        jev_calls = (3 if contradiction_dict else (2 if fact_grounding else 1)) if _tool_evidence_ran(messages) else 0
        token_telemetry = calculate_reasoning_token_telemetry(
            llm_prompt_tokens=prompt_tokens,
            llm_completion_tokens=completion_tokens,
            jev_calls_count=jev_calls,
            jev_input_tokens=min(3600, max(300, (len(answer_text) // 4) * max(1, jev_calls))),
            tool_rounds_executed=tool_rounds,
            early_exit_triggered=early_exit,
        )

        attach_token_telemetry(token_telemetry)
        attach_trace_metadata(
            class_a_alert=final.class_a_alert,
            escalation_level=final.escalation_level,
            recall_criteria_met=final.recall_criteria_met,
            early_exit_triggered=early_exit,
            tool_rounds_executed=tool_rounds,
        )

        final_dump = final.model_dump()
        final_dump["token_telemetry"] = token_telemetry
        if fact_grounding:
            final_dump["fact_groundedness"] = fact_grounding
        if contradiction_dict:
            final_dump["contradiction_resolution"] = contradiction_dict

        return {
            "messages": [AIMessage(content=rendered)],
            "pending_recommendation": recommendation,
            "answered_parts": final.answered_questions,
            "unanswered_parts": final.unanswered_questions,
            "final_answer": final_dump,
            "token_usage": token_telemetry,
            "fact_groundedness": fact_grounding,
            "contradiction_resolution": contradiction_dict,
        }

    return _synthesis_node


# ── Node: hitl ───────────────────────────────────────────────────────────────

def _make_hitl_node(app_state_db_path: str, approver=None, episodic_memory=None):
    @traceable_if_enabled(run_type="chain", name="orchestrator.hitl")
    def _hitl_node(state: AgentState) -> dict[str, Any]:
        recommendation = state.get("pending_recommendation")
        if not recommendation:
            return {"pending_recommendation": recommendation}

        question = next(
            (m.content for m in reversed(state["messages"]) if getattr(m, "type", "") == "human"),
            "",
        )

        # Extract Class A and escalation info from final answer
        final_dict = state.get("final_answer") or {}
        class_a_alert = bool(final_dict.get("class_a_alert", False))
        escalation_level = final_dict.get("escalation_level")
        user_role = state.get("user_role", "operator")

        category = _get_searched_categories(_get_current_turn_messages(state.get("messages", [])))

        decision_id = insert_recommendation(
            app_state_db_path,
            question=question,
            recommendation=recommendation,
            category=category,
            user_role=user_role,
            class_a_alert=class_a_alert,
            escalation_level=escalation_level,
            thread_id=state.get("thread_id"),
        )

        if approver is not None:
            # Pluggable approver (server/auto/eval). Interactive input() is
            # unsafe in non-TTY contexts — EOFError crashes the graph.
            status, note = approver(
                recommendation,
                class_a_alert=class_a_alert,
                escalation_level=escalation_level,
            )
        else:
            status, note = request_approval(
                recommendation,
                class_a_alert=class_a_alert,
                escalation_level=escalation_level,
            )
        update_decision(app_state_db_path, decision_id, status=status, approver=note)

        if status == "approved" and episodic_memory is not None:
            try:
                episodic_memory.record_approved_decision(
                    question=question,
                    recommendation=recommendation,
                    category=category,
                    class_a_alert=class_a_alert,
                    approver=note or "Authorized Approver",
                    user_id=user_role,
                )
            except Exception as exc:
                logger.warning("failed_to_record_episodic_memory err=%s", exc)

        return {"pending_recommendation": recommendation}

    return _hitl_node


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_orchestrator(
    settings: Settings,
    checkpointer,
    sql_tool,
    retrieval_tool,
    approver=None,
    headroom_mgr=None,
    episodic_memory=None,
):
    """Build and compile the NGA LangGraph orchestrator.

    `approver`: optional callable(recommendation, *, class_a_alert,
    escalation_level) -> (status, note). Defaults to the interactive CLI
    approver (nga.hitl.approval.request_approval). Server/eval callers MUST
    pass a non-interactive approver (input() crashes outside a TTY).
    """
    wrapped_sql = _wrap_tool_with_logging(sql_tool)
    wrapped_retrieval = _wrap_tool_with_logging(retrieval_tool)
    all_tools = [wrapped_sql, wrapped_retrieval]
    wrapped_retrieve_evidence = None

    if settings.headroom_enabled:
        from nga.compression.manager import HeadroomManager

        if headroom_mgr is None:
            headroom_mgr = HeadroomManager(
                target_ratio=settings.headroom_compression_ratio,
                min_tokens=settings.headroom_min_tokens_to_compress,
                enabled=settings.headroom_compression_enabled,
            )
        from nga.tools.tool_factory import make_headroom_retrieve_tool

        retrieve_evidence_tool = make_headroom_retrieve_tool(headroom_mgr)
        wrapped_retrieve_evidence = _wrap_tool_with_logging(retrieve_evidence_tool)
        all_tools.append(wrapped_retrieve_evidence)

        if settings.headroom_memory_enabled and episodic_memory is None:
            from nga.memory.headroom_memory import NgaEpisodicMemory

            episodic_memory = NgaEpisodicMemory(db_path=settings.headroom_memory_db_path)

    try:
        schema = describe_schema(settings.nga_db_path)
    except Exception as exc:
        logger.info("schema_description_unavailable reason=%s", exc)
        schema = ""

    graph = StateGraph(AgentState)
    graph.add_node("prepare", _make_prepare_node(settings, episodic_memory=episodic_memory))
    graph.add_node("guard_rejection", _make_guard_rejection_node())
    graph.add_node(
        "agent",
        _make_agent_node(
            settings,
            wrapped_sql,
            wrapped_retrieval,
            schema=schema,
            headroom_mgr=headroom_mgr,
            retrieve_evidence_tool=wrapped_retrieve_evidence,
        ),
    )
    graph.add_node("tools", ToolNode(all_tools))
    graph.add_node("synthesis", _make_synthesis_node(settings))
    graph.add_node(
        "hitl",
        _make_hitl_node(
            settings.app_state_db_path,
            approver=approver,
            episodic_memory=episodic_memory,
        ),
    )

    graph.set_entry_point("prepare")
    graph.add_conditional_edges(
        "prepare",
        _route_after_prepare,
        {"agent": "agent", "guard_rejection": "guard_rejection"},
    )
    graph.add_edge("guard_rejection", "hitl")
    graph.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tools": "tools", "synthesis": "synthesis"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("synthesis", "hitl")
    graph.add_edge("hitl", END)

    return graph.compile(checkpointer=checkpointer)
