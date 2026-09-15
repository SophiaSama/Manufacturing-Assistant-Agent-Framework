"""LangGraph orchestrator for the NGA Manufacturing Assistant.

Workflow: prepare → agent ⇆ tools → synthesis → hitl → END

Roles feed role-specific system prompts and RBAC-filtered retrieval.
The synthesis node detects Class A defects and recall criteria automatically.
"""

from __future__ import annotations

import json
import logging
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
from nga.rag_agent.rbac import SYSTEM_PROMPTS
from nga.tools.sql_tool import describe_schema

logger = logging.getLogger("nga.orchestrator")

MAX_TOOL_CALL_ROUNDS = 8
MAX_IDENTICAL_TOOL_CALL_REPEATS = 3


def _truncate_for_log(value: Any, limit: int = 400) -> str:
    text = str(value)
    return text if len(text) <= limit else f"{text[:limit]}... [+{len(text)-limit} chars]"


def _wrap_tool_with_logging(tool_obj: BaseTool) -> BaseTool:
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

def _prepare_node(state: AgentState) -> dict[str, Any]:
    last = state["messages"][-1]
    question = last.content if hasattr(last, "content") else str(last)
    return {"question_parts": extract_question_parts(question)}


# ── Response policy prompt ─────────────────────────────────────────────────────

def _extract_recent_tool_error(messages: list[Any]) -> str | None:
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content.startswith("Query not allowed:"):
                return content
    return None


def _summarize_tool_results(messages: list[Any], per_result_limit: int = 600) -> str:
    summaries: list[str] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        name = getattr(msg, "name", None) or "tool"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        summaries.append(
            f"[{len(summaries)+1}] {name}: "
            f"{_truncate_for_log(content, limit=per_result_limit)}"
        )
    return "\n".join(summaries)


def _count_tool_rounds(messages: list[Any]) -> int:
    return sum(1 for m in messages if getattr(m, "tool_calls", None))


def _build_response_policy(state: AgentState, schema: str = "") -> str:
    question_parts = state.get("question_parts", [])
    user_role = state.get("user_role", "operator")
    role_prompt = SYSTEM_PROMPTS.get(user_role, SYSTEM_PROMPTS["operator"])
    recent_error = _extract_recent_tool_error(state.get("messages", []))
    tool_rounds = _count_tool_rounds(state.get("messages", []))
    collected = _summarize_tool_results(state.get("messages", []))

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
        f"{schema_block}{error_block}{collected_block}{escalation_block}"
        f"\nQuestion parts: {question_parts}"
    )


# ── Node: agent ──────────────────────────────────────────────────────────────

def _make_agent_node(settings: Settings, sql_tool, retrieval_tool, schema: str = ""):
    llm = make_chat_model(settings).bind_tools([sql_tool, retrieval_tool])

    def _agent_node(state: AgentState) -> dict[str, Any]:
        messages = [
            SystemMessage(content=_build_response_policy(state, schema)),
            *state["messages"],
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
    if _count_tool_rounds(messages) >= MAX_TOOL_CALL_ROUNDS:
        return True
    repeated = 0
    last_sig = None
    for msg in messages:
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


def _route_after_agent(state: AgentState) -> str:
    messages = state["messages"]
    last = messages[-1]
    if not getattr(last, "tool_calls", None):
        return "synthesis"
    if _tool_loop_detected(messages):
        return "synthesis"
    return "tools"


# ── Node: synthesis ──────────────────────────────────────────────────────────

def _tool_evidence_ran(messages: list[Any]) -> bool:
    return any(isinstance(m, ToolMessage) for m in messages)


def _is_ungrounded(state: AgentState, answer: FinalAnswer) -> bool:
    if _tool_evidence_ran(state.get("messages", [])):
        return False
    return not answer.evidence


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
    structured_model = None
    try:
        base_model = make_chat_model(settings)
        structured_model = base_model.with_structured_output(FinalAnswer)
    except Exception as exc:
        logger.info("structured_output_unavailable reason=%s", exc)

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
        if structured_model is not None:
            try:
                prompt = (
                    "Create a structured final answer for a manufacturing decision-support query. "
                    "Use only grounded details from the provided answer. "
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
            except Exception:
                logger.exception("structured_output_parse_failed")

        if final is None:
            final = parse_final_answer(answer_text, state.get("question_parts", []))

        if _is_ungrounded(state, final):
            final = FinalAnswer(
                direct_answer=(
                    "I could not gather supporting evidence from the NGA database or "
                    "reference documents. Please retry; verify that data tools are available."
                ),
                unanswered_questions=list(state.get("question_parts", [])),
            )

        # Auto-detect Class A if not already flagged
        if not final.class_a_alert and is_class_a_defect(final.direct_answer):
            final = final.model_copy(update={"class_a_alert": True})

        rendered = render_final_answer(final)
        recommendation = final.recommendation or extract_recommendation(rendered)

        return {
            "messages": [AIMessage(content=rendered)],
            "pending_recommendation": recommendation,
            "answered_parts": final.answered_questions,
            "unanswered_parts": final.unanswered_questions,
            "final_answer": final.model_dump(),
        }

    return _synthesis_node


# ── Node: hitl ───────────────────────────────────────────────────────────────

def _make_hitl_node(app_state_db_path: str, approver=None):
    def _hitl_node(state: AgentState) -> dict[str, Any]:
        recommendation = state.get("pending_recommendation")
        if not recommendation:
            return {"pending_recommendation": recommendation}

        question = next(
            (m.content for m in state["messages"] if getattr(m, "type", "") == "human"),
            "",
        )

        # Extract Class A and escalation info from final answer
        final_dict = state.get("final_answer") or {}
        class_a_alert = bool(final_dict.get("class_a_alert", False))
        escalation_level = final_dict.get("escalation_level")
        user_role = state.get("user_role", "operator")

        decision_id = insert_recommendation(
            app_state_db_path,
            question=question,
            recommendation=recommendation,
            category=_get_searched_categories(state.get("messages", [])),
            user_role=user_role,
            class_a_alert=class_a_alert,
            escalation_level=escalation_level,
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
        return {"pending_recommendation": recommendation}

    return _hitl_node


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_orchestrator(settings: Settings, checkpointer, sql_tool, retrieval_tool,
                       approver=None):
    """Build and compile the NGA LangGraph orchestrator.

    `approver`: optional callable(recommendation, *, class_a_alert,
    escalation_level) -> (status, note). Defaults to the interactive CLI
    approver (nga.hitl.approval.request_approval). Server/eval callers MUST
    pass a non-interactive approver (input() crashes outside a TTY).
    """
    wrapped_sql = _wrap_tool_with_logging(sql_tool)
    wrapped_retrieval = _wrap_tool_with_logging(retrieval_tool)

    try:
        schema = describe_schema(settings.nga_db_path)
    except Exception as exc:
        logger.info("schema_description_unavailable reason=%s", exc)
        schema = ""

    graph = StateGraph(AgentState)
    graph.add_node("prepare", _prepare_node)
    graph.add_node(
        "agent",
        _make_agent_node(settings, wrapped_sql, wrapped_retrieval, schema),
    )
    graph.add_node("tools", ToolNode([wrapped_sql, wrapped_retrieval]))
    graph.add_node("synthesis", _make_synthesis_node(settings))
    graph.add_node("hitl", _make_hitl_node(settings.app_state_db_path, approver=approver))

    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "agent")
    graph.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tools": "tools", "synthesis": "synthesis"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("synthesis", "hitl")
    graph.add_edge("hitl", END)

    return graph.compile(checkpointer=checkpointer)
