"""Generic LangGraph orchestrator for enterprise decision support.

Workflow: prepare → agent ⇆ tools → synthesis → hitl → END
Configured dynamically by DomainConfig.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from enterprise_agent.config.domain_config import DomainConfig
from enterprise_agent.graph.nodes import extract_question_parts
from enterprise_agent.graph.state import AgentState
from enterprise_agent.hitl.policy_gate import (
    evaluate_governance,
    extract_recommendation,
    request_approval,
)
from enterprise_agent.models.answer_schema import (
    FinalAnswer,
    parse_final_answer,
)

logger = logging.getLogger("enterprise_agent.orchestrator")

MAX_TOOL_CALL_ROUNDS = 8


def _build_response_policy(
    state: AgentState,
    domain_config: DomainConfig,
    schema_summary: str = "",
) -> str:
    user_role = state.get("user_role", domain_config.rbac.default_role)
    role_prompt = domain_config.rbac.role_prompts.get(
        user_role,
        f"You are an AI decision support assistant for {domain_config.domain.name}."
    )
    question_parts = state.get("question_parts", [])

    parts_block = ""
    if len(question_parts) > 1:
        parts_block = "\nAnswer each sub-question specifically:\n" + "\n".join(
            f"  {i+1}. {q}" for i, q in enumerate(question_parts)
        )

    schema_block = f"\nDatabase Schema:\n{schema_summary}\n" if schema_summary else ""

    instructions = (
        f"{role_prompt}\n"
        f"{schema_block}"
        f"{parts_block}\n\n"
        "Guidelines:\n"
        "- Use the provided tools to query databases and reference documents.\n"
        "- Ground all claims in tool outputs with exact citations.\n"
        "- Return your final output as valid JSON matching the FinalAnswer schema:\n"
        "{\n"
        '  "direct_answer": "...",\n'
        '  "findings": ["..."],\n'
        '  "evidence": [{"source_type": "sql|document", "citation": "...", "supports": ["..."]}],\n'
        '  "recommendation": "..."\n'
        "}\n"
    )
    return instructions


def build_enterprise_agent_graph(
    domain_config: DomainConfig,
    chat_model: Any,
    tools: list[BaseTool],
    schema_summary: str = "",
    enable_hitl: bool = True,
):
    """Compile a LangGraph workflow customized for a specific domain."""

    def prepare_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        question = last.content if hasattr(last, "content") else str(last)
        return {"question_parts": extract_question_parts(question)}

    model_with_tools = chat_model.bind_tools(tools)

    def agent_node(state: AgentState) -> dict[str, Any]:
        policy = _build_response_policy(state, domain_config, schema_summary)
        messages = [SystemMessage(content=policy)] + list(state["messages"])
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None)
        if tool_calls:
            # Count rounds
            rounds = sum(1 for m in state["messages"] if getattr(m, "tool_calls", None))
            if rounds < MAX_TOOL_CALL_ROUNDS:
                return "tools"
        return "synthesis"

    def synthesis_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        content = last.content if hasattr(last, "content") else str(last)
        final_answer = parse_final_answer(content)

        # Check governance rules
        rec = final_answer.recommendation or extract_recommendation(
            final_answer.direct_answer,
            domain_config.governance.hitl_action_verbs,
        )
        if rec:
            final_answer.recommendation = rec

        gov = evaluate_governance(
            final_answer.direct_answer,
            rec,
            domain_config.governance,
        )
        final_answer.governance = gov

        return {"final_answer": final_answer}

    def hitl_node(state: AgentState) -> dict[str, Any]:
        final_answer = state.get("final_answer")
        if not final_answer or not enable_hitl:
            return {}

        gov = final_answer.governance
        rec = final_answer.recommendation
        if rec and (gov.hitl_required or gov.is_critical):
            status, note = request_approval(
                rec,
                governance=gov,
                config=domain_config.governance,
            )
            return {"approval_status": status, "approval_note": note}

        return {"approval_status": "none", "approval_note": None}

    # Build Graph
    graph = StateGraph(AgentState)
    graph.add_node("prepare", prepare_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("hitl", hitl_node)

    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "synthesis": "synthesis"})
    graph.add_edge("tools", "agent")
    graph.add_edge("synthesis", "hitl")
    graph.add_edge("hitl", END)

    return graph.compile()
