"""LangGraph state schema for the NGA Manufacturing Assistant orchestrator."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    question_parts: list[str]
    answered_parts: list[str]
    unanswered_parts: list[str]
    sql_results: list[dict[str, Any]]
    retrieved_docs: list[dict[str, Any]]
    final_answer: dict[str, Any] | None
    pending_recommendation: str | None
    # NGA-specific: role of the requesting user
    user_role: str
    user_level: int
    model_route: dict[str, Any] | None
    evidence_sufficiency: dict[str, Any] | None
    token_usage: dict[str, Any] | None
