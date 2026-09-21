"""State definition for the Enterprise Agentic LangGraph workflow."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages

from enterprise_agent.models.answer_schema import FinalAnswer


class AgentState(TypedDict, total=False):
    """Complete state passed between nodes in the agent graph."""
    messages: Annotated[list[Any], add_messages]
    question_parts: list[str]
    user_role: str
    final_answer: FinalAnswer | None
    approval_status: str | None
    approval_note: str | None
