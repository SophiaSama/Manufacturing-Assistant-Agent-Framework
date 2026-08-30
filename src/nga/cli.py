"""Interactive terminal chat loop for the NGA Manufacturing Assistant."""

from __future__ import annotations

import argparse
import logging
import uuid

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage

from nga.cache import CachedEmbeddings, make_cache_from_settings
from nga.config import Settings
from nga.graph.orchestrator import build_orchestrator
from nga.ingestion.build_graph import load_graph
from nga.memory.checkpointer import build_checkpointer
from nga.memory.decision_log import init_decision_log
from nga.models.answer_schema import (
    FinalAnswer,
    parse_final_answer,
    render_final_answer,
)
from nga.providers.factory import make_embeddings
from nga.rag_agent.rbac import ACCESS_LEVELS
from nga.tools.tool_factory import make_retrieval_tool, make_sql_tool

logger = logging.getLogger("nga.cli")

ROLE_CHOICES = list(ACCESS_LEVELS.keys())  # operator, technician, engineer, manager


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NGA Manufacturing Assistant — interactive chat."
    )
    parser.add_argument(
        "--role",
        choices=ROLE_CHOICES,
        default="operator",
        help="User role: operator, technician, engineer, or manager (default: operator)",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Logging level (default: WARNING)",
    )
    return parser.parse_args(argv)


def _render_synthesis_answer(synthesis_update: dict, question_parts: list[str]) -> str | None:
    payload = synthesis_update.get("final_answer")
    if isinstance(payload, dict):
        try:
            return render_final_answer(FinalAnswer.model_validate(payload))
        except Exception:
            pass

    messages = synthesis_update.get("messages")
    if not messages:
        return None
    last = messages[-1]
    raw = last.content if hasattr(last, "content") else str(last)
    if isinstance(raw, str):
        return render_final_answer(parse_final_answer(raw, question_parts))
    return str(raw)


def run_turn(graph, question: str, *, thread_id: str, role: str, print_fn=print) -> str | None:
    answer = None
    printed = False
    question_parts: list[str] = []

    initial_state = {
        "messages": [HumanMessage(content=question)],
        "user_role": role,
        "user_level": ACCESS_LEVELS.get(role, 1),
    }

    for update in graph.stream(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="updates",
    ):
        prepare_update = update.get("prepare") if isinstance(update, dict) else None
        if prepare_update and isinstance(prepare_update.get("question_parts"), list):
            question_parts = [
                p for p in prepare_update["question_parts"]
                if isinstance(p, str) and p.strip()
            ]

        synthesis_update = update.get("synthesis") if isinstance(update, dict) else None
        if answer is None and synthesis_update:
            answer = _render_synthesis_answer(synthesis_update, question_parts)

        if answer is not None and not printed:
            print_fn(f"\nAssistant [{role.upper()}]: {answer}\n")
            printed = True

    return answer


def _print_welcome(role: str) -> None:
    role_desc = {
        "operator":   "Assembly procedures, torque specs, and SOPs",
        "technician": "Troubleshooting, calibration, and maintenance",
        "engineer":   "Root-cause analysis, defect investigation, and supplier quality",
        "manager":    "Escalation, recall criteria, stop-ship, and regulatory compliance",
    }
    print("=" * 60)
    print("  NGA Manufacturing Assistant — Northgate Assembly Plant")
    print("=" * 60)
    print(f"  Role     : {role.upper()}")
    print(f"  Access   : {role_desc.get(role, '')}")
    print("  Commands : 'exit' to quit | 'role <name>' to switch role")
    print("=" * 60)
    print()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING))

    settings = Settings.from_env()
    init_decision_log(settings.app_state_db_path)

    cache = make_cache_from_settings(settings, env="prod")
    if cache is not None:
        logger.info("cache enabled env=prod db=%s", settings.cache_db_path)

    embeddings = make_embeddings(settings)
    if cache is not None:
        embeddings = CachedEmbeddings(embeddings, cache=cache, model_id=settings.embedding_model)
    store = Chroma(
        collection_name="nga_reference_docs",
        embedding_function=embeddings,
        persist_directory=settings.vector_store_dir,
    )

    graph_data = load_graph(settings.graph_store_dir)
    role = args.role
    user_level = ACCESS_LEVELS.get(role, 1)

    sql_tool = make_sql_tool(settings.nga_db_path, cache=cache)
    retrieval_tool = make_retrieval_tool(
        store, graph=graph_data, embeddings=embeddings, user_level=user_level,
        cache=cache,
    )
    checkpointer = build_checkpointer(settings.app_state_db_path)
    agent_graph = build_orchestrator(settings, checkpointer, sql_tool, retrieval_tool)

    thread_id = str(uuid.uuid4())
    _print_welcome(role)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        # Allow role switching mid-session
        if user_input.lower().startswith("role "):
            new_role = user_input[5:].strip().lower()
            if new_role in ROLE_CHOICES:
                role = new_role
                user_level = ACCESS_LEVELS[role]
                retrieval_tool = make_retrieval_tool(
                    store, graph=graph_data, embeddings=embeddings,
                    user_level=user_level, cache=cache,
                )
                agent_graph = build_orchestrator(
                    settings, checkpointer, sql_tool, retrieval_tool
                )
                thread_id = str(uuid.uuid4())
                print(f"\n[Switched to role: {role.upper()}]\n")
            else:
                print(f"Unknown role '{new_role}'. Choose from: {', '.join(ROLE_CHOICES)}")
            continue

        run_turn(agent_graph, user_input, thread_id=thread_id, role=role)


if __name__ == "__main__":
    main()
