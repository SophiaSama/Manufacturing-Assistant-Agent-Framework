"""Interactive terminal chat loop for the Enterprise Agentic Framework."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage

from enterprise_agent.config.domain_config import DomainConfig
from enterprise_agent.database.engine import SqliteAdapter
from enterprise_agent.graph.orchestrator import build_enterprise_agent_graph
from enterprise_agent.models.answer_schema import render_final_answer
from enterprise_agent.providers.factory import make_chat_model
from enterprise_agent.rag.retrieval import (
    build_or_open_vector_store,
    make_generic_retrieval_tool,
)
from enterprise_agent.tools.tool_factory import make_generic_sql_tool

logger = logging.getLogger("enterprise_agent.cli")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enterprise Agentic Framework — Multi-Industry Decision Support."
    )
    parser.add_argument(
        "--domain",
        default="domains/manufacturing/pack.yaml",
        help="Path to domain pack YAML (default: domains/manufacturing/pack.yaml)",
    )
    parser.add_argument(
        "--role",
        default=None,
        help="User role (defaults to domain default role)",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Logging level (default: WARNING)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING))

    pack_path = Path(args.domain).resolve()
    if not pack_path.exists():
        print(f"Domain pack not found: {pack_path}")
        return 1

    domain_config = DomainConfig.from_yaml_file(pack_path)
    role = args.role or domain_config.rbac.default_role
    print(f"Loaded Domain: {domain_config.domain.name}")
    print(f"Active Role: {role} (Rank: {domain_config.rbac.get_rank(role)})")

    # Connect DB
    db_path = Path(domain_config.database.connection_uri).resolve()
    db_adapter = SqliteAdapter(db_path, allowed_tables=domain_config.database.allowed_tables)
    sql_tool = make_generic_sql_tool(
        db_adapter,
        tool_name=domain_config.database.tool_name,
        tool_description=domain_config.database.tool_description,
    )

    # Ingest documents and setup retrieval tool
    data_dir_path = None
    if domain_config.domain.data_dir:
        cand1 = Path(domain_config.domain.data_dir)
        cand2 = pack_path.parent / domain_config.domain.data_dir
        if cand1.exists():
            data_dir_path = cand1.resolve()
        elif cand2.exists():
            data_dir_path = cand2.resolve()
    elif (pack_path.parent / "data").exists():
        data_dir_path = (pack_path.parent / "data").resolve()

    settings = None
    try:
        from nga.config import Settings
        settings = Settings.from_env()
    except Exception:
        settings = None

    embeddings = None
    try:
        from enterprise_agent.providers.factory import make_embeddings
        embeddings = make_embeddings(settings)
    except Exception:
        embeddings = None

    store = build_or_open_vector_store(
        domain_config=domain_config,
        documents_dir=data_dir_path,
        embeddings=embeddings,
    )
    user_rank = domain_config.rbac.get_rank(role)
    retrieval_tool = make_generic_retrieval_tool(
        store=store,
        user_level=user_rank,
        doc_config=domain_config.documents,
    )

    tools = [sql_tool, retrieval_tool]
    schema_summary = db_adapter.describe_schema()

    # Chat model
    try:
        chat_model = make_chat_model(settings)
    except TypeError:
        chat_model = make_chat_model()

    graph = build_enterprise_agent_graph(
        domain_config=domain_config,
        chat_model=chat_model,
        tools=tools,
        schema_summary=schema_summary,
    )

    print("\nAssistant ready. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            prompt = input(f"[{role}] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not prompt or prompt.lower() in ("exit", "quit"):
            break

        result = graph.invoke({
            "messages": [HumanMessage(content=prompt)],
            "user_role": role,
        })
        final_answer = result.get("final_answer")
        if final_answer:
            print("\n" + render_final_answer(final_answer) + "\n")
        else:
            print("\nNo answer produced.\n")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
