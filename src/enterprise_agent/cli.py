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
from enterprise_agent.tools.tool_factory import make_generic_sql_tool
from nga.providers.factory import make_chat_model

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

    tools = [sql_tool]
    schema_summary = db_adapter.describe_schema()

    # Chat model
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
