"""CLI launcher for the Enterprise Agent Web Interface."""

from __future__ import annotations

import argparse
import logging
import os
import webbrowser

import uvicorn

logger = logging.getLogger("enterprise_agent.ui.runner")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enterprise Agent Web Interface Server."
    )
    parser.add_argument(
        "--domain",
        default="domains/manufacturing/pack.yaml",
        help="Path to domain pack YAML (default: domains/manufacturing/pack.yaml)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000)",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        default=False,
        help="Automatically open web browser upon launch",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload on code changes",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.domain:
        os.environ["DOMAIN_PACK"] = args.domain

    print("=" * 66)
    print("  🚀 Enterprise Agent — Decision Support Web User Interface")
    print("=" * 66)
    print(f"  URL        : http://{args.host}:{args.port}")
    print(f"  Domain Pack: {args.domain}")
    print("  Clearance  : Multi-Role RBAC Clearance Gate")
    print("  Safety     : Critical Decision Gate & HITL Workflow")
    print("  Evaluation : Live Benchmark Runner & DeepEval Evaluation Studio")
    print("=" * 66)
    print()

    if args.open_browser:
        webbrowser.open(f"http://{args.host}:{args.port}")

    uvicorn.run(
        "enterprise_agent.ui.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
