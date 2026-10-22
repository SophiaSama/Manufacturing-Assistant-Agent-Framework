"""CLI launcher for the NGA Manufacturing Assistant Web Interface."""

from __future__ import annotations

import argparse
import logging
import webbrowser

import uvicorn

logger = logging.getLogger("nga.ui.runner")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NGA Manufacturing Assistant Web Interface Server."
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

    print("=" * 66)
    print("  🚀 NGA Manufacturing Assistant — Web User Interface")
    print("=" * 66)
    print(f"  URL        : http://{args.host}:{args.port}")
    print("  Clearance  : Multi-Role RBAC (Operator, Technician, Engineer, Manager)")
    print("  Safety     : Class A Defect Detection & ESC-402 / QCR-501 HITL")
    print("  Evaluation : Live Benchmark Runner & Model Comparison Studio")
    print("=" * 66)
    print()

    if args.open_browser:
        webbrowser.open(f"http://{args.host}:{args.port}")

    uvicorn.run(
        "nga.ui.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
