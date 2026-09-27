#!/usr/bin/env python3
"""Run TypeSafe AI entity alignment on the saved knowledge graph.

Loads data/graph_store/graph.json, detects duplicate candidates,
evaluates pairs using TypeSafe Score and Nouls, merges duplicates,
and updates the persisted graph.
"""

from __future__ import annotations

import argparse
import logging
import sys

from nga.config import Settings
from nga.ingestion.build_graph import load_graph, save_graph
from nga.ingestion.entity_alignment import (
    align_entities,
    get_typesafe_api_key,
    get_typesafe_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("align_graph")


def main() -> None:
    parser = argparse.ArgumentParser(description="TypeSafe Knowledge Graph Entity Alignment")
    parser.add_argument(
        "--model",
        default=None,
        help="TypeSafe model override (default: from TYPESAFE_MODEL or 'jev-1.12')",
    )
    parser.add_argument(
        "--store-dir",
        default=None,
        help="Path to graph store dir (default: from Settings.graph_store_dir)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Score and route pairs without saving changes to disk",
    )
    args = parser.parse_args()

    # Verify credentials
    key = get_typesafe_api_key()
    if not key:
        logger.error(
            "ERROR: Neither TYPESAFE_API_KEY nor TYPESAFE_API_TEST is set. "
            "Cannot run entity alignment."
        )
        sys.exit(1)

    settings = Settings.from_env()
    store_dir = args.store_dir or settings.graph_store_dir
    logger.info("Loading graph from %s...", store_dir)
    graph = load_graph(store_dir)

    if graph is None:
        logger.error("No graph found at %s. Please run build_graph first.", store_dir)
        sys.exit(1)

    initial_nodes = graph.number_of_nodes()
    initial_edges = graph.number_of_edges()
    logger.info("Loaded graph with %d nodes and %d edges.", initial_nodes, initial_edges)

    model = args.model or get_typesafe_model()
    logger.info("Using TypeSafe model: %s", model)

    report = align_entities(
        graph,
        model=model,
        require_api_key=True,
    )

    final_nodes = graph.number_of_nodes()
    final_edges = graph.number_of_edges()

    print("\n" + "=" * 50)
    print("      TypeSafe Entity Alignment Summary")
    print("=" * 50)
    print(f"Model used:          {model}")
    print(f"Candidates analyzed: {report['candidates_found']}")
    print(f"Entities merged:     {report['merged_count']}")
    print(f"Sent to review queue:{report['review_count']}")
    print(f"Kept separate:       {report['separate_count']}")
    print(f"Nodes before/after:  {initial_nodes} -> {final_nodes} (-{initial_nodes - final_nodes})")
    print(f"Edges before/after:  {initial_edges} -> {final_edges}")
    print("=" * 50 + "\n")

    if not args.dry_run:
        save_graph(graph, store_dir)
        logger.info("Saved aligned graph back to %s/graph.json.", store_dir)
    else:
        logger.info("Dry-run mode: changes were not saved to disk.")


if __name__ == "__main__":
    main()
