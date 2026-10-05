"""GraphRAG knowledge graph builder for the NGA corpus.

Builds a NetworkX directed multigraph of entities and relations from the
ChromaDB documents using LLM extraction, with ontology validation, confidence
filtering and per-item provenance. Louvain community detection supports
multi-hop reasoning.

Run: uv run python -m nga.ingestion.build_graph
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from nga.ingestion.normalize import (
    NGA_ENTITY_TYPES,
    NGA_RELATION_TYPES,
    canonical_id,
    check_relation,
    normalize_entity_type,
    normalize_relation_type,
    ontology_prompt_text,
)

logger = logging.getLogger(__name__)

# Items below this confidence are routed to the review queue, not the graph.
MIN_CONFIDENCE = 0.5
# Max characters sent per extraction call; longer chunks are split in windows.
WINDOW_CHARS = 3000
WINDOW_OVERLAP = 300
# Abort the build after this many back-to-back LLM API errors (quota, auth, outage).
MAX_CONSECUTIVE_API_ERRORS = 5
# Refuse to return a graph if more than this share of windows failed.
MAX_FAILURE_RATE = 0.5


class ExtractionAborted(RuntimeError):
    """Raised when extraction is too unhealthy to produce a trustworthy graph."""

EXTRACTION_PROMPT = """Extract manufacturing knowledge entities and relations from the text below.

Entity types: {entity_types}
Relation types: {relation_types}

Allowed direction and endpoint types (source types --relation--> target types):
{ontology}

Return a JSON object with:
{{
  "entities": [
    {{"id": "unique_slug", "type": "EntityType", "name": "Display Name",
      "aliases": ["other names or codes used in the text"],
      "description": "brief description", "level_rank": 1,
      "confidence": 0.0-1.0, "evidence": "short verbatim quote"}}
  ],
  "relations": [
    {{"source": "entity_id_1", "relation": "relation_type",
      "target": "entity_id_2", "description": "context", "level_rank": 1,
      "confidence": 0.0-1.0, "evidence": "short verbatim quote"}}
  ]
}}

Rules:
- Keep entity IDs as lowercase slug (e.g. "tq-6012", "sop-opr-101")
- Use only the entity and relation types listed above
- Relations are directed: source --relation--> target; every source/target must be an entity id you return
- Use level_rank matching the source document's access level
- Only extract entities explicitly mentioned in the text; never guess
- confidence reflects how explicitly the text supports the item
- Return valid JSON only, no markdown

Example:
Text: "SOP-OPR-101 requires nutrunner TQ-6012 (TorqMaster 6012) at Station 144; target torque is 105 Nm."
Output: {{"entities": [
  {{"id": "sop-opr-101", "type": "SOP", "name": "SOP-OPR-101", "aliases": [], "description": "Lug nut tightening SOP", "level_rank": 1, "confidence": 0.95, "evidence": "SOP-OPR-101"}},
  {{"id": "tq-6012", "type": "Machine", "name": "TorqMaster TQ-6012", "aliases": ["TorqMaster 6012"], "description": "Nutrunner", "level_rank": 1, "confidence": 0.95, "evidence": "nutrunner TQ-6012 (TorqMaster 6012)"}}],
 "relations": [
  {{"source": "sop-opr-101", "relation": "requires", "target": "tq-6012", "description": "SOP requires this tool", "level_rank": 1, "confidence": 0.9, "evidence": "SOP-OPR-101 requires nutrunner TQ-6012"}}]}}

Text:
{text}
"""

REPAIR_PROMPT = """The previous reply was not valid JSON ({error}).
Return ONLY the corrected JSON object, with no commentary and no markdown.

Previous reply:
{reply}
"""


def _parse_json(content: str) -> dict[str, Any]:
    """Parse an LLM reply into a dict, tolerating fences and surrounding prose."""
    content = content.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
    if fence:
        content = fence.group(1).strip()
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end <= start:
            raise
        payload = json.loads(content[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("extraction payload is not a JSON object")
    return payload


def _split_windows(text: str) -> list[str]:
    """Split long text into overlapping windows instead of truncating it."""
    if len(text) <= WINDOW_CHARS:
        return [text]
    windows, start = [], 0
    while start < len(text):
        windows.append(text[start:start + WINDOW_CHARS])
        if start + WINDOW_CHARS >= len(text):
            break
        start += WINDOW_CHARS - WINDOW_OVERLAP
    return windows


def _extract_window(text: str, llm: Any, stats: Counter[str]) -> dict[str, Any]:
    """Extract from one window; retry once with a JSON-repair prompt on failure."""
    from langchain_core.messages import HumanMessage

    prompt = EXTRACTION_PROMPT.format(
        entity_types=", ".join(NGA_ENTITY_TYPES),
        relation_types=", ".join(NGA_RELATION_TYPES),
        ontology=ontology_prompt_text(),
        text=text,
    )
    content = ""
    stats["windows"] += 1
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
    except Exception as exc:
        # Auth / quota / network error: a repair prompt cannot help, so don't retry.
        stats["api_errors"] += 1
        stats["failed_windows"] += 1
        stats["consecutive_api_errors"] += 1
        logger.warning("LLM call failed: %s", exc)
        if stats["consecutive_api_errors"] >= MAX_CONSECUTIVE_API_ERRORS:
            raise ExtractionAborted(
                f"{stats['consecutive_api_errors']} consecutive LLM API errors (last: {exc}); "
                "aborting so an empty graph is not saved."
            ) from exc
        return {"entities": [], "relations": []}
    stats["consecutive_api_errors"] = 0
    try:
        content = response.content if hasattr(response, "content") else str(response)
        return _parse_json(content)
    except Exception as exc:
        stats["parse_retries"] += 1
        logger.warning("Extraction parse failed (%s); retrying with repair prompt", exc)
        try:
            repair = llm.invoke(
                [HumanMessage(content=REPAIR_PROMPT.format(error=exc, reply=content[:4000]))]
            )
            fixed = repair.content if hasattr(repair, "content") else str(repair)
            return _parse_json(fixed)
        except Exception as exc2:
            stats["failed_windows"] += 1
            logger.warning("Extraction failed after retry: %s", exc2)
            return {"entities": [], "relations": []}


def extract_entities_and_relations(
    text: str,
    llm: Any,
    level_rank: int = 1,
    stats: Counter[str] | None = None,
) -> dict[str, Any]:
    """Run LLM entity/relation extraction over a text chunk (windowed, with retry)."""
    stats = stats if stats is not None else Counter()
    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    for window in _split_windows(text):
        payload = _extract_window(window, llm, stats)
        entities.extend(e for e in payload.get("entities", []) if isinstance(e, dict))
        relations.extend(r for r in payload.get("relations", []) if isinstance(r, dict))
    for e in entities:
        e.setdefault("level_rank", level_rank)
    for r in relations:
        r.setdefault("level_rank", level_rank)
    return {"entities": entities, "relations": relations}


def _confidence(item: dict[str, Any]) -> float:
    try:
        return float(item.get("confidence", 1.0))
    except (TypeError, ValueError):
        return 1.0


def _add_source(target: dict[str, Any], source: dict[str, Any]) -> None:
    sources = target.setdefault("sources", [])
    if source not in sources:
        sources.append(source)


def build_graph_from_documents(
    documents: list[Any],
    llm: Any,
    max_docs: int | None = None,
    min_confidence: float = MIN_CONFIDENCE,
) -> Any:
    """Build a NetworkX knowledge graph (``MultiDiGraph``) from chunked documents.

    Entities and relations outside the ontology, violating its type rules, or
    below ``min_confidence`` are kept out of the graph and recorded in
    ``graph.graph["review_queue"]``; counters are in ``graph.graph["build_report"]``.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required: pip install networkx")

    graph = nx.MultiDiGraph()
    stats: Counter[str] = Counter()
    review_queue: list[dict[str, Any]] = []
    pending_relations: list[tuple[dict[str, Any], dict[str, Any]]] = []

    limit = len(documents) if not max_docs else min(len(documents), max_docs)
    logger.info("Building graph from %d/%d chunks", limit, len(documents))

    for i, doc in enumerate(documents[:limit]):
        level_rank = doc.metadata.get("level_rank", 1)
        src_meta = {
            "document": doc.metadata.get("doc_id") or doc.metadata.get("source") or doc.metadata.get("source_path", "?"),
            "chunk": doc.metadata.get("chunk_id", doc.metadata.get("chunk_index", i)),
        }
        if "page" in doc.metadata:
            src_meta["page"] = doc.metadata["page"]
        logger.info("[%d/%d] Extracting from: %s", i + 1, limit, src_meta["document"])
        payload = extract_entities_and_relations(doc.page_content, llm, level_rank, stats)

        for entity in payload["entities"]:
            stats["entities_seen"] += 1
            etype = normalize_entity_type(entity.get("type"))
            eid = canonical_id(entity.get("id") or entity.get("name"))
            if not eid:
                stats["entities_dropped_no_id"] += 1
                continue
            if etype is None:
                stats["entities_dropped_invalid_type"] += 1
                review_queue.append({"kind": "entity", "reason": "invalid_type", "item": entity, "source": src_meta})
                continue
            if _confidence(entity) < min_confidence:
                stats["entities_low_confidence"] += 1
                review_queue.append({"kind": "entity", "reason": "low_confidence", "item": entity, "source": src_meta})
                continue

            aliases = {a for a in entity.get("aliases", []) if isinstance(a, str) and a.strip()}
            desc = (entity.get("description") or "").strip()
            if eid not in graph:
                graph.add_node(
                    eid,
                    id=eid,
                    type=etype,
                    name=entity.get("name") or eid,
                    description=desc,
                    descriptions=[desc] if desc else [],
                    aliases=sorted(aliases),
                    level_rank=entity.get("level_rank", level_rank),
                    confidence=_confidence(entity),
                    sources=[],
                )
            else:
                node = graph.nodes[eid]
                if desc and desc not in node["descriptions"]:
                    node["descriptions"].append(desc)
                    node["description"] = " | ".join(node["descriptions"])
                node["aliases"] = sorted(set(node["aliases"]) | aliases)
                # Conservative RBAC: keep the highest (most restricted) level.
                node["level_rank"] = max(node.get("level_rank", 1), entity.get("level_rank", level_rank))
                node["confidence"] = max(node.get("confidence", 0.0), _confidence(entity))
            _add_source(graph.nodes[eid], {**src_meta, "evidence": entity.get("evidence", "")})

        for rel in payload["relations"]:
            pending_relations.append((rel, src_meta))
            stats["relations_seen"] += 1

    # Relations are resolved after all entities so cross-chunk endpoints match.
    for rel, src_meta in pending_relations:
        rtype = normalize_relation_type(rel.get("relation"))
        src = canonical_id(rel.get("source"))
        tgt = canonical_id(rel.get("target"))
        if rtype is None:
            stats["relations_dropped_invalid_type"] += 1
            review_queue.append({"kind": "relation", "reason": "invalid_type", "item": rel, "source": src_meta})
            continue
        if src not in graph or tgt not in graph:
            stats["relations_dropped_unknown_endpoint"] += 1
            logger.debug("Dropping relation with unknown endpoint: %s -[%s]-> %s", src, rtype, tgt)
            review_queue.append({"kind": "relation", "reason": "unknown_endpoint", "item": rel, "source": src_meta})
            continue
        violation = check_relation(rtype, graph.nodes[src].get("type", ""), graph.nodes[tgt].get("type", ""))
        if violation and check_relation(rtype, graph.nodes[tgt].get("type", ""), graph.nodes[src].get("type", "")) is None:
            # Same fact, written backwards (e.g. "threshold threshold_for SOP"): fix direction.
            src, tgt = tgt, src
            violation = None
            stats["relations_direction_fixed"] += 1
        if violation:
            stats["relations_dropped_ontology"] += 1
            review_queue.append({"kind": "relation", "reason": f"ontology: {violation}", "item": rel, "source": src_meta})
            continue
        if _confidence(rel) < min_confidence:
            stats["relations_low_confidence"] += 1
            review_queue.append({"kind": "relation", "reason": "low_confidence", "item": rel, "source": src_meta})
            continue

        source_entry = {**src_meta, "evidence": rel.get("evidence", "")}
        if graph.has_edge(src, tgt, key=rtype):
            edge = graph.edges[src, tgt, rtype]
            desc = (rel.get("description") or "").strip()
            if desc and desc not in edge["description"]:
                edge["description"] = (edge["description"] + " | " + desc).strip(" |")
            edge["level_rank"] = max(edge.get("level_rank", 1), rel.get("level_rank", 1))
            edge["confidence"] = max(edge.get("confidence", 0.0), _confidence(rel))
            _add_source(edge, source_entry)
        else:
            graph.add_edge(
                src, tgt, key=rtype,
                relation=rtype,
                description=(rel.get("description") or "").strip(),
                level_rank=rel.get("level_rank", 1),
                confidence=_confidence(rel),
                sources=[source_entry],
            )
            stats["relations_added"] += 1

    if stats["windows"] and stats["failed_windows"] / stats["windows"] > MAX_FAILURE_RATE:
        raise ExtractionAborted(
            f"{stats['failed_windows']}/{stats['windows']} extraction windows failed "
            f"(> {MAX_FAILURE_RATE:.0%}); refusing to return a partial graph."
        )

    graph.graph["build_report"] = dict(stats)
    graph.graph["review_queue"] = review_queue
    logger.info(
        "Graph built: %d nodes, %d edges, %d items for review; stats=%s",
        graph.number_of_nodes(), graph.number_of_edges(), len(review_queue), dict(stats),
    )
    if stats["failed_windows"]:
        logger.warning("%d extraction window(s) failed permanently", stats["failed_windows"])
    return graph


def detect_communities(graph: Any) -> dict[Any, int]:
    """Detect Louvain communities, falling back to greedy modularity."""
    import networkx as nx

    # Community algorithms need a simple undirected graph.
    simple = nx.Graph(graph.to_undirected()) if graph.is_directed() or graph.is_multigraph() else graph
    try:
        from community import best_partition
        return best_partition(simple)
    except ImportError:
        pass
    try:
        import networkx.algorithms.community as nx_comm
        communities = list(nx_comm.greedy_modularity_communities(simple))
        return {node: i for i, comm in enumerate(communities) for node in comm}
    except Exception as exc:
        logger.warning("Community detection failed: %s", exc)
        return {node: 0 for node in graph.nodes}


def save_graph(graph: Any, graph_store_dir: str) -> None:
    """Persist the graph as JSON for fast loading."""
    import networkx as nx
    store_dir = Path(graph_store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(graph)
    with open(store_dir / "graph.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Graph saved to %s/graph.json", graph_store_dir)


def write_review_queue(graph: Any, path: str = "reports/graph_review_queue.json") -> int:
    """Write rejected / low-confidence items for human review; returns the count."""
    queue = graph.graph.get("review_queue", []) if graph is not None else []
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {"build_report": graph.graph.get("build_report", {}), "items": queue},
            f, indent=2, default=str,
        )
    return len(queue)


def load_graph(graph_store_dir: str) -> Any | None:
    """Load graph from JSON, returning None if not found."""
    graph_path = Path(graph_store_dir) / "graph.json"
    if not graph_path.exists():
        logger.info("No graph found at %s — graph evidence disabled.", graph_path)
        return None
    try:
        import networkx as nx
        with open(graph_path, encoding="utf-8") as f:
            data = json.load(f)
        return nx.node_link_graph(data)
    except Exception as exc:
        logger.warning("Failed to load graph: %s", exc)
        return None


def main() -> None:
    import argparse
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    from langchain_chroma import Chroma

    from nga.config import Settings
    from nga.ingestion.build_vector_store import (
        CORPUS_PROFILES,
        profile_collection_name,
        profile_store_dir,
    )
    from nga.providers.factory import make_chat_model, make_embeddings

    parser = argparse.ArgumentParser(description="Build GraphRAG knowledge graph")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        metavar="N",
        help="Max number of chunks to process (0 or omit = all). "
             "Overrides GRAPH_MAX_DOCS env var.",
    )
    parser.add_argument(
        "--profile",
        choices=CORPUS_PROFILES,
        default=None,
        help="Corpus profile (default: from CORPUS_PROFILE env, 'main')",
    )
    parser.add_argument(
        "--skip-alignment",
        action="store_true",
        help="Skip TypeSafe AI entity alignment post-processing step",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=MIN_CONFIDENCE,
        help="Items below this confidence go to the review queue instead of the graph",
    )
    parser.add_argument(
        "--review-queue",
        default="reports/graph_review_queue.json",
        help="Where to write rejected / low-confidence items",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    profile = args.profile or settings.corpus_profile

    # Resolve max_docs: CLI flag > env var (GRAPH_MAX_DOCS) > all docs
    if args.max_docs is not None:
        max_docs = args.max_docs if args.max_docs > 0 else None
    elif settings.graph_max_docs > 0:
        max_docs = settings.graph_max_docs
    else:
        max_docs = None  # process all

    # Load vector store documents for graph extraction (profile-aware)
    embeddings = make_embeddings(settings)

    store = Chroma(
        collection_name=profile_collection_name(profile),
        embedding_function=embeddings,
        persist_directory=profile_store_dir(settings, profile),
    )
    all_docs = store.get(include=["documents", "metadatas"])
    from langchain_core.documents import Document
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    logger.info("Extracting entities from %d chunks...", len(docs))

    llm = make_chat_model(settings)
    graph = build_graph_from_documents(docs, llm, max_docs=max_docs, min_confidence=args.min_confidence)

    if not args.skip_alignment:
        from nga.ingestion.entity_alignment import align_entities
        logger.info("Running TypeSafe AI entity alignment...")
        # require_api_key=True ensures failure if neither TYPESAFE_API_KEY nor TYPESAFE_API_TEST is set
        alignment_report = align_entities(graph, require_api_key=True)
        logger.info("Entity alignment complete: %s", alignment_report)

    n_review = write_review_queue(graph, args.review_queue)
    logger.info("Wrote %d review items to %s", n_review, args.review_queue)

    # Keep the saved graph lean: the queue lives in its own report file.
    graph.graph.pop("review_queue", None)
    save_graph(graph, settings.graph_store_dir)
    logger.info("GraphRAG build complete.")


if __name__ == "__main__":
    main()
