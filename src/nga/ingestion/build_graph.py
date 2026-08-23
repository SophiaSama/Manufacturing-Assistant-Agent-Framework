"""GraphRAG knowledge graph stub for the NGA corpus.

Builds a NetworkX entity-relation graph from the ChromaDB documents using
LLM extraction with Louvain community detection for multi-hop reasoning.

Run: uv run python -m nga.ingestion.build_graph
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# NGA entity types for extraction
NGA_ENTITY_TYPES = [
    "Machine", "Station", "SOP", "FaultCode", "DefectCode", "Part",
    "Supplier", "Personnel", "Torque", "Threshold", "Procedure",
    "NCRecord", "WorkOrder", "RecallCriteria", "EscalationLevel",
]

NGA_RELATION_TYPES = [
    "causes", "indicates", "prevents", "monitors", "threshold_for",
    "documented_in", "escalates_to", "requires", "audited_by",
    "owned_by", "certified_for", "triggers_recall",
]

EXTRACTION_PROMPT = """Extract manufacturing knowledge entities and relations from the text below.

Entity types: {entity_types}
Relation types: {relation_types}

Return a JSON object with:
{{
  "entities": [
    {{"id": "unique_slug", "type": "EntityType", "name": "Display Name",
      "description": "brief description", "level_rank": 1}}
  ],
  "relations": [
    {{"source": "entity_id_1", "relation": "relation_type",
      "target": "entity_id_2", "description": "context", "level_rank": 1}}
  ]
}}

Rules:
- Keep entity IDs as lowercase slug (e.g. "tq-6012", "sop-opr-101")
- Use level_rank matching the source document's access level
- Only extract entities explicitly mentioned in the text
- Return valid JSON only, no markdown

Text:
{text}
"""


def extract_entities_and_relations(
    text: str, llm: Any, level_rank: int = 1
) -> dict[str, Any]:
    """Run LLM entity/relation extraction over a text chunk."""
    prompt = EXTRACTION_PROMPT.format(
        entity_types=", ".join(NGA_ENTITY_TYPES),
        relation_types=", ".join(NGA_RELATION_TYPES),
        text=text[:3000],  # limit to avoid token overflow
    )
    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        # Strip markdown fences if present
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        payload = json.loads(content)
        # Enforce level_rank from source
        for e in payload.get("entities", []):
            e.setdefault("level_rank", level_rank)
        for r in payload.get("relations", []):
            r.setdefault("level_rank", level_rank)
        return payload
    except Exception as exc:
        logger.debug("Entity extraction failed: %s", exc)
        return {"entities": [], "relations": []}


def build_graph_from_documents(
    documents: list[Any],
    llm: Any,
    max_docs: int = 50,
) -> Any:
    """Build a NetworkX knowledge graph from chunked documents."""
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required: pip install networkx")

    graph = nx.Graph()
    entity_registry: dict[str, dict] = {}

    for i, doc in enumerate(documents[:max_docs]):
        level_rank = doc.metadata.get("level_rank", 1)
        payload = extract_entities_and_relations(doc.page_content, llm, level_rank)

        for entity in payload.get("entities", []):
            eid = entity.get("id", "").lower().strip()
            if not eid:
                continue
            if eid not in entity_registry:
                entity_registry[eid] = entity
                graph.add_node(eid, **entity)
            else:
                # Merge descriptions, take max level_rank (conservative)
                existing = entity_registry[eid]
                if entity.get("description") and existing.get("description"):
                    merged = existing["description"] + " | " + entity["description"]
                    graph.nodes[eid]["description"] = merged[:500]
                graph.nodes[eid]["level_rank"] = max(
                    existing.get("level_rank", 1), entity.get("level_rank", 1)
                )

        for rel in payload.get("relations", []):
            src = rel.get("source", "").lower()
            tgt = rel.get("target", "").lower()
            rtype = rel.get("relation", "related")
            if src in graph and tgt in graph:
                graph.add_edge(src, tgt, relation=rtype,
                               description=rel.get("description", ""),
                               level_rank=rel.get("level_rank", 1))

        if (i + 1) % 10 == 0:
            logger.info("Processed %d/%d documents for graph extraction", i+1, len(documents))

    logger.info(
        "Graph built: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges()
    )
    return graph


def detect_communities(graph: Any) -> dict[Any, int]:
    """Detect Louvain communities, falling back to greedy modularity."""
    try:
        from community import best_partition
        return best_partition(graph)
    except ImportError:
        pass
    try:
        import networkx.algorithms.community as nx_comm
        communities = list(nx_comm.greedy_modularity_communities(graph))
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
        json.dump(data, f, indent=2)
    logger.info("Graph saved to %s/graph.json", graph_store_dir)


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
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    from nga.config import Settings
    from nga.providers.factory import make_chat_model
    from nga.ingestion.build_vector_store import discover_documents
    from langchain_chroma import Chroma
    from nga.providers.factory import make_embeddings

    settings = Settings.from_env()
    base_dir = Path(settings.nga_db_path).parent.parent

    # Load vector store documents for graph extraction
    embeddings = make_embeddings(settings)
    store = Chroma(
        collection_name="nga_reference_docs",
        embedding_function=embeddings,
        persist_directory=settings.vector_store_dir,
    )
    all_docs = store.get(include=["documents", "metadatas"])
    from langchain_core.documents import Document
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    logger.info("Extracting entities from %d chunks...", len(docs))

    llm = make_chat_model(settings)
    graph = build_graph_from_documents(docs, llm)
    save_graph(graph, settings.graph_store_dir)
    logger.info("GraphRAG build complete.")


if __name__ == "__main__":
    main()
