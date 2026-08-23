"""GraphRAG search: RBAC-filtered entity/community retrieval for NGA corpus."""

from __future__ import annotations

from typing import Any


def build_graph_evidence(
    graph: Any,
    query: str,
    embeddings: Any,
    *,
    k_entities: int = 5,
    max_hops: int = 2,
    user_level: int = 1,
) -> dict[str, Any]:
    """Run RBAC-filtered graph-aware local search.

    Returns entities, relations, and community summaries accessible
    to the given user_level.
    """
    if graph is None:
        return {"entities": [], "relations": [], "community_summaries": []}

    try:
        import numpy as np
    except ImportError:
        return {"entities": [], "relations": [], "community_summaries": []}

    # Embed query and find seed entities by cosine similarity
    try:
        query_embedding = embeddings.embed_query(query)
    except Exception:
        return {"entities": [], "relations": [], "community_summaries": []}

    # Score nodes by cosine similarity to query
    scored: list[tuple[str, float]] = []
    for node_id, node_data in graph.nodes(data=True):
        node_level = node_data.get("level_rank", 1)
        if node_level > user_level:
            continue
        node_emb = node_data.get("embedding")
        if node_emb is None:
            # Use name as proxy — give small constant score
            scored.append((node_id, 0.1))
            continue
        try:
            q = np.array(query_embedding)
            n = np.array(node_emb)
            cos_sim = float(np.dot(q, n) / (np.linalg.norm(q) * np.linalg.norm(n) + 1e-9))
            scored.append((node_id, cos_sim))
        except Exception:
            scored.append((node_id, 0.0))

    scored.sort(key=lambda x: x[1], reverse=True)
    seed_nodes = [node_id for node_id, _ in scored[:k_entities]]

    # BFS expansion
    visited: set[str] = set()
    frontier = list(seed_nodes)
    hop = 0
    while frontier and hop < max_hops:
        next_frontier: list[str] = []
        for node_id in frontier:
            if node_id in visited:
                continue
            visited.add(node_id)
            for neighbor in graph.neighbors(node_id):
                n_data = graph.nodes[neighbor]
                if n_data.get("level_rank", 1) <= user_level and neighbor not in visited:
                    next_frontier.append(neighbor)
        frontier = next_frontier
        hop += 1

    # Build output
    entities: list[dict] = []
    for node_id in visited:
        node_data = dict(graph.nodes[node_id])
        node_data.pop("embedding", None)
        entities.append({"id": node_id, **node_data})

    relations: list[dict] = []
    for u, v, edge_data in graph.edges(data=True):
        if u in visited and v in visited:
            if edge_data.get("level_rank", 1) <= user_level:
                relations.append({"source": u, "target": v, **edge_data})

    return {
        "entities": entities[:20],
        "relations": relations[:30],
        "community_summaries": [],
    }


def format_graph_evidence(evidence: dict[str, Any]) -> str:
    """Format graph evidence as an LLM-readable context block."""
    lines: list[str] = []

    entities = evidence.get("entities", [])
    if entities:
        lines.append("=== Knowledge Graph Entities ===")
        for e in entities[:10]:
            name = e.get("name", e.get("id", ""))
            desc = e.get("description", "")
            etype = e.get("type", "")
            lines.append(f"[{etype}] {name}: {desc}")

    relations = evidence.get("relations", [])
    if relations:
        lines.append("\n=== Knowledge Graph Relations ===")
        for r in relations[:15]:
            lines.append(
                f"{r.get('source', '')} --{r.get('relation', '?')}--> "
                f"{r.get('target', '')}: {r.get('description', '')}"
            )

    return "\n".join(lines) if lines else ""
