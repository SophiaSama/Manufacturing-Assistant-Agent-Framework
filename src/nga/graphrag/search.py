"""GraphRAG search: RBAC-filtered entity/community retrieval for NGA corpus."""

from __future__ import annotations

import re
from typing import Any

from nga.ingestion.normalize import canonical_id, resolve_canonical_id


def _neighbors(graph: Any, node_id: Any) -> set[Any]:
    """Neighbors regardless of edge direction (DiGraph.neighbors is successors-only)."""
    if graph.is_directed():
        return set(graph.successors(node_id)) | set(graph.predecessors(node_id))
    return set(graph.neighbors(node_id))


_QUERY_STOPWORDS: frozenset[str] = frozenset({
    "what", "is", "are", "was", "were", "be", "been", "being",
    "the", "an", "and", "or", "for", "to", "in", "on", "at", "of",
    "by", "with", "from", "as", "into", "about", "before", "after",
    "how", "why", "when", "where", "who", "which", "that", "this",
    "these", "those", "it", "its", "they", "them", "their", "you", "your",
    "does", "do", "did", "can", "could", "would", "should", "must",
    "has", "have", "had", "used", "use", "if", "both", "all", "any",
})


def _tokenize_query(text: str) -> set[str]:
    if not text:
        return set()
    cleaned = re.sub(r"[^a-zA-Z0-9%\.\s\-]", " ", text.lower())
    tokens: set[str] = set()
    for w in cleaned.split():
        w_strip = w.strip("-")
        if len(w_strip) > 1 and w_strip not in _QUERY_STOPWORDS:
            tokens.add(w_strip)
            for part in w_strip.split("-"):
                if len(part) > 1 and part not in _QUERY_STOPWORDS:
                    tokens.add(part)
    return tokens


def _lexical_node_score(
    node_id: str,
    node_data: dict[str, Any],
    query_lower: str,
    query_slug: str,
    query_tokens: set[str],
    query_canonical_hits: set[str],
) -> float:
    """Compute a lexical + alias match score in [0.0, 1.0] for a graph node."""
    nid_str = str(node_id).lower()
    if nid_str in query_canonical_hits or (nid_str and len(nid_str) >= 3 and nid_str in query_slug):
        return 1.0

    name = str(node_data.get("name", "")).lower()
    aliases = [str(a).lower() for a in (node_data.get("aliases") or []) if a]
    for label in [name, *aliases]:
        if not label:
            continue
        lbl_slug = canonical_id(label)
        if (len(label) >= 3 and label in query_lower) or (len(lbl_slug) >= 3 and lbl_slug in query_slug):
            return 0.95

    if not query_tokens:
        return 0.05

    source_texts = [
        f"{s.get('document', '')} {s.get('evidence', '')}"
        for s in (node_data.get("sources") or [])
        if isinstance(s, dict)
    ]
    node_text = " ".join(
        [nid_str, name, *aliases, str(node_data.get("description", "")).lower(), *source_texts]
    ).lower()
    node_tokens = _tokenize_query(node_text)
    if not node_tokens:
        return 0.05

    overlap = len(query_tokens & node_tokens)
    if overlap == 0:
        return 0.05
    return min(0.85, 0.1 + 0.75 * (overlap / max(min(len(query_tokens), 6), 1)))


def _extract_doc_bridge_signals(
    graph: Any,
    retrieved_docs: list[dict[str, Any]] | None,
) -> tuple[set[str], set[str], set[str]]:
    """Extract canonical node hits and document provenance IDs from vector-retrieved chunks.

    Bridges semantic symptom queries (resolved by ChromaDB + Jev reranker) into
    Knowledge Graph seed entities without requiring exact entity codes in the user prompt.
    """
    bridged_doc_nodes: set[str] = set()
    bridged_entity_nodes: set[str] = set()
    bridged_doc_codes: set[str] = set()

    if not retrieved_docs or graph is None:
        return bridged_doc_nodes, bridged_entity_nodes, bridged_doc_codes

    for doc in retrieved_docs[:5]:
        if not isinstance(doc, dict):
            continue
        raw_doc_id = str(doc.get("doc_id") or "").strip()
        if raw_doc_id:
            bridged_doc_codes.add(raw_doc_id.upper())
            resolved_doc = resolve_canonical_id(raw_doc_id)
            if resolved_doc and resolved_doc in graph:
                bridged_doc_nodes.add(resolved_doc)

        chunk_text = str(doc.get("text") or "").lower()
        if chunk_text:
            for code in re.findall(r"[a-zA-Z]{1,4}[-_ ]?[0-9]{2,5}[a-zA-Z]?", chunk_text):
                resolved_ent = resolve_canonical_id(code)
                if resolved_ent and resolved_ent in graph:
                    bridged_entity_nodes.add(resolved_ent)
            if "class a" in chunk_text and "class-a-defect" in graph:
                bridged_entity_nodes.add("class-a-defect")

    return bridged_doc_nodes, bridged_entity_nodes, bridged_doc_codes


def build_graph_evidence(
    graph: Any,
    query: str,
    embeddings: Any,
    *,
    k_entities: int = 5,
    max_hops: int = 2,
    user_level: int = 1,
    retrieved_docs: list[dict[str, Any]] | None = None,
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

    # Embed query if embeddings provider is available
    query_embedding = None
    if embeddings is not None:
        try:
            query_embedding = embeddings.embed_query(query)
        except Exception:
            query_embedding = None

    query_lower = (query or "").lower()
    query_slug = canonical_id(query)
    query_tokens = _tokenize_query(query)

    # Check if any token or code in the query resolves to a known canonical ID
    code_matches = re.findall(r"[a-zA-Z]{1,4}[-_ ]?[0-9]{2,5}[a-zA-Z]?", query_lower)
    query_canonical_hits: set[str] = set()
    for candidate in [*code_matches, *query_tokens]:
        resolved = resolve_canonical_id(candidate)
        if resolved and resolved in graph:
            query_canonical_hits.add(resolved)

    bridged_doc_nodes, bridged_entity_nodes, bridged_doc_codes = _extract_doc_bridge_signals(
        graph, retrieved_docs
    )

    # Score nodes by hybrid lexical/alias + vector-doc bridge + cosine similarity to query
    scored: list[tuple[str, float]] = []
    for node_id, node_data in graph.nodes(data=True):
        node_level = node_data.get("level_rank", 1)
        if node_level > user_level:
            continue
        nid_str = str(node_id)
        lex_score = _lexical_node_score(
            nid_str, node_data, query_lower, query_slug, query_tokens, query_canonical_hits
        )
        if nid_str in bridged_doc_nodes:
            lex_score = max(lex_score, 0.92)
        elif nid_str in bridged_entity_nodes:
            lex_score = max(lex_score, 0.88)
        elif bridged_doc_codes:
            node_doc_codes = {
                str(s.get("document") or "").upper()
                for s in (node_data.get("sources") or [])
                if isinstance(s, dict)
            }
            if node_doc_codes & bridged_doc_codes:
                lex_score = min(0.86, lex_score + 0.35)
        node_emb = node_data.get("embedding")
        if node_emb is None or query_embedding is None:
            scored.append((node_id, lex_score))
            continue
        try:
            q = np.array(query_embedding)
            n = np.array(node_emb)
            cos_sim = float(np.dot(q, n) / (np.linalg.norm(q) * np.linalg.norm(n) + 1e-9))
            scored.append((node_id, max(cos_sim, lex_score)))
        except Exception:
            scored.append((node_id, lex_score))

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
            for neighbor in _neighbors(graph, node_id):
                n_data = graph.nodes[neighbor]
                if n_data.get("level_rank", 1) <= user_level and neighbor not in visited:
                    next_frontier.append(neighbor)
        frontier = next_frontier
        hop += 1

    # Build output (preserve seed score ordering, then rank expanded neighbors by score)
    score_map = dict(scored)
    ordered_visited = [n for n in seed_nodes if n in visited] + sorted(
        (n for n in visited if n not in seed_nodes),
        key=lambda n: score_map.get(n, 0.0),
        reverse=True,
    )
    entities: list[dict] = []
    communities_seen: dict[int, list[str]] = {}
    for node_id in ordered_visited:
        node_data = dict(graph.nodes[node_id])
        node_data.pop("embedding", None)
        entities.append({"id": node_id, **node_data})
        comm = node_data.get("community")
        if isinstance(comm, int):
            communities_seen.setdefault(comm, []).append(node_data.get("name") or str(node_id))

    relations: list[dict] = []
    for u, v, edge_data in graph.edges(data=True):
        if u in visited and v in visited:
            if edge_data.get("level_rank", 1) <= user_level:
                relations.append({"source": u, "target": v, **edge_data})

    community_summaries = [
        f"Community {cid}: {', '.join(members[:6])}"
        for cid, members in sorted(communities_seen.items())
        if len(members) >= 2
    ]

    return {
        "entities": entities[:20],
        "relations": relations[:30],
        "community_summaries": community_summaries[:5],
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
