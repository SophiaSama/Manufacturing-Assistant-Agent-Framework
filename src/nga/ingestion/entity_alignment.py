"""Knowledge graph entity alignment using TypeSafe AI System One models.

Follows the TypeSafe entity alignment cookbook:
https://docs.typesafe.ai/cookbooks/entity_alignment

Decides whether two entity descriptions refer to the same real-world entity
using a Score question (different / related / same) plus companion Noul questions
for name, type, and description comparison.
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import networkx as nx
from typesafe_sdk import Noul, Score, TypeSafeClient

logger = logging.getLogger(__name__)

# Alignment levels according to the TypeSafe cookbook pattern
LEVELS = [
    "They describe two different things.",
    "They describe closely related or ambiguous things that may or may not be the same one: "
    "a sub-component, a variant, or a name that could plausibly refer to either.",
    "They describe one and the same entity.",
]

OUTCOME = {0: "leave_separate", 1: "review", 2: "merge"}

QUESTIONS = {
    "link_state": Score(
        instructions="How do the two entity descriptions relate as manufacturing entities?",
        criteria=LEVELS,
    ),
    "same_name": Noul(
        instructions="Do the two entities state or refer to the same item name or identifier?",
    ),
    "same_type": Noul(
        instructions="Are the two entities describing the same category or type of entity?",
    ),
    "same_description": Noul(
        instructions="Do the two entity descriptions refer to the exact same functional role or equipment?",
    ),
}


def get_typesafe_api_key() -> str | None:
    """Resolve TypeSafe API key from environment variables."""
    return os.environ.get("TYPESAFE_API_KEY") or os.environ.get("TYPESAFE_API_TEST")


def get_typesafe_model() -> str:
    """Resolve TypeSafe model, defaulting back to jev-1.12 if not set."""
    return os.environ.get("TYPESAFE_MODEL", "jev-1.12")


def create_typesafe_client(api_key: str | None = None) -> TypeSafeClient:
    """Create a TypeSafeClient configured for the environment."""
    key = api_key or get_typesafe_api_key()
    try:
        import httpx2
        http_client = httpx2.Client(verify=False, timeout=60.0)
        return TypeSafeClient(api_key=key, http_client=http_client)
    except Exception:
        return TypeSafeClient(api_key=key)



def _tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase alphanumeric words."""
    if not text:
        return set()
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return {w for w in cleaned.split() if len(w) > 1}


def _extract_codes(text: str) -> set[str]:
    """Extract machine/part/SOP codes like 'rb-07', 'tq-6012', 'sop-opr-101', 'e-4012'."""
    if not text:
        return set()
    pattern = r"[a-zA-Z]{1,4}[-_]?[0-9]{2,5}[a-zA-Z]?"
    matches = re.findall(pattern, text.lower())
    return {m.replace("_", "-") for m in matches}


def generate_candidate_pairs(graph: nx.Graph) -> list[tuple[str, str]]:
    """Generate candidate duplicate pairs using a fast blocking pass.

    Pairs are filtered by:
    1. Matching or compatible entity types
    2. Substring overlap, common technical code/number, or token similarity
    """
    candidates: set[tuple[str, str]] = set()
    nodes = list(graph.nodes(data=True))

    for i in range(len(nodes)):
        id_a, data_a = nodes[i]
        type_a = str(data_a.get("type", "")).lower()
        name_a = str(data_a.get("name", "")).lower()
        id_a_clean = id_a.lower().replace("_", "-")
        tokens_a = _tokenize(name_a) | _tokenize(id_a_clean)
        codes_a = _extract_codes(id_a_clean) | _extract_codes(name_a)

        for j in range(i + 1, len(nodes)):
            id_b, data_b = nodes[j]
            type_b = str(data_b.get("type", "")).lower()
            name_b = str(data_b.get("name", "")).lower()
            id_b_clean = id_b.lower().replace("_", "-")
            tokens_b = _tokenize(name_b) | _tokenize(id_b_clean)
            codes_b = _extract_codes(id_b_clean) | _extract_codes(name_b)

            # Must have compatible types unless types are generic/missing
            if type_a and type_b and type_a != type_b:
                # Allow Machine/Part overlap, Fault/FaultCode overlap, Procedure/SOP overlap
                compat = {
                    frozenset(["machine", "part"]),
                    frozenset(["fault", "faultcode"]),
                    frozenset(["procedure", "sop"]),
                }
                if frozenset([type_a, type_b]) not in compat:
                    continue

            is_candidate = False

            # Check 1: Shared technical code (e.g. "e-4012", "rb-07", "tq-6012")
            if codes_a and codes_b:
                if codes_a & codes_b:
                    is_candidate = True

            # Check 2: Substring inclusion in IDs or names
            if not is_candidate:
                if (
                    id_a_clean in id_b_clean
                    or id_b_clean in id_a_clean
                    or (name_a and name_b and (name_a in name_b or name_b in name_a))
                ):
                    is_candidate = True

            # Check 3: Token Jaccard overlap >= 0.5
            if not is_candidate and tokens_a and tokens_b:
                intersect = len(tokens_a & tokens_b)
                union = len(tokens_a | tokens_b)
                if union > 0 and (intersect / union) >= 0.5:
                    is_candidate = True

            if is_candidate:
                pair = tuple(sorted([id_a, id_b]))
                candidates.add(pair)

    return sorted(list(candidates))


def score_pair(
    client: TypeSafeClient,
    entity_a: dict[str, Any],
    entity_b: dict[str, Any],
    model: str | None = None,
) -> dict[str, Any]:
    """Score one candidate pair with TypeSafe."""
    model_name = model or get_typesafe_model()
    response = client.system_one(
        state={"entity_a": entity_a, "entity_b": entity_b},
        questions=QUESTIONS,
        model=model_name,
    )

    link = response.answers["link_state"]
    nouls = {
        k: response.answers[k].noul
        for k in QUESTIONS
        if k != "link_state"
    }

    return {
        "score": link.score,
        "probabilities": link.probabilities,
        "confidence": link.confidence,
        "properties": nouls,
        "input_tokens": getattr(response.usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(response.usage, "output_tokens", 0) or 0,
    }


def route(score_value: float) -> str:
    """The decision rule: round nearest level to outcome name."""
    level_index = min(int(score_value + 0.5), len(LEVELS) - 1)
    return OUTCOME[max(0, level_index)]


def merge_entities(graph: nx.Graph, merge_pairs: list[tuple[str, str]]) -> int:
    """Merge duplicate entity nodes in the graph.

    For each pair (id_a, id_b), merges the duplicate into the canonical node,
    preserving edges, aliases, descriptions, and highest level_rank.
    Returns the number of merged nodes.
    """
    # Build disjoint-set / equivalence groups to handle transitive merges
    parent: dict[str, str] = {}

    def find(i: str) -> str:
        if i not in parent:
            parent[i] = i
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i: str, j: str) -> None:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    for u, v in merge_pairs:
        if u in graph and v in graph:
            union(u, v)

    groups: dict[str, set[str]] = {}
    for node in list(graph.nodes):
        root = find(node)
        groups.setdefault(root, set()).add(node)

    merged_count = 0

    for _, cluster in groups.items():
        if len(cluster) <= 1:
            continue

        # Choose canonical node: prefer the one without prefix ("machine-", "procedure-", etc.)
        def canonical_key(nid: str) -> tuple[int, int, str]:
            has_redundant_prefix = any(
                nid.startswith(p)
                for p in ["machine-", "procedure-", "personnel-", "nc-record-", "part-"]
            )
            return (1 if has_redundant_prefix else 0, len(nid), nid)

        cluster_sorted = sorted(cluster, key=canonical_key)
        canonical = cluster_sorted[0]
        duplicates = cluster_sorted[1:]

        c_data = graph.nodes[canonical]
        aliases = set(c_data.get("aliases", []))
        aliases.add(canonical)
        c_name = c_data.get("name", "")
        if c_name:
            aliases.add(c_name)

        combined_desc = c_data.get("description", "")
        max_level_rank = c_data.get("level_rank", 1)

        for dup in duplicates:
            if dup not in graph:
                continue
            d_data = graph.nodes[dup]
            aliases.add(dup)
            d_name = d_data.get("name", "")
            if d_name:
                aliases.add(d_name)
            for a in d_data.get("aliases", []):
                aliases.add(a)

            d_desc = d_data.get("description", "")
            if d_desc and d_desc not in combined_desc:
                combined_desc = (combined_desc + " | " + d_desc).strip(" | ")[:600]

            max_level_rank = max(max_level_rank, d_data.get("level_rank", 1))

            # Rewire all edges from duplicate to canonical
            for neighbor, edge_data in list(graph[dup].items()):
                if neighbor != canonical and not graph.has_edge(canonical, neighbor):
                    graph.add_edge(canonical, neighbor, **edge_data)

            graph.remove_node(dup)
            merged_count += 1

        # Update canonical node attributes
        c_data["aliases"] = sorted(list(aliases))
        c_data["description"] = combined_desc
        c_data["level_rank"] = max_level_rank

    logger.info("Entity merging complete: merged %d duplicate nodes.", merged_count)
    return merged_count


def align_entities(
    graph: nx.Graph,
    client: TypeSafeClient | None = None,
    model: str | None = None,
    max_workers: int = 6,
    review_queue_path: str = "reports/alignment_review_queue.json",
    require_api_key: bool = True,
) -> dict[str, Any]:
    """Top-level entity alignment workflow.

    1. Checks TypeSafe API key (or raises ValueError if require_api_key=True).
    2. Identifies candidate duplicate pairs via blocking.
    3. Scores pairs using TypeSafe Score + Nouls.
    4. Routes decisions: merges confirmed duplicates, saves review queue.
    """
    api_key = get_typesafe_api_key()
    if not api_key:
        if require_api_key:
            raise ValueError(
                "Neither TYPESAFE_API_KEY nor TYPESAFE_API_TEST is set. Cannot perform entity alignment."
            )
        logger.warning(
            "Neither TYPESAFE_API_KEY nor TYPESAFE_API_TEST is set. Skipping entity alignment."
        )
        return {
            "status": "skipped",
            "reason": "Missing API key",
            "candidates_found": 0,
            "merged_count": 0,
            "review_count": 0,
        }

    if client is None:
        client = create_typesafe_client(api_key=api_key)

    model_name = model or get_typesafe_model()
    candidates = generate_candidate_pairs(graph)
    logger.info(
        "Candidate generation complete: %d candidate pairs identified.",
        len(candidates),
    )

    if not candidates:
        return {
            "status": "completed",
            "candidates_found": 0,
            "merged_count": 0,
            "review_count": 0,
        }

    def process_pair(pair: tuple[str, str]) -> tuple[tuple[str, str], dict[str, Any], str]:
        u, v = pair
        data_a = dict(graph.nodes[u])
        data_b = dict(graph.nodes[v])
        data_a["id"] = u
        data_b["id"] = v
        res = score_pair(client, data_a, data_b, model=model_name)
        decision = route(res["score"])
        return pair, res, decision

    scored_results: list[tuple[tuple[str, str], dict[str, Any], str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        scored_results = list(executor.map(process_pair, candidates))

    to_merge: list[tuple[str, str]] = []
    review_queue: list[dict[str, Any]] = []
    leave_separate: list[tuple[str, str]] = []

    for pair, res, decision in scored_results:
        u, v = pair
        if decision == "merge":
            to_merge.append(pair)
        elif decision == "review":
            review_queue.append({
                "pair": [u, v],
                "entity_a": dict(graph.nodes[u]),
                "entity_b": dict(graph.nodes[v]),
                "score": res["score"],
                "confidence": res["confidence"],
                "properties": res["properties"],
            })
        else:
            leave_separate.append(pair)

    logger.info(
        "Scoring results: %d merge, %d review queue, %d separate",
        len(to_merge),
        len(review_queue),
        len(leave_separate),
    )

    # Save review queue if ambiguous items exist
    if review_queue:
        out_path = Path(review_queue_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(review_queue, f, indent=2)
        logger.info("Saved %d review queue pairs to %s", len(review_queue), out_path)

    # Merge duplicates in the graph
    merged_count = merge_entities(graph, to_merge)

    return {
        "status": "completed",
        "model": model_name,
        "candidates_found": len(candidates),
        "merged_count": merged_count,
        "review_count": len(review_queue),
        "separate_count": len(leave_separate),
    }
