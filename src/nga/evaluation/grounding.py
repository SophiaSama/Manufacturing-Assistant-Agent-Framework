"""Deterministic retrieval-grounding hallucination evaluation.

Verifies that every citation in the agent's answer traces back to a real
chunk and document in the corpus.  Two verification tiers:

- Tier 1 — Citation Existence: validates cited doc_ids / chunk_ids exist
  in the corpus manifest.
- Tier 2 — Retrieval Provenance: verifies cited documents actually appeared
  in retrieval tool outputs (no orphan citations).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("nga.evaluation.grounding")

# Pattern for well-formed chunk IDs produced by build_vector_store:
#   {doc_id}:c{index}:{source_tag}
#   e.g. "SOP-OPR-101:c0:can", "QCR-501:c2:var"
_CHUNK_ID_RE = re.compile(r"^[\w-]+:c\d+:(can|var)$")

GROUNDING_THRESHOLD = 0.70


@dataclass(frozen=True)
class GroundingResult:
    """Result of deterministic grounding evaluation for a single question."""

    question_id: str

    # Tier 1 — Citation Existence
    citation_validity_score: float = 0.0  # valid citations / total citations
    fabricated_citations: list[str] = field(default_factory=list)
    invalid_chunk_ids: list[str] = field(default_factory=list)
    total_citations: int = 0
    valid_citations: int = 0

    # Tier 2 — Retrieval Provenance
    provenance_score: float = 0.0  # cited-and-retrieved / total cited
    orphan_citations: list[str] = field(default_factory=list)
    retrieval_refs_valid: int = 0
    retrieval_refs_total: int = 0

    # Composite
    grounding_score: float = 0.0
    grounded: bool = False


# ---------------------------------------------------------------------------
# Corpus manifest
# ---------------------------------------------------------------------------

def build_corpus_manifest(
    base_dir: Path,
    profile: str = "main",
    variant_dir: Path | None = None,
) -> set[str]:
    """Build the set of valid doc_ids from the corpus for citation validation.

    Uses the ingestion pipeline's ``build_documents()`` to discover all
    documents and extract their ``doc_id`` metadata — guaranteeing the
    manifest is consistent with what was actually ingested.
    """
    from nga.ingestion.build_vector_store import build_documents

    kwargs: dict[str, Any] = {"base_dir": base_dir, "profile": profile}
    if variant_dir is not None:
        kwargs["variant_dir"] = variant_dir
    docs = build_documents(**kwargs)
    manifest = {d.metadata["doc_id"] for d in docs if d.metadata.get("doc_id")}
    logger.info(
        "Built corpus manifest: profile=%s doc_ids=%d", profile, len(manifest),
    )
    return manifest


# ---------------------------------------------------------------------------
# Tier 1 — Citation Existence
# ---------------------------------------------------------------------------

def _extract_answer_citations(
    answer_text: str,
    evidence: list[dict[str, Any]],
) -> list[str]:
    """Extract all document IDs cited in the answer and evidence list."""
    citations: list[str] = []

    # From structured evidence references
    for ref in evidence:
        citation = ref.get("citation", "")
        if citation and isinstance(citation, str):
            citations.append(citation.strip())

    # From inline [DOC-ID] references in the answer text
    inline = re.findall(r"\[([A-Z][\w-]*-\d+[A-Za-z]*)\]", answer_text)
    for doc_id in inline:
        if doc_id not in citations:
            citations.append(doc_id)

    return citations


def _extract_chunk_ids_from_tool_outputs(
    tool_outputs: list[str],
) -> list[str]:
    """Extract all chunk_ids from retrieval tool output payloads."""
    chunk_ids: list[str] = []
    for output in tool_outputs:
        try:
            payload = json.loads(output)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        for ref in payload.get("references", []):
            if isinstance(ref, dict) and ref.get("chunk_id"):
                chunk_ids.append(str(ref["chunk_id"]))
        for result in payload.get("results", []):
            if isinstance(result, dict) and result.get("chunk_id"):
                cid = str(result["chunk_id"])
                if cid not in chunk_ids:
                    chunk_ids.append(cid)
    return chunk_ids


def evaluate_citation_existence(
    answer_text: str,
    evidence: list[dict[str, Any]],
    tool_outputs: list[str],
    corpus_manifest: set[str],
) -> dict[str, Any]:
    """Tier 1: Validate that cited doc_ids and chunk_ids exist in the corpus."""
    citations = _extract_answer_citations(answer_text, evidence)
    chunk_ids = _extract_chunk_ids_from_tool_outputs(tool_outputs)

    # Validate doc_id citations against manifest
    fabricated: list[str] = []
    valid_count = 0
    for cit in citations:
        if cit in corpus_manifest:
            valid_count += 1
        else:
            fabricated.append(cit)

    total = len(citations)
    citation_score = valid_count / max(total, 1) if total > 0 else 1.0

    # Validate chunk_id format and doc_id prefix
    invalid_chunks: list[str] = []
    for cid in chunk_ids:
        if not _CHUNK_ID_RE.match(cid):
            invalid_chunks.append(cid)
        else:
            # Extract doc_id prefix and check against manifest
            doc_prefix = cid.rsplit(":c", 1)[0]
            if doc_prefix not in corpus_manifest:
                invalid_chunks.append(cid)

    return {
        "citation_validity_score": round(citation_score, 4),
        "fabricated_citations": fabricated,
        "invalid_chunk_ids": invalid_chunks,
        "total_citations": total,
        "valid_citations": valid_count,
    }


# ---------------------------------------------------------------------------
# Tier 2 — Retrieval Provenance
# ---------------------------------------------------------------------------

def _extract_retrieved_doc_ids(tool_outputs: list[str]) -> set[str]:
    """Extract all doc_ids that the retrieval tool actually returned."""
    retrieved: set[str] = set()
    for output in tool_outputs:
        try:
            payload = json.loads(output)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        # From references array
        for ref in payload.get("references", []):
            if isinstance(ref, dict) and ref.get("doc_id"):
                retrieved.add(str(ref["doc_id"]))
        # From results array
        for result in payload.get("results", []):
            if isinstance(result, dict) and result.get("doc_id"):
                retrieved.add(str(result["doc_id"]))
    return retrieved


def _count_valid_retrieval_refs(tool_outputs: list[str]) -> tuple[int, int]:
    """Count retrieval references that have complete metadata (doc_id + chunk_id + corpus_source)."""
    valid = 0
    total = 0
    for output in tool_outputs:
        try:
            payload = json.loads(output)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        for ref in payload.get("references", []):
            if not isinstance(ref, dict):
                continue
            total += 1
            if ref.get("doc_id") and ref.get("chunk_id") and ref.get("corpus_source"):
                valid += 1
    return valid, total


def evaluate_retrieval_provenance(
    answer_text: str,
    evidence: list[dict[str, Any]],
    tool_outputs: list[str],
) -> dict[str, Any]:
    """Tier 2: Verify cited documents were actually retrieved by the tool."""
    citations = _extract_answer_citations(answer_text, evidence)
    retrieved_ids = _extract_retrieved_doc_ids(tool_outputs)
    refs_valid, refs_total = _count_valid_retrieval_refs(tool_outputs)

    orphan: list[str] = []
    cited_and_retrieved = 0
    for cit in citations:
        if cit in retrieved_ids:
            cited_and_retrieved += 1
        else:
            orphan.append(cit)

    total_cited = len(citations)
    provenance_score = (
        cited_and_retrieved / max(total_cited, 1) if total_cited > 0 else 1.0
    )

    return {
        "provenance_score": round(provenance_score, 4),
        "orphan_citations": orphan,
        "retrieval_refs_valid": refs_valid,
        "retrieval_refs_total": refs_total,
    }


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def evaluate_grounding(
    answer_text: str,
    evidence: list[dict[str, Any]],
    tool_outputs: list[str],
    corpus_manifest: set[str],
    question_id: str,
    threshold: float = GROUNDING_THRESHOLD,
) -> GroundingResult:
    """Run both deterministic grounding tiers and return a composite result.

    Parameters
    ----------
    answer_text:
        The agent's rendered answer string.
    evidence:
        List of evidence reference dicts (from ``FinalAnswer.evidence``).
    tool_outputs:
        Raw JSON strings from tool messages collected during the agent run.
    corpus_manifest:
        Set of valid ``doc_id`` values from ``build_corpus_manifest()``.
    question_id:
        Identifier for the evaluation question.
    threshold:
        Minimum composite score to consider the answer "grounded".
    """
    tier1 = evaluate_citation_existence(
        answer_text, evidence, tool_outputs, corpus_manifest,
    )
    tier2 = evaluate_retrieval_provenance(answer_text, evidence, tool_outputs)

    composite = round(
        0.5 * tier1["citation_validity_score"]
        + 0.5 * tier2["provenance_score"],
        4,
    )

    return GroundingResult(
        question_id=question_id,
        # Tier 1
        citation_validity_score=tier1["citation_validity_score"],
        fabricated_citations=tier1["fabricated_citations"],
        invalid_chunk_ids=tier1["invalid_chunk_ids"],
        total_citations=tier1["total_citations"],
        valid_citations=tier1["valid_citations"],
        # Tier 2
        provenance_score=tier2["provenance_score"],
        orphan_citations=tier2["orphan_citations"],
        retrieval_refs_valid=tier2["retrieval_refs_valid"],
        retrieval_refs_total=tier2["retrieval_refs_total"],
        # Composite
        grounding_score=composite,
        grounded=composite >= threshold,
    )
