"""Deterministic retrieval-grounding hallucination evaluation.

Backward-compatibility shim re-exporting from enterprise_agent.evaluation.grounding.
"""

from enterprise_agent.evaluation.grounding import (
    _CHUNK_ID_RE,
    GROUNDING_THRESHOLD,
    GroundingResult,
    _count_valid_retrieval_refs,
    _extract_answer_citations,
    _extract_chunk_ids_from_tool_outputs,
    _extract_retrieved_doc_ids,
    evaluate_citation_existence,
    evaluate_grounding,
    evaluate_retrieval_provenance,
)
from enterprise_agent.evaluation.grounding import (
    build_corpus_manifest as _build_corpus_manifest,
)


def build_corpus_manifest(
    base_dir,
    profile: str = "main",
    variant_dir=None,
    build_docs_fn=None,
):
    if build_docs_fn is None:
        try:
            from nga.ingestion.build_vector_store import (
                build_documents as build_docs_fn,
            )
        except ImportError:
            build_docs_fn = None
    return _build_corpus_manifest(
        base_dir,
        profile=profile,
        variant_dir=variant_dir,
        build_docs_fn=build_docs_fn,
    )


__all__ = [
    "GROUNDING_THRESHOLD",
    "_CHUNK_ID_RE",
    "GroundingResult",
    "build_corpus_manifest",
    "evaluate_citation_existence",
    "evaluate_grounding",
    "evaluate_retrieval_provenance",
    "_extract_answer_citations",
    "_extract_chunk_ids_from_tool_outputs",
    "_extract_retrieved_doc_ids",
    "_count_valid_retrieval_refs",
]
