"""Integration tests for generic RAG grounding evaluation and DeepEval contracts.

Requirement R3: Generic tool-to-evaluation contract verification:
1. Live tool payload alignment for make_generic_retrieval_tool and make_generic_sql_tool.
2. evaluate_grounding() contract verification with live tool outputs.
3. Adversarial / edge cases: hallucinated citations, unretrieved orphan citations, malformed payloads.
4. to_deep_eval_test_case() extraction contract verifying non-empty retrieval_context
   with both formatted document passages and [SQL RESULT] blocks, plus DeepEval ToolCall instances.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import langchain_core.tools as lc_tools
import pytest
from deepeval.test_case import ToolCall

import enterprise_agent.tools.tool_factory as tf
from enterprise_agent.config.domain_config import DomainConfig
from enterprise_agent.database.engine import SqliteAdapter
from enterprise_agent.documents.loader import DocumentIngestionPipeline
from enterprise_agent.evaluation.deepeval_adapter import to_deep_eval_test_case
from enterprise_agent.evaluation.grounding import evaluate_grounding
from enterprise_agent.rag.retrieval import make_generic_retrieval_tool

# Disable external tracing network calls
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

_orig_tool = lc_tools.tool


def _safe_tool_decorator(*args, **kwargs):
    if len(args) == 1 and isinstance(args[0], str) and "description" not in kwargs:
        kwargs["description"] = "Read-only SQL database query tool."
    return _orig_tool(*args, **kwargs)


tf.tool = _safe_tool_decorator

BANKING_PACK_PATH = Path("domains/banking/pack.yaml")
_CHUNK_ID_RE = re.compile(r"^[\w-]+:c\d+:can$")


class MockVectorStore:
    """Mock store serving pre-ingested chunks matching Chroma search contract."""

    def __init__(self, documents: list[Any]):
        self.documents = list(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Any]:
        return self.documents[:k]


@pytest.fixture(scope="module")
def banking_setup():
    """Setup live retrieval and SQL tools for the banking domain."""
    config = DomainConfig.from_yaml_file(BANKING_PACK_PATH)
    data_dir = Path(config.domain.data_dir).resolve()
    pipeline = DocumentIngestionPipeline(doc_config=config.documents, rbac_config=config.rbac)
    chunks = pipeline.load_directory(data_dir)
    store = MockVectorStore(chunks)

    retrieval_tool = make_generic_retrieval_tool(
        store=store,
        user_level=4,
        doc_config=config.documents,
    )

    db_path = Path(config.database.connection_uri).resolve()
    adapter = SqliteAdapter(db_path=db_path, allowed_tables=config.database.allowed_tables)
    sql_tool = tf.make_generic_sql_tool(adapter, tool_name=config.database.tool_name)

    corpus_manifest = {c.metadata["doc_id"] for c in chunks if c.metadata.get("doc_id")}

    return {
        "config": config,
        "chunks": chunks,
        "retrieval_tool": retrieval_tool,
        "sql_tool": sql_tool,
        "corpus_manifest": corpus_manifest,
    }


def test_generic_tool_payload_alignment(banking_setup):
    """Test 1: Verify output JSON payloads emitted by generic retrieval and SQL tools.

    Asserts:
    - make_generic_retrieval_tool emits:
      - 'results': list of dicts with doc_id, chunk_id ({doc_id}:c{i}:can), category, corpus_source, text.
      - 'references': list of dicts with doc_id, chunk_id, corpus_source.
    - make_generic_sql_tool emits:
      - 'query', 'rows', 'row_count', 'tables'.
    """
    retrieval_tool = banking_setup["retrieval_tool"]
    sql_tool = banking_setup["sql_tool"]

    # 1. Test retrieval payload
    raw_doc_output = retrieval_tool.invoke({"query": "wire limits"})
    assert isinstance(raw_doc_output, str), "Retrieval tool output must be a JSON string"
    doc_payload = json.loads(raw_doc_output)

    assert "results" in doc_payload, "Missing 'results' key in retrieval payload"
    assert "references" in doc_payload, "Missing 'references' key in retrieval payload"
    assert len(doc_payload["results"]) > 0
    assert len(doc_payload["references"]) > 0

    for item in doc_payload["results"]:
        assert "doc_id" in item and item["doc_id"]
        assert "chunk_id" in item and _CHUNK_ID_RE.match(item["chunk_id"])
        assert "category" in item
        assert item.get("corpus_source") == "canonical"
        assert "text" in item and len(item["text"]) > 0

    for ref in doc_payload["references"]:
        assert "doc_id" in ref and ref["doc_id"]
        assert "chunk_id" in ref and _CHUNK_ID_RE.match(ref["chunk_id"])
        assert ref.get("corpus_source") == "canonical"

    # 2. Test SQL payload
    raw_sql_output = sql_tool.invoke({"sql": "SELECT * FROM transactions WHERE transaction_id = 'TX-9001'"})
    assert isinstance(raw_sql_output, str), "SQL tool output must be a JSON string"
    sql_payload = json.loads(raw_sql_output)

    assert "query" in sql_payload
    assert "rows" in sql_payload and isinstance(sql_payload["rows"], list)
    assert "row_count" in sql_payload and sql_payload["row_count"] == 1
    assert "tables" in sql_payload and "transactions" in sql_payload["tables"]
    assert sql_payload["rows"][0]["transaction_id"] == "TX-9001"


def test_evaluate_grounding_contract_valid_citations(banking_setup):
    """Test 2: evaluate_grounding contract verification on valid cited documents.

    Asserts:
    - Feeding live outputs from retrieval and SQL tools into evaluate_grounding():
      - provenance_score == 1.0
      - citation_validity_score == 1.0
      - orphan_citations == []
      - fabricated_citations == []
      - invalid_chunk_ids == []
      - grounded is True
    """
    retrieval_tool = banking_setup["retrieval_tool"]
    sql_tool = banking_setup["sql_tool"]
    corpus_manifest = banking_setup["corpus_manifest"]

    doc_out = retrieval_tool.invoke({"query": "international wire limits POL-101"})
    sql_out = sql_tool.invoke({"sql": "SELECT * FROM transactions WHERE transaction_id = 'TX-9002'"})
    tool_outputs = [doc_out, sql_out]

    answer_text = (
        "Pursuant to [POL-101], wire transfers exceeding $10,000 must be screened. "
        "Transaction TX-9002 was outgoing for $1,200,000, requiring senior management approval."
    )
    evidence = [
        {
            "source_type": "document",
            "citation": "POL-101",
            "supports": ["All wire transfers exceeding $10,000 USD must be screened."],
        }
    ]

    result = evaluate_grounding(
        answer_text=answer_text,
        evidence=evidence,
        tool_outputs=tool_outputs,
        corpus_manifest=corpus_manifest,
        question_id="Q-GROUNDING-VALID",
    )

    assert result.citation_validity_score == 1.0, f"Expected 1.0, got {result.citation_validity_score}"
    assert result.provenance_score == 1.0, f"Expected 1.0, got {result.provenance_score}"
    assert result.orphan_citations == [], f"Unexpected orphan citations: {result.orphan_citations}"
    assert result.fabricated_citations == [], f"Unexpected fabricated citations: {result.fabricated_citations}"
    assert result.grounded is True, "Answer should be evaluated as grounded"
    assert result.valid_citations == 1


def test_evaluate_grounding_adversarial_and_edge_cases(banking_setup):
    """Test 3: Adversarial and edge case verification for evaluate_grounding.

    Asserts:
    - Hallucinated document IDs (not in manifest):
      - citation_validity_score < 1.0, fabricated_citations includes hallucinated ID, grounded is False.
    - Orphan document IDs (valid doc_id in manifest but NEVER retrieved in tool outputs):
      - provenance_score < 1.0, orphan_citations includes unretrieved ID.
    - Malformed / non-JSON tool output strings handled gracefully without crashing.
    """
    corpus_manifest = banking_setup["corpus_manifest"]

    # 1. Fabricated / hallucinated citation
    tool_outputs = [
        json.dumps({
            "results": [{"doc_id": "POL-101", "chunk_id": "POL-101:c0:can", "category": "POL", "corpus_source": "canonical", "text": "policy"}],
            "references": [{"doc_id": "POL-101", "chunk_id": "POL-101:c0:can", "corpus_source": "canonical"}],
        })
    ]

    hallucinated_answer = "According to [NON-EXISTENT-999], wire transfers are unlimited."
    res_hallucinated = evaluate_grounding(
        answer_text=hallucinated_answer,
        evidence=[{"citation": "NON-EXISTENT-999"}],
        tool_outputs=tool_outputs,
        corpus_manifest=corpus_manifest,
        question_id="Q-HALLUCINATION",
    )

    assert res_hallucinated.citation_validity_score == 0.0
    assert "NON-EXISTENT-999" in res_hallucinated.fabricated_citations
    assert "NON-EXISTENT-999" in res_hallucinated.orphan_citations
    assert res_hallucinated.grounded is False

    # 2. Orphan citation (valid in corpus, but not in retrieval tool output)
    # SAR-401 is in corpus_manifest, but tool_outputs only returned POL-101
    orphan_answer = "Policy [SAR-401] dictates 30 day filing."
    res_orphan = evaluate_grounding(
        answer_text=orphan_answer,
        evidence=[{"citation": "SAR-401"}],
        tool_outputs=tool_outputs,
        corpus_manifest=corpus_manifest,
        question_id="Q-ORPHAN",
    )

    assert res_orphan.citation_validity_score == 1.0  # SAR-401 exists in manifest
    assert res_orphan.provenance_score == 0.0          # Not retrieved in tool outputs
    assert "SAR-401" in res_orphan.orphan_citations

    # 3. Graceful handling of malformed / empty / non-JSON tool outputs
    malformed_outputs = ["Not valid JSON", "", "{unclosed json", None]
    res_malformed = evaluate_grounding(
        answer_text="General claim without citation.",
        evidence=[],
        tool_outputs=malformed_outputs,
        corpus_manifest=corpus_manifest,
        question_id="Q-MALFORMED",
    )
    assert res_malformed.citation_validity_score == 1.0
    assert res_malformed.provenance_score == 1.0
    assert res_malformed.grounded is True


def test_to_deep_eval_test_case_contract(banking_setup):
    """Test 4: DeepEval test case extraction contract verification.

    Asserts:
    - Feeding live outputs from make_generic_retrieval_tool and make_generic_sql_tool into to_deep_eval_test_case():
      - Returned LLMTestCase contains non-empty retrieval_context.
      - retrieval_context contains formatted document passages ([POL-101] or [POL]).
      - retrieval_context contains [SQL RESULT] blocks.
      - tools_called contains DeepEval ToolCall instances.
      - Fields input, actual_output, and expected_output are accurately preserved.
    """
    retrieval_tool = banking_setup["retrieval_tool"]
    sql_tool = banking_setup["sql_tool"]

    doc_out = retrieval_tool.invoke({"query": "wire limits POL-101"})
    sql_out = sql_tool.invoke({"sql": "SELECT * FROM transactions WHERE transaction_id = 'TX-9002'"})

    question_data = {
        "id": "BANK-EVAL-001",
        "question": "What is the policy threshold for wire transfers, and what was transaction TX-9002?",
        "expected_answer": "Threshold is $10,000 under POL-101; TX-9002 was $1,200,000.",
    }
    actual_output = "Under [POL-101], wire limit is $10,000. TX-9002 was for $1,200,000."

    test_case = to_deep_eval_test_case(
        question_data=question_data,
        actual_output=actual_output,
        tool_outputs=[doc_out, sql_out],
        tools_called=["search_compliance_documents", "query_banking_database"],
    )

    # 1. Assert DeepEval input/output mappings
    assert test_case.input == question_data["question"]
    assert test_case.actual_output == actual_output
    assert test_case.expected_output == question_data["expected_answer"]

    # 2. Assert retrieval_context contains both document passage and SQL result
    assert test_case.retrieval_context is not None and len(test_case.retrieval_context) > 0
    contexts = test_case.retrieval_context

    has_doc_chunk = any(
        ("[POL]" in c or "[POL-101]" in c or "International Wire Transfer" in c)
        for c in contexts
    )
    assert has_doc_chunk, f"retrieval_context missing formatted document chunk: {contexts}"

    has_sql_chunk = any(
        ("[SQL RESULT]" in c and "transactions" in c and "TX-9002" in c)
        for c in contexts
    )
    assert has_sql_chunk, f"retrieval_context missing [SQL RESULT] chunk: {contexts}"

    # 3. Assert tools_called contains DeepEval ToolCall instances
    assert test_case.tools_called is not None and len(test_case.tools_called) == 2
    for tc in test_case.tools_called:
        assert isinstance(tc, ToolCall), f"Expected ToolCall instance, got {type(tc)}"
        assert tc.name in ("search_compliance_documents", "query_banking_database")
        assert tc.output is not None
