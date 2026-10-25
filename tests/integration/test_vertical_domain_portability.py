"""Integration tests for vertical domain pack portability (banking_aml).

Requirement R3: End-to-end vertical domain portability:
1. Multi-format ingestion (.md, .json, .csv) for banking_aml.
2. RBAC clearance rank filtering across user roles.
3. Multi-step LangGraph execution with SQL + retrieval tools.
4. Critical governance trigger detection (OFAC match / asset freeze).
5. HITL decisions_log persistence in SQLite app_state.db.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import langchain_core.tools as lc_tools
from langchain_core.messages import AIMessage, HumanMessage

import enterprise_agent.tools.tool_factory as tf
from enterprise_agent.config.domain_config import DomainConfig
from enterprise_agent.database.engine import SqliteAdapter
from enterprise_agent.documents.loader import DocumentIngestionPipeline
from enterprise_agent.graph.orchestrator import build_enterprise_agent_graph
from enterprise_agent.hitl.policy_gate import evaluate_governance
from enterprise_agent.memory.decision_log import get_decisions
from enterprise_agent.rag.retrieval import make_generic_retrieval_tool

# Ensure LangChain tracing does not attempt external network requests
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Non-invasive patch for make_generic_sql_tool decorator compatibility
_orig_tool = lc_tools.tool


def _safe_tool_decorator(*args, **kwargs):
    if len(args) == 1 and isinstance(args[0], str) and "description" not in kwargs:
        kwargs["description"] = "Read-only SQL database query tool."
    return _orig_tool(*args, **kwargs)


tf.tool = _safe_tool_decorator

BANKING_PACK_PATH = Path("domains/banking/pack.yaml")
_CHUNK_ID_RE = re.compile(r"^[\w-]+:c\d+:can$")


class MockInMemoryVectorStore:
    """In-memory vector store matching Chroma's similarity_search contract."""

    def __init__(self, documents: list[Any]):
        self.documents = list(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Any]:
        max_rank = 999
        allowed_cats: set[str] | None = None

        if filter:
            if "level_rank" in filter and isinstance(filter["level_rank"], dict):
                max_rank = filter["level_rank"].get("$lte", 999)
            elif "$and" in filter and isinstance(filter["$and"], list):
                for cond in filter["$and"]:
                    if "level_rank" in cond and isinstance(cond["level_rank"], dict):
                        max_rank = cond["level_rank"].get("$lte", max_rank)
                    if "category" in cond and isinstance(cond["category"], dict):
                        allowed_cats = set(cond["category"].get("$in", []))

        results = []
        for doc in self.documents:
            rank = doc.metadata.get("level_rank", 1)
            cat = doc.metadata.get("category", "")
            if rank <= max_rank:
                if allowed_cats is None or cat in allowed_cats:
                    results.append(doc)
            if len(results) >= k:
                break
        return results


def test_banking_multi_format_ingestion():
    """Test 1: Ingest domains/banking/data/ (.md, .json, .csv) via DocumentIngestionPipeline.

    Verifies:
    - Ingestion of markdown, JSON, and CSV records.
    - Chunk count > 0.
    - Correct doc_ids (POL-101, AML-201, SAN-301, SAR-401).
    - Preservation of internal record identifiers (AML-201-C01, SAN-1001).
    - Fully populated metadata (chunk_id, corpus_source, text, level_rank, category).
    """
    assert BANKING_PACK_PATH.exists(), f"Banking pack missing at {BANKING_PACK_PATH}"
    config = DomainConfig.from_yaml_file(BANKING_PACK_PATH)
    data_dir = Path(config.domain.data_dir).resolve()
    assert data_dir.exists(), f"Data directory missing at {data_dir}"

    pipeline = DocumentIngestionPipeline(
        doc_config=config.documents,
        rbac_config=config.rbac,
    )
    chunks = pipeline.load_directory(data_dir)

    assert len(chunks) > 0, "Ingestion pipeline returned 0 chunks for banking domain"

    doc_ids = {c.metadata.get("doc_id") for c in chunks}
    assert "POL-101" in doc_ids, f"Expected POL-101 in doc_ids: {doc_ids}"
    assert "AML-201" in doc_ids, f"Expected AML-201 in doc_ids: {doc_ids}"
    assert "SAN-301" in doc_ids, f"Expected SAN-301 in doc_ids: {doc_ids}"
    assert "SAR-401" in doc_ids, f"Expected SAR-401 in doc_ids: {doc_ids}"

    # Verify internal content identifiers from JSON & CSV
    all_text = " ".join(c.page_content for c in chunks)
    assert "AML-201-C01" in all_text, "AML-201-C01 record missing from ingested text"
    assert "SAN-1001" in all_text, "SAN-1001 entity missing from ingested CSV rows"

    # Verify metadata invariants
    for c in chunks:
        cid = c.metadata.get("chunk_id", "")
        assert _CHUNK_ID_RE.match(cid), f"Malformed chunk_id: {cid}"
        assert c.metadata.get("corpus_source") == "canonical"
        assert c.metadata.get("text") == c.page_content
        assert c.metadata.get("level_rank") in (1, 2, 3, 4)
        assert c.metadata.get("category") in ("POL", "AML", "SAN", "SAR")


def test_banking_rbac_rank_filtering():
    """Test 2: RBAC rank filtering in retrieval tool.

    Verifies:
    - user_level=1 (Teller) retrieves only rank 1 documents.
    - user_level=2 (Compliance Analyst) retrieves ranks 1 and 2.
    - user_level=4 (Chief Risk Officer) accesses full spectrum (ranks 1-4).
    """
    config = DomainConfig.from_yaml_file(BANKING_PACK_PATH)
    pipeline = DocumentIngestionPipeline(doc_config=config.documents, rbac_config=config.rbac)
    chunks = pipeline.load_directory(Path(config.domain.data_dir).resolve())
    store = MockInMemoryVectorStore(chunks)

    # 1. Level 1: Teller
    tool_l1 = make_generic_retrieval_tool(store, user_level=1, doc_config=config.documents)
    res_l1 = json.loads(tool_l1.invoke({"query": "wire limits and regulations"}))
    ranks_l1 = {r["metadata"]["level_rank"] for r in res_l1["results"]}
    assert ranks_l1 == {1}, f"Level 1 user accessed unauthorized ranks: {ranks_l1}"
    assert all(r["doc_id"] == "POL-101" for r in res_l1["results"])

    # 2. Level 2: Compliance Analyst
    tool_l2 = make_generic_retrieval_tool(store, user_level=2, doc_config=config.documents)
    res_l2 = json.loads(tool_l2.invoke({"query": "red flags and wire policies"}))
    ranks_l2 = {r["metadata"]["level_rank"] for r in res_l2["results"]}
    assert ranks_l2.issubset({1, 2}), f"Level 2 user accessed unauthorized ranks: {ranks_l2}"
    assert 2 in ranks_l2, "Level 2 user failed to retrieve rank 2 guidelines"

    # 3. Level 4: Chief Risk Officer
    tool_l4 = make_generic_retrieval_tool(store, user_level=4, doc_config=config.documents)
    res_l4 = json.loads(tool_l4.invoke({"query": "FinCEN escalation and sanctions"}))
    ranks_l4 = {r["metadata"]["level_rank"] for r in res_l4["results"]}
    assert 4 in ranks_l4, f"Level 4 user could not access rank 4 docs; retrieved ranks: {ranks_l4}"
    assert max(ranks_l4) == 4

    # 4. Explicit category filter
    res_cat = json.loads(tool_l4.invoke({"query": "OFAC", "categories": ["SAN"]}))
    assert all(r["category"] == "SAN" for r in res_cat["results"])


def test_banking_multistep_langgraph_execution():
    """Test 3: Multi-step LangGraph execution with SQL + retrieval tools.

    Verifies:
    - Graph compilation with domain configuration.
    - Multi-step execution trajectory: SQL tool call -> retrieval tool call -> synthesis.
    - Accurate FinalAnswer construction and state updates.
    """
    config = DomainConfig.from_yaml_file(BANKING_PACK_PATH)
    pipeline = DocumentIngestionPipeline(doc_config=config.documents, rbac_config=config.rbac)
    chunks = pipeline.load_directory(Path(config.domain.data_dir).resolve())
    store = MockInMemoryVectorStore(chunks)

    retrieval_tool = make_generic_retrieval_tool(store, user_level=4, doc_config=config.documents)
    db_path = Path(config.database.connection_uri).resolve()
    adapter = SqliteAdapter(db_path=db_path, allowed_tables=config.database.allowed_tables)
    sql_tool = tf.make_generic_sql_tool(adapter, tool_name=config.database.tool_name)

    class MockMultiStepAgentLLM:
        def __init__(self):
            self.invocations = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            self.invocations += 1
            if self.invocations == 1:
                # Step 1: SQL query
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": config.database.tool_name,
                        "args": {"sql": "SELECT * FROM transactions WHERE transaction_id = 'TX-9002'"},
                        "id": "call_sql_tx9002",
                    }],
                )
            elif self.invocations == 2:
                # Step 2: Document retrieval query
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": config.documents.tool_name,
                        "args": {"query": "wire transfer thresholds POL-101"},
                        "id": "call_retrieval_pol101",
                    }],
                )
            else:
                # Step 3: Synthesis
                return AIMessage(
                    content=json.dumps({
                        "direct_answer": (
                            "Transaction TX-9002 of $1,200,000 to Orion Minerals International exceeds "
                            "the $1,000,000 limit in [POL-101] and matches confirmed sanction entity [SAN-301]."
                        ),
                        "findings": [
                            "Transaction amount: $1,200,000 OUTBOUND",
                            "Counterparty Orion Minerals International has CONFIRMED_MATCH under BURMA sanctions",
                        ],
                        "evidence": [
                            {"source_type": "document", "citation": "POL-101", "supports": ["Threshold $1,000,000"]},
                            {"source_type": "sql", "citation": "transactions:TX-9002", "supports": ["Amount $1,200,000"]},
                        ],
                        "recommendation": "Freeze account and block wire execution immediately.",
                    })
                )

    mock_llm = MockMultiStepAgentLLM()
    graph = build_enterprise_agent_graph(
        domain_config=config,
        chat_model=mock_llm,
        tools=[sql_tool, retrieval_tool],
        enable_hitl=True,
        approver=lambda rec, **kw: ("approved", "CRO Risk Authorization"),
    )

    state = {
        "messages": [HumanMessage(content="Investigate transaction TX-9002 and apply wire policy.")],
        "user_role": "chief_risk_officer",
    }
    result = graph.invoke(state)

    assert mock_llm.invocations == 3, f"Expected 3 model steps, got {mock_llm.invocations}"
    final_answer = result.get("final_answer")
    assert final_answer is not None, "final_answer missing from result state"
    assert "TX-9002" in final_answer.direct_answer
    assert len(final_answer.findings) >= 2
    assert final_answer.recommendation == "Freeze account and block wire execution immediately."
    assert result.get("approval_status") == "approved"
    assert result.get("approval_note") == "CRO Risk Authorization"


def test_banking_governance_trigger_detection():
    """Test 4: Critical governance trigger detection and evaluation.

    Verifies:
    - OFAC match / asset freeze triggers is_critical and hitl_required.
    - Correct extraction of critical terms and action verbs.
    - Benign inquiries do not trigger governance escalation.
    """
    config = DomainConfig.from_yaml_file(BANKING_PACK_PATH)
    gov_cfg = config.governance

    # 1. Critical OFAC scenario
    critical_text = "OFAC match detected against designated entity in Cyprus."
    rec_freeze = "Freeze account and notify FinCEN immediately."
    gov_critical = evaluate_governance(critical_text, rec_freeze, gov_cfg)

    assert gov_critical.is_critical is True, "Expected is_critical=True for OFAC match"
    assert gov_critical.hitl_required is True, "Expected hitl_required=True for freeze action"
    assert gov_critical.action_verb == "freeze account"
    assert "ofac match" in gov_critical.critical_triggers

    # 2. Benign non-critical scenario
    benign_text = "Customer checking account inquiry for standard direct deposit."
    rec_benign = "Provide routing number and balance details."
    gov_benign = evaluate_governance(benign_text, rec_benign, gov_cfg)

    assert gov_benign.is_critical is False
    assert gov_benign.hitl_required is False
    assert gov_benign.action_verb is None
    assert len(gov_benign.critical_triggers) == 0


def test_banking_hitl_decisions_log_persistence():
    """Test 5: HITL decisions_log persistence in SQLite app_state.db.

    Verifies:
    - In non-interactive mode with mock approver, decision is recorded in decisions_log table.
    - Retrievable via get_decisions(db_path) with accurate status, approver note, and metadata.
    - Validates both approval and rejection workflows.
    """
    config = DomainConfig.from_yaml_file(BANKING_PACK_PATH)

    class FastMockLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            return AIMessage(
                content=json.dumps({
                    "direct_answer": "OFAC match verified for account ACC-202. Transfer must be stopped.",
                    "findings": ["High risk rating", "Confirmed sanctions match"],
                    "evidence": [],
                    "recommendation": "Freeze account and block wire execution.",
                })
            )

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db_path = f.name

    try:
        # 1. Test Approval Path
        graph_approved = build_enterprise_agent_graph(
            domain_config=config,
            chat_model=FastMockLLM(),
            tools=[],
            enable_hitl=True,
            approver=lambda rec, **kw: ("approved", "Authorized by CRO Jane Smith"),
            app_state_db_path=temp_db_path,
        )

        state_1 = {
            "messages": [HumanMessage(content="Action required on sanctions alert for ACC-202")],
            "user_role": "chief_risk_officer",
            "thread_id": "thread-cro-001",
        }
        res_1 = graph_approved.invoke(state_1)
        assert res_1.get("approval_status") == "approved"

        decisions_1 = get_decisions(temp_db_path)
        assert len(decisions_1) == 1
        d1 = decisions_1[0]
        assert d1["category"] == "banking_aml"
        assert d1["user_role"] == "chief_risk_officer"
        assert d1["status"] == "approved"
        assert d1["approver"] == "Authorized by CRO Jane Smith"
        assert d1["class_a_alert"] == 1
        assert "Freeze account" in d1["recommendation"]

        # 2. Test Rejection Path
        graph_rejected = build_enterprise_agent_graph(
            domain_config=config,
            chat_model=FastMockLLM(),
            tools=[],
            enable_hitl=True,
            approver=lambda rec, **kw: ("rejected", "Insufficient evidence from compliance analyst"),
            app_state_db_path=temp_db_path,
        )

        state_2 = {
            "messages": [HumanMessage(content="Second escalation attempt for ACC-202")],
            "user_role": "risk_manager",
            "thread_id": "thread-rm-002",
        }
        res_2 = graph_rejected.invoke(state_2)
        assert res_2.get("approval_status") == "rejected"

        decisions_2 = get_decisions(temp_db_path)
        assert len(decisions_2) == 2
        # get_decisions orders by id DESC: index 0 is newest (rejected), index 1 is oldest (approved)
        d_latest = decisions_2[0]
        assert d_latest["id"] == 2
        assert d_latest["status"] == "rejected"
        assert d_latest["approver"] == "Insufficient evidence from compliance analyst"
        assert d_latest["user_role"] == "risk_manager"

        d_prev = decisions_2[1]
        assert d_prev["id"] == 1
        assert d_prev["status"] == "approved"

    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
