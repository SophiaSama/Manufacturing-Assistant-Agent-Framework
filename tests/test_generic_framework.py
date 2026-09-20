"""Unit tests for the Generic Enterprise Agentic Framework."""

import json
from pathlib import Path

import pytest

from enterprise_agent.config.domain_config import DomainConfig
from enterprise_agent.database.engine import (
    SqliteAdapter,
    SqlValidationError,
    validate_select_only,
)
from enterprise_agent.documents.loader import (
    CsvDocumentParser,
    DocumentIngestionPipeline,
    JsonDocumentParser,
    MarkdownParser,
    TextParser,
)
from enterprise_agent.hitl.policy_gate import (
    evaluate_governance,
)
from enterprise_agent.models.answer_schema import (
    FinalAnswer,
    GovernanceAlert,
    render_final_answer,
)


def test_domain_config_loading():
    """Verify loading domain packs for manufacturing and banking."""
    mfg_path = Path("domains/manufacturing/pack.yaml")
    assert mfg_path.exists()
    mfg_config = DomainConfig.from_yaml_file(mfg_path)
    assert mfg_config.domain.id == "manufacturing_nga"
    assert mfg_config.rbac.get_rank("operator") == 1
    assert mfg_config.rbac.get_rank("manager") == 4
    assert "lines" in mfg_config.database.allowed_tables

    bank_path = Path("domains/banking/pack.yaml")
    assert bank_path.exists()
    bank_config = DomainConfig.from_yaml_file(bank_path)
    assert bank_config.domain.id == "banking_aml"
    assert bank_config.rbac.get_rank("teller") == 1
    assert bank_config.rbac.get_rank("chief_risk_officer") == 4
    assert "accounts" in bank_config.database.allowed_tables


def test_multi_format_document_parsers(tmp_path):
    """Test generic loaders for markdown, json, csv, and text."""
    # 1. Text parser
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Hello plain text world", encoding="utf-8")
    raw_txt = TextParser().parse(txt_file)
    assert "plain text" in raw_txt.content

    # 2. Markdown parser with frontmatter
    md_file = tmp_path / "sample.md"
    md_file.write_text(
        "---\ntitle: Security Notice\ncategory: SEC\ndoc_id: SEC-01\n---\n# Header\nBody content.",
        encoding="utf-8",
    )
    raw_md = MarkdownParser().parse(md_file)
    assert "Body content" in raw_md.content
    assert raw_md.metadata.get("category") == "SEC"
    assert raw_md.metadata.get("doc_id") == "SEC-01"

    # 3. JSON parser
    json_file = tmp_path / "records.json"
    records = [{"id": 1, "name": "Item A"}, {"id": 2, "name": "Item B"}]
    json_file.write_text(json.dumps(records), encoding="utf-8")
    raw_json = JsonDocumentParser().parse(json_file)
    assert "Item A" in raw_json.content
    assert "Record Count: 2" in raw_json.content

    # 4. CSV parser
    csv_file = tmp_path / "table.csv"
    csv_file.write_text("col1,col2\nval1,val2\nval3,val4", encoding="utf-8")
    raw_csv = CsvDocumentParser().parse(csv_file)
    assert "col1: val1" in raw_csv.content
    assert "Table columns: col1, col2" in raw_csv.content


def test_document_ingestion_pipeline(tmp_path):
    """Test recursive ingestion with RBAC tagging."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    pol_file = docs_dir / "POL-101-wire.md"
    pol_file.write_text("# Wire Limits\nStep 1: Verify amount.\nStep 2: Check OFAC.", encoding="utf-8")

    bank_config = DomainConfig.from_yaml_file("domains/banking/pack.yaml")
    pipeline = DocumentIngestionPipeline(
        doc_config=bank_config.documents,
        rbac_config=bank_config.rbac,
    )

    chunks = pipeline.load_directory(docs_dir)
    assert len(chunks) > 0
    assert chunks[0].metadata["doc_id"] == "POL-101"
    assert chunks[0].metadata["level_rank"] == 1


def test_sqlite_adapter_and_security_validation():
    """Test SQLGlot AST security validation against malicious and read-only queries."""
    # Valid SELECT
    statement = validate_select_only("SELECT name, balance FROM accounts WHERE balance > 1000")
    assert statement is not None

    # Forbidden INSERT
    with pytest.raises(SqlValidationError, match="Only SELECT statements are allowed"):
        validate_select_only("INSERT INTO accounts VALUES ('ACC-999', 500)")

    # Forbidden DROP
    with pytest.raises(SqlValidationError, match="Only SELECT statements are allowed"):
        validate_select_only("DROP TABLE accounts")

    # Forbidden Multiple Statements
    with pytest.raises(SqlValidationError, match="Only a single SQL statement is allowed"):
        validate_select_only("SELECT 1; DROP TABLE accounts;")


def test_sqlite_adapter_banking_db():
    """Test schema introspection and query execution on banking.db."""
    db_path = Path("domains/banking/data/banking.db")
    if not db_path.exists():
        pytest.skip("banking.db not yet created")

    adapter = SqliteAdapter(db_path)
    schema = adapter.load_schema()
    assert "customers" in schema
    assert "accounts" in schema
    assert "balance" in schema["accounts"]

    # Valid query
    res = adapter.run_query("SELECT account_id, balance FROM accounts WHERE balance > 100000")
    assert res["row_count"] >= 1
    assert "ACC-101" in [r["account_id"] for r in res["rows"]]

    # Invalid column
    with pytest.raises(SqlValidationError, match="Unknown column `non_existent_col`"):
        adapter.run_query("SELECT non_existent_col FROM accounts")


def test_governance_and_hitl_manufacturing():
    """Test governance rule triggers on manufacturing domain."""
    mfg_config = DomainConfig.from_yaml_file("domains/manufacturing/pack.yaml")

    text = "Class A defect identified in brake system. Line should be stopped."
    rec = "Stop-ship Aurora AU-2025 immediately."
    gov = evaluate_governance(text, rec, mfg_config.governance)

    assert gov.is_critical is True
    assert "class a" in gov.critical_triggers
    assert "brake" in gov.critical_triggers
    assert gov.hitl_required is True
    assert gov.action_verb == "stop-ship"


def test_governance_and_hitl_banking():
    """Test governance rule triggers on banking domain."""
    bank_config = DomainConfig.from_yaml_file("domains/banking/pack.yaml")

    text = "Transaction TX-9002 was sent to a confirmed OFAC match entity."
    rec = "Freeze account ACC-202 and file SAR with FinCEN."
    gov = evaluate_governance(text, rec, bank_config.governance)

    assert gov.is_critical is True
    assert "ofac match" in gov.critical_triggers
    assert gov.hitl_required is True
    assert gov.action_verb in ("freeze account", "file sar")


def test_final_answer_backward_compatibility():
    """Ensure FinalAnswer bridges generic governance with legacy NGA properties."""
    answer = FinalAnswer(
        direct_answer="Safety defect detected.",
        governance=GovernanceAlert(
            is_critical=True,
            critical_triggers=["brake"],
            escalation_tier="L4",
            compliance_codes=["QCR-501"],
        ),
    )
    # Backward compat checks
    assert answer.class_a_alert is True
    assert answer.escalation_level == "L4"
    assert answer.recall_criteria_met == ["QCR-501"]

    # Render check
    rendered = render_final_answer(answer)
    assert "CRITICAL RISK / POLICY ALERT" in rendered
    assert "Triggered: brake" in rendered
