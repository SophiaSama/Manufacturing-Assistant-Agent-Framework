"""Integration tests for manufacturing domain pack parity and Web UI multi-domain switching.

Requirement R3: Manufacturing domain pack parity & UI multi-domain bootstrap:
1. Symlink validation: Iterate every symlink under domains/manufacturing/data/ and assert p.resolve().exists().
2. DocumentIngestionPipeline parity: Run DocumentIngestionPipeline on domains/manufacturing/data/ with
   mfg_config.documents and mfg_config.rbac. Verify valid chunk_ids ({doc_id}:c{i}:can), valid categories
   (OPR, TEC, FA, QCR, WO, SQ, TR), and full rank spectrum (level_rank 1, 2, 3, 4).
3. Web UI multi-domain bootstrap and role switching: Using TestClient(app) from enterprise_agent.ui.server,
   test default/manufacturing (id: 'manufacturing_nga', 4 automotive roles) and switched
   DOMAIN_PACK=domains/banking/pack.yaml (id: 'banking_aml', 4 banking roles).
4. Configuration schema completeness: Verify DomainConfig preserves data_dir and evaluation settings.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from fastapi.testclient import TestClient

from enterprise_agent.config.domain_config import DomainConfig
from enterprise_agent.documents.loader import DocumentIngestionPipeline
from enterprise_agent.ui.server import app

MFG_PACK_PATH = Path("domains/manufacturing/pack.yaml")
BANKING_PACK_PATH = Path("domains/banking/pack.yaml")
_CHUNK_ID_RE = re.compile(r"^[\w-]+:c\d+:can$")


def test_manufacturing_symlinks_integrity():
    """Test 1: Verify all symlinks under domains/manufacturing/data/ resolve to existing files.

    Asserts:
    - Every symlink in the directory tree resolves cleanly.
    - Zero broken symlinks exist.
    - Specific validation of domains/manufacturing/data/eval/ground-truth.md.
    """
    assert MFG_PACK_PATH.exists(), f"Manufacturing pack missing: {MFG_PACK_PATH}"
    config = DomainConfig.from_yaml_file(MFG_PACK_PATH)
    data_dir = Path(config.domain.data_dir).resolve()
    assert data_dir.exists(), f"Manufacturing data directory missing: {data_dir}"

    symlinks: list[Path] = []
    broken_symlinks: list[tuple[Path, Path]] = []

    for path in data_dir.rglob("*"):
        if path.is_symlink():
            symlinks.append(path)
            try:
                target = path.resolve(strict=True)
                if not target.exists():
                    broken_symlinks.append((path, target))
            except (FileNotFoundError, RuntimeError) as e:
                broken_symlinks.append((path, Path(str(e))))

    assert len(symlinks) > 0, "No symlinks discovered under domains/manufacturing/data/"
    assert len(broken_symlinks) == 0, (
        f"Found {len(broken_symlinks)} broken symlinks: "
        + ", ".join(f"{src} -> {tgt}" for src, tgt in broken_symlinks)
    )

    # Explicit check on eval ground truth
    ground_truth_symlink = data_dir / "eval" / "ground-truth.md"
    assert ground_truth_symlink.is_symlink(), "eval/ground-truth.md should be a symlink"
    assert ground_truth_symlink.resolve().exists(), (
        f"eval/ground-truth.md points to non-existent target: {ground_truth_symlink.readlink()}"
    )


def test_manufacturing_document_ingestion_parity():
    """Test 2: DocumentIngestionPipeline parity on manufacturing domain pack.

    Asserts:
    - Run DocumentIngestionPipeline on domains/manufacturing/data/.
    - Chunks count > 0 (all 25 documents ingested).
    - Every chunk has valid chunk_id matching regex ^{doc_id}:c{i}:can$.
    - Required categories present: OPR, TEC, FA, QCR, WO, SQ, TR.
    - Full rank spectrum present: {1, 2, 3, 4}.
    - Every chunk has corpus_source == 'canonical' and populated text.
    """
    config = DomainConfig.from_yaml_file(MFG_PACK_PATH)
    data_dir = Path(config.domain.data_dir).resolve()

    pipeline = DocumentIngestionPipeline(
        doc_config=config.documents,
        rbac_config=config.rbac,
    )
    chunks = pipeline.load_directory(data_dir)

    assert len(chunks) >= 80, f"Expected at least 80 chunks for manufacturing domain, got {len(chunks)}"

    # 1. Ranks 1..4 spectrum
    ranks = {c.metadata.get("level_rank") for c in chunks}
    assert ranks == {1, 2, 3, 4}, f"Expected full rank spectrum {{1, 2, 3, 4}}, got {ranks}"

    # 2. Categories
    categories = {c.metadata.get("category") for c in chunks}
    required_cats = {"OPR", "TEC", "FA", "QCR", "WO", "SQ", "TR"}
    missing_cats = required_cats - categories
    assert not missing_cats, f"Missing expected categories from ingested chunks: {missing_cats}"

    # 3. Chunk IDs and canonical metadata
    for c in chunks:
        cid = c.metadata.get("chunk_id", "")
        assert _CHUNK_ID_RE.match(cid), f"Chunk ID '{cid}' does not match standard pattern"
        assert c.metadata.get("corpus_source") == "canonical"
        assert c.metadata.get("text") == c.page_content
        assert len(c.page_content.strip()) > 0


def test_web_ui_multi_domain_bootstrap_and_role_switching():
    """Test 3: Web UI multi-domain bootstrap and role switching.

    Asserts:
    - Default / manufacturing domain (DOMAIN_PACK unset):
      - /api/domain/bootstrap returns id: 'manufacturing_nga' and manufacturing display name.
      - /api/roles returns 4 automotive roles: operator, technician, engineer, manager.
    - Switched domain (DOMAIN_PACK=domains/banking/pack.yaml):
      - /api/domain/bootstrap returns id: 'banking_aml' and banking display name.
      - /api/roles returns 4 banking roles: teller, compliance_analyst, risk_manager, chief_risk_officer.
    """
    orig_env = os.environ.get("DOMAIN_PACK")

    try:
        # 1. Default Domain (Manufacturing)
        os.environ.pop("DOMAIN_PACK", None)
        client = TestClient(app)

        resp_boot = client.get("/api/domain/bootstrap")
        assert resp_boot.status_code == 200, f"Bootstrap returned {resp_boot.status_code}"
        boot_data = resp_boot.json()
        assert boot_data.get("id") == "manufacturing_nga"
        assert "Apex Automotive" in boot_data.get("display_name", "") or "NGA" in boot_data.get("display_name", "")

        resp_roles = client.get("/api/roles")
        assert resp_roles.status_code == 200
        roles_data = resp_roles.json()
        assert "roles" in roles_data
        mfg_role_ids = [r["id"] for r in roles_data["roles"]]
        assert mfg_role_ids == ["operator", "technician", "engineer", "manager"]

        # Verify access levels for manufacturing roles
        role_map = {r["id"]: r["level"] for r in roles_data["roles"]}
        assert role_map["operator"] == 1
        assert role_map["technician"] == 2
        assert role_map["engineer"] == 3
        assert role_map["manager"] == 4

        # 2. Switched Domain (Banking AML)
        os.environ["DOMAIN_PACK"] = str(BANKING_PACK_PATH)

        resp_boot_bank = client.get("/api/domain/bootstrap")
        assert resp_boot_bank.status_code == 200
        boot_bank_data = resp_boot_bank.json()
        assert boot_bank_data.get("id") == "banking_aml"
        assert "Global Horizon Bank" in boot_bank_data.get("display_name", "")

        resp_roles_bank = client.get("/api/roles")
        assert resp_roles_bank.status_code == 200
        roles_bank_data = resp_roles_bank.json()
        assert "roles" in roles_bank_data
        bank_role_ids = [r["id"] for r in roles_bank_data["roles"]]
        assert bank_role_ids == ["teller", "compliance_analyst", "risk_manager", "chief_risk_officer"]

        bank_role_map = {r["id"]: r["level"] for r in roles_bank_data["roles"]}
        assert bank_role_map["teller"] == 1
        assert bank_role_map["compliance_analyst"] == 2
        assert bank_role_map["risk_manager"] == 3
        assert bank_role_map["chief_risk_officer"] == 4

    finally:
        # Restore environment
        if orig_env is not None:
            os.environ["DOMAIN_PACK"] = orig_env
        else:
            os.environ.pop("DOMAIN_PACK", None)


def test_domain_config_schema_completeness():
    """Test 4: Verify DomainConfig preserves data_dir and evaluation block.

    Asserts:
    - DomainConfig.from_yaml_file preserves data_dir in domain metadata.
    - DomainConfig preserves EvaluationConfig with hard_release_criteria.
    """
    # Manufacturing pack
    mfg_cfg = DomainConfig.from_yaml_file(MFG_PACK_PATH)
    assert mfg_cfg.domain.data_dir == "domains/manufacturing/data"
    assert mfg_cfg.evaluation is not None
    assert mfg_cfg.evaluation.hard_release_criteria.min_overall_pass_rate == 0.85
    assert mfg_cfg.evaluation.system_one_model == "jev-1.13.0"

    # Banking pack
    bank_cfg = DomainConfig.from_yaml_file(BANKING_PACK_PATH)
    assert bank_cfg.domain.data_dir == "domains/banking/data"
    assert bank_cfg.evaluation is not None
    assert bank_cfg.evaluation.hard_release_criteria.min_overall_pass_rate == 0.80
    assert bank_cfg.evaluation.system_one_model == "jev-1.13.0"
