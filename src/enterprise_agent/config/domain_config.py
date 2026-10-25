"""Domain Configuration Schema for the Enterprise Agentic Framework.

Allows defining any knowledge domain (Manufacturing, Banking, Healthcare, etc.)
declaratively via YAML or Python dictionary.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DomainMetadata(BaseModel):
    """Metadata identifying the domain."""
    id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    data_dir: str = ""



class RoleDefinition(BaseModel):
    """Definition of an individual RBAC role."""
    name: str
    rank: int  # 1 = lowest clearance, higher = greater clearance
    display_name: str = ""
    description: str = ""


class PathLevelRule(BaseModel):
    """Mapping a folder path pattern to a minimum clearance rank."""
    pattern: str  # e.g. "operator-sops", "internal-audit/"
    min_rank: int


class RbacConfig(BaseModel):
    """Role-Based Access Control configuration."""
    roles: dict[str, int] = Field(
        default_factory=lambda: {"user": 1, "admin": 2},
        description="Map role name to clearance rank",
    )
    role_prompts: dict[str, str] = Field(
        default_factory=dict,
        description="Custom system prompts for each role",
    )
    path_level_rules: list[PathLevelRule] = Field(
        default_factory=list,
        description="Path-based clearance level derivations",
    )
    default_role: str = "user"

    def get_rank(self, role: str) -> int:
        return self.roles.get(role, self.roles.get(self.default_role, 1))

    def level_from_path(self, path: str | Path) -> int:
        p = Path(path)
        path_str = str(p).replace("\\", "/")
        try:
            resolved_str = str(p.resolve()).replace("\\", "/")
        except Exception:
            resolved_str = path_str

        matched_rank = 1
        for rule in self.path_level_rules:
            if rule.pattern in path_str or rule.pattern in resolved_str:
                matched_rank = max(matched_rank, rule.min_rank)
        return matched_rank


class DatabaseConfig(BaseModel):
    """Configuration for relational or SQL databases."""
    dialect: str = "sqlite"  # sqlite, postgres, mysql, duckdb
    connection_uri: str = ""  # File path or connection URI
    allowed_tables: list[str] = Field(default_factory=list)
    tool_name: str = "query_database"
    tool_description: str | None = None
    read_only: bool = True


class DocumentCategory(BaseModel):
    """Document category tag and description."""
    code: str  # e.g. "SOP", "AML", "CPG"
    name: str
    description: str = ""
    folder_patterns: list[str] = Field(default_factory=list)


class DocumentConfig(BaseModel):
    """Configuration for document ingestion and retrieval."""
    supported_extensions: list[str] = Field(
        default_factory=lambda: [".md", ".txt", ".json", ".csv", ".pdf"]
    )
    categories: list[DocumentCategory] = Field(default_factory=list)
    id_patterns: list[str] = Field(
        default_factory=lambda: [r"([A-Z]{2,}-\d+)"],
        description="Regex patterns to extract document identifiers from filename/title",
    )
    excluded_files: list[str] = Field(
        default_factory=lambda: ["ground-truth.md", "README.md"]
    )
    tool_name: str = "search_documents"
    tool_description: str | None = None

    def get_category_for_path(self, path: str | Path) -> str:
        p = Path(path)
        path_str = str(p).replace("\\", "/")
        try:
            resolved_str = str(p.resolve()).replace("\\", "/")
        except Exception:
            resolved_str = path_str

        for cat in self.categories:
            for pattern in cat.folder_patterns:
                if pattern in path_str or pattern in resolved_str:
                    return cat.code
        return self.categories[0].code if self.categories else "DOC"

    def extract_doc_id(self, path: str | Path) -> str:
        stem = Path(path).stem
        for pat in self.id_patterns:
            m = re.search(pat, stem, re.IGNORECASE)
            if m:
                return m.group(1).upper()
        return stem


class KnowledgeGraphConfig(BaseModel):
    """Configuration for GraphRAG extraction and ontology."""
    entity_types: list[str] = Field(default_factory=list)
    relation_types: list[str] = Field(default_factory=list)
    extraction_prompt_template: str | None = None


class GovernanceConfig(BaseModel):
    """Governance, compliance, and Human-In-The-Loop safety configuration."""
    hitl_action_verbs: list[str] = Field(
        default_factory=list,
        description="Action verbs that trigger mandatory human review",
    )
    critical_terms: list[str] = Field(
        default_factory=list,
        description="Keywords indicating critical safety or high-risk event",
    )
    escalation_levels: list[str] = Field(
        default_factory=lambda: ["L1", "L2", "L3", "L4"]
    )
    mandatory_approver_roles: list[str] = Field(default_factory=list)
    justification_required: bool = True


class HardReleaseCriteria(BaseModel):
    """Hard release criteria for domain evaluations."""
    min_overall_pass_rate: float = 0.85
    min_graph_uplift_l3_l4_pct: float = 10.0
    max_allowed_regressions: int = 0
    min_class_a_safety_acc: float = 1.00
    max_avg_latency_s: float = 12.0
    require_sql_sanitization: bool = True


class EvaluationConfig(BaseModel):
    """Evaluation, benchmarking, and gating configuration."""
    questions_path: str = ""
    ground_truth_path: str = ""
    system_one_model: str = "jev-1.13.0"
    eval_mode: str = "system_one"
    pass_threshold: float = 0.70
    hard_release_criteria: HardReleaseCriteria = Field(default_factory=HardReleaseCriteria)


class DomainConfig(BaseModel):
    """Complete domain configuration specification."""
    domain: DomainMetadata
    rbac: RbacConfig = Field(default_factory=RbacConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    documents: DocumentConfig = Field(default_factory=DocumentConfig)
    knowledge_graph: KnowledgeGraphConfig = Field(default_factory=KnowledgeGraphConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DomainConfig":
        return cls.model_validate(data)

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> "DomainConfig":
        # Supports reading YAML file or simple JSON
        content = Path(path).read_text(encoding="utf-8")
        try:
            import yaml
            raw = yaml.safe_load(content)
        except ImportError:
            import json
            raw = json.loads(content)
        return cls.from_dict(raw)
