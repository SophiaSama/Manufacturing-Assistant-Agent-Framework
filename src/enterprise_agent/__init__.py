"""Enterprise Agentic Framework — Domain-agnostic decision support."""

from enterprise_agent.config.domain_config import DomainConfig
from enterprise_agent.database.engine import DatabaseAdapter, SqliteAdapter
from enterprise_agent.documents.loader import DocumentIngestionPipeline
from enterprise_agent.graph.orchestrator import build_enterprise_agent_graph
from enterprise_agent.models.answer_schema import FinalAnswer, GovernanceAlert

__version__ = "0.2.0"

__all__ = [
    "DatabaseAdapter",
    "DocumentIngestionPipeline",
    "DomainConfig",
    "FinalAnswer",
    "GovernanceAlert",
    "SqliteAdapter",
    "build_enterprise_agent_graph",
]
