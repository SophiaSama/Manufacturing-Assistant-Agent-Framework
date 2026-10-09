"""Enterprise Agentic Framework — Domain-agnostic decision support."""

from enterprise_agent.config.domain_config import DomainConfig
from enterprise_agent.database.engine import DatabaseAdapter, SqliteAdapter
from enterprise_agent.models.answer_schema import FinalAnswer, GovernanceAlert

__version__ = "0.2.0"

_LAZY_IMPORTS = {
    "DocumentIngestionPipeline": "enterprise_agent.documents.loader",
    "build_enterprise_agent_graph": "enterprise_agent.graph.orchestrator",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        mod = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DatabaseAdapter",
    "DocumentIngestionPipeline",
    "DomainConfig",
    "FinalAnswer",
    "GovernanceAlert",
    "SqliteAdapter",
    "build_enterprise_agent_graph",
]
