"""Central configuration loading for the NGA Manufacturing Assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load the project-local .env without overriding real environment variables
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

_VALID_ENVIRONMENTS = {"local": "Local", "cloud": "Cloud"}

_VALID_CORPUS_PROFILES = {"main", "variant", "conflict"}


_VALID_EMBEDDING_STRATEGIES = {"batch", "sequential"}


def _parse_corpus_profile() -> str:
    raw = (os.getenv("CORPUS_PROFILE", "main") or "main").strip().lower()
    if raw not in _VALID_CORPUS_PROFILES:
        raise ValueError(
            f"CORPUS_PROFILE must be one of {sorted(_VALID_CORPUS_PROFILES)}, got: {raw!r}"
        )
    return raw


def _parse_embedding_strategy() -> str:
    raw = (os.getenv("EMBEDDING_STRATEGY", "batch") or "").strip().lower()
    if raw not in _VALID_EMBEDDING_STRATEGIES:
        raise ValueError(
            f"EMBEDDING_STRATEGY must be 'batch' or 'sequential', got: {raw!r}"
        )
    return raw


@dataclass(frozen=True)
class Settings:
    execution_environment: str
    openrouter_api_key: str
    openrouter_model: str
    openrouter_base_url: str
    embedding_model: str
    ollama_base_url: str
    ollama_chat_model: str
    ollama_embedding_model: str
    # NGA-specific paths
    nga_db_path: str
    app_state_db_path: str
    vector_store_dir: str
    documents_dir: str
    graph_store_dir: str
    graph_max_docs: int
    # Corpus profile (main | variant | conflict) — docs/variant-corpus-ingestion-design.md
    corpus_profile: str
    variant_corpus_dir: str
    variant_vector_store_dir: str
    conflict_vector_store_dir: str
    langsmith_tracing_enabled: bool
    # RBAC JWT
    jwt_secret: str
    jwt_dev_mode: bool
    # Tier routing
    rag_tier1_model: str
    rag_tier2_model: str
    rag_tier3_model: str
    rag_tier1_context_window: int
    rag_tier2_context_window: int
    rag_tier3_context_window: int
    rag_tier1_max_hops: int
    rag_tier2_max_hops: int
    rag_tier3_max_hops: int
    # Batch embedding tuning
    embedding_poll_interval: float
    embedding_max_wait: float
    embedding_strategy: str  # "batch" | "sequential"
    # Cache (docs/cache-design.md)
    cache_enabled: bool
    cache_mode: str  # "cold" | "hot" (eval); prod always serves hot when enabled
    cache_db_path: str
    cache_eval_db_path: str
    cache_ttl_emb_s: float
    cache_ttl_retr_s: float
    cache_ttl_sql_s: float
    cache_max_entries: int

    @property
    def provider(self) -> str:
        return "ollama" if self.execution_environment == "Local" else "openrouter"

    @classmethod
    def from_env(cls) -> "Settings":
        raw_env = (os.getenv("EXECUTION_ENVIRONMENT", "Cloud") or "").strip().lower()
        if raw_env not in _VALID_ENVIRONMENTS:
            raise ValueError(
                f"EXECUTION_ENVIRONMENT must be 'Local' or 'Cloud', got: "
                f"{os.getenv('EXECUTION_ENVIRONMENT')!r}"
            )
        environment = _VALID_ENVIRONMENTS[raw_env]

        openrouter_api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_chat_model = os.getenv("OLLAMA_CHAT_MODEL", "")
        ollama_embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "")

        if environment == "Cloud":
            if not openrouter_api_key or openrouter_api_key.lower().startswith("your-"):
                raise ValueError(
                    "OPENROUTER_API_KEY is missing or still a placeholder. "
                    "Set a real key in .env for Cloud mode."
                )
        else:
            missing = [
                name
                for name, value in (
                    ("OLLAMA_CHAT_MODEL", ollama_chat_model),
                    ("OLLAMA_EMBEDDING_MODEL", ollama_embedding_model),
                )
                if not value.strip()
            ]
            if missing:
                raise ValueError(
                    "Local mode requires these variables: " + ", ".join(missing)
                )

        tracing_enabled = (
            False
            if environment == "Local"
            else os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        )

        return cls(
            execution_environment=environment,
            openrouter_api_key=openrouter_api_key,
            openrouter_model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
            openrouter_base_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            embedding_model=os.getenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-4b"),
            ollama_base_url=ollama_base_url,
            ollama_chat_model=ollama_chat_model,
            ollama_embedding_model=ollama_embedding_model,
            nga_db_path=os.getenv("NGA_DB_PATH", "database/nga.db"),
            app_state_db_path=os.getenv("APP_STATE_DB_PATH", "data/app_state.db"),
            vector_store_dir=os.getenv("VECTOR_STORE_DIR", "data/vector_store"),
            documents_dir=os.getenv("DOCUMENTS_DIR", ""),  # auto-discovered if empty
            graph_store_dir=os.getenv("GRAPH_STORE_DIR", "data/graph_store"),
            graph_max_docs=int(os.getenv("GRAPH_MAX_DOCS", "0")),
            corpus_profile=_parse_corpus_profile(),
            variant_corpus_dir=os.getenv("VARIANT_CORPUS_DIR", "variant-corpus"),
            variant_vector_store_dir=os.getenv(
                "VARIANT_VECTOR_STORE_DIR", "data/variant_vector_store"
            ),
            conflict_vector_store_dir=os.getenv(
                "CONFLICT_VECTOR_STORE_DIR", "data/conflict_vector_store"
            ),
            langsmith_tracing_enabled=tracing_enabled,
            jwt_secret=os.getenv("JWT_SECRET", "dev-secret"),
            jwt_dev_mode=os.getenv("JWT_DEV_MODE", "true").lower() == "true",
            rag_tier1_model=os.getenv(
                "RAG_TIER1_MODEL", "anthropic/claude-haiku-4-5"
            ),
            rag_tier2_model=os.getenv(
                "RAG_TIER2_MODEL", "anthropic/claude-sonnet-4-5"
            ),
            rag_tier3_model=os.getenv(
                "RAG_TIER3_MODEL", "anthropic/claude-opus-4-5"
            ),
            rag_tier1_context_window=int(
                os.getenv("RAG_TIER1_CONTEXT_WINDOW", "8000")
            ),
            rag_tier2_context_window=int(
                os.getenv("RAG_TIER2_CONTEXT_WINDOW", "32000")
            ),
            rag_tier3_context_window=int(
                os.getenv("RAG_TIER3_CONTEXT_WINDOW", "200000")
            ),
            rag_tier1_max_hops=int(os.getenv("RAG_TIER1_MAX_HOPS", "1")),
            rag_tier2_max_hops=int(os.getenv("RAG_TIER2_MAX_HOPS", "3")),
            rag_tier3_max_hops=int(os.getenv("RAG_TIER3_MAX_HOPS", "8")),
            embedding_poll_interval=float(
                os.getenv("EMBEDDING_POLL_INTERVAL", "10")
            ),
            embedding_max_wait=float(
                os.getenv("EMBEDDING_MAX_WAIT", "1800")
            ),
            embedding_strategy=_parse_embedding_strategy(),
            cache_enabled=os.getenv("CACHE_ENABLED", "false").lower()
            in ("1", "true", "yes"),
            cache_mode=(os.getenv("CACHE_MODE", "cold") or "cold").strip().lower(),
            cache_db_path=os.getenv("CACHE_DB_PATH", "data/cache/nga_cache.db"),
            cache_eval_db_path=os.getenv(
                "CACHE_EVAL_DB_PATH", "data/cache/eval_cache.db"
            ),
            cache_ttl_emb_s=float(os.getenv("CACHE_TTL_EMB_S", "86400")),
            cache_ttl_retr_s=float(os.getenv("CACHE_TTL_RETR_S", "3600")),
            cache_ttl_sql_s=float(os.getenv("CACHE_TTL_SQL_S", "300")),
            cache_max_entries=int(os.getenv("CACHE_MAX_ENTRIES", "100000")),
        )
