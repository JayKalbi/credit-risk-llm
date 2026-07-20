"""
Centralized Configuration Management for Hybrid RAG Engine.

All system-wide settings are defined here and loaded from environment
variables with sensible defaults. This eliminates hardcoded values
scattered across the codebase and ensures consistency.

Usage:
    from src.config import get_settings
    settings = get_settings()
    model = settings.embedding.model_name
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for embedding and reranking models."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM generation via Groq."""

    model_name: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant")
    )
    judge_model_name: str = field(
        default_factory=lambda: os.getenv("JUDGE_MODEL_NAME", "llama-3.3-70b-versatile")
    )
    temperature: float = 0.0
    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""), repr=False)


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for the hybrid retrieval pipeline."""

    dense_top_k: int = 15
    sparse_top_k: int = 15
    rrf_k: int = 60
    final_top_k: int = 5
    rerank_candidates: int = 10


@dataclass(frozen=True)
class IngestionConfig:
    """Configuration for document ingestion and indexing."""

    chunk_size: int = 512
    chunk_overlap: int = 50
    documents_dir: str = field(default_factory=lambda: str(PROJECT_ROOT / "data" / "documents"))
    chroma_dir: str = field(default_factory=lambda: str(PROJECT_ROOT / "data" / "chroma_db"))
    bm25_path: str = field(default_factory=lambda: str(PROJECT_ROOT / "data" / "bm25_index.pkl"))


@dataclass(frozen=True)
class AppConfig:
    """Configuration for the Flask web application."""

    host: str = field(default_factory=lambda: os.getenv("APP_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("APP_PORT", "5000")))
    debug: bool = False
    max_query_length: int = 2000
    rate_limit_requests: int = 20
    rate_limit_window_seconds: int = 60
    request_timeout_seconds: int = 120
    cors_origins: str = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*"))


@dataclass
class Settings:
    """Root configuration container aggregating all sub-configs."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    app: AppConfig = field(default_factory=AppConfig)
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    golden_dataset_path: str = field(
        default_factory=lambda: str(PROJECT_ROOT / "data" / "golden_dataset.json")
    )

    def validate(self) -> None:
        """Validate critical settings. Call at application startup."""
        if not self.llm.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or export it as an environment variable. "
                "See .env.example for reference."
            )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the singleton Settings instance (created on first call)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings singleton. Used in testing."""
    global _settings
    _settings = None
