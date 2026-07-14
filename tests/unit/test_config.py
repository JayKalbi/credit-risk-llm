"""Unit tests for centralized configuration."""

import os
from unittest.mock import patch

import pytest

from src.config import Settings, get_settings, reset_settings


class TestSettings:
    """Test the Settings configuration system."""

    def setup_method(self):
        """Reset singleton between tests."""
        reset_settings()

    def test_default_settings_load(self):
        """Settings should load with defaults when env vars are minimal."""
        settings = get_settings()
        assert settings.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert settings.retrieval.rrf_k == 60
        assert settings.retrieval.final_top_k == 5
        assert settings.ingestion.chunk_size == 512
        assert settings.app.debug is False

    def test_singleton_returns_same_instance(self):
        """get_settings() should return the same instance on repeated calls."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reset_clears_singleton(self):
        """reset_settings() should force a new instance on next call."""
        s1 = get_settings()
        reset_settings()
        s2 = get_settings()
        assert s1 is not s2

    def test_api_key_loaded_from_env(self):
        """API key should be read from GROQ_API_KEY env var."""
        settings = get_settings()
        assert settings.llm.api_key == os.environ.get("GROQ_API_KEY", "")

    def test_validate_raises_without_api_key(self):
        """validate() should raise ValueError when API key is empty."""
        reset_settings()
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False):
            reset_settings()
            settings = Settings()
            # Override api_key via a new LLMConfig
            from src.config import LLMConfig
            settings.llm = LLMConfig()
            if not settings.llm.api_key:
                with pytest.raises(ValueError, match="GROQ_API_KEY"):
                    settings.validate()

    def test_debug_always_false(self):
        """App debug should always default to False for security."""
        settings = get_settings()
        assert settings.app.debug is False

    def test_embedding_config_frozen(self):
        """EmbeddingConfig should be immutable (frozen dataclass)."""
        settings = get_settings()
        with pytest.raises(AttributeError):
            settings.embedding.model_name = "different-model"

    def test_retrieval_config_values(self):
        """RetrievalConfig should have sensible defaults."""
        settings = get_settings()
        assert settings.retrieval.dense_top_k == 15
        assert settings.retrieval.sparse_top_k == 15
        assert settings.retrieval.rerank_candidates == 10
