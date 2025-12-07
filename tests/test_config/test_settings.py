"""Tests for configuration settings."""

from pathlib import Path

import pytest

from recall_core.config.settings import (
    EmbeddingProviderType,
    get_settings,
    reset_settings,
)


class TestRecallSettings:
    """Tests for RecallSettings."""

    def test_default_settings(self, reset_config: None) -> None:
        """Test default settings values."""
        settings = get_settings()

        assert settings.embedding_provider == EmbeddingProviderType.OLLAMA
        assert settings.ollama_model == "nomic-embed-text"
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.anthropic_api_key is None
        assert settings.openai_api_key is None
        assert settings.data_dir == Path.home() / ".recallapp"
        assert settings.claude_code_dir == Path.home() / ".claude"

    def test_env_var_override(self, reset_config: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that environment variables override defaults."""
        monkeypatch.setenv("RECALL_OLLAMA_MODEL", "mxbai-embed-large")
        monkeypatch.setenv("RECALL_OLLAMA_BASE_URL", "http://my-server:11434")

        settings = get_settings()

        assert settings.ollama_model == "mxbai-embed-large"
        assert settings.ollama_base_url == "http://my-server:11434"

    def test_embedding_provider_enum(self, reset_config: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test embedding provider enum parsing from env var."""
        monkeypatch.setenv("RECALL_EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("RECALL_OPENAI_API_KEY", "test-key")

        settings = get_settings()

        assert settings.embedding_provider == EmbeddingProviderType.OPENAI

    def test_anthropic_requires_api_key(self, reset_config: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that Anthropic provider requires API key."""
        monkeypatch.setenv("RECALL_EMBEDDING_PROVIDER", "anthropic")

        with pytest.raises(ValueError, match="anthropic_api_key required"):
            get_settings()

    def test_openai_requires_api_key(self, reset_config: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that OpenAI provider requires API key."""
        monkeypatch.setenv("RECALL_EMBEDDING_PROVIDER", "openai")

        with pytest.raises(ValueError, match="openai_api_key required"):
            get_settings()

    def test_ollama_does_not_require_api_key(self, reset_config: None) -> None:
        """Test that Ollama provider works without API key."""
        settings = get_settings()

        assert settings.embedding_provider == EmbeddingProviderType.OLLAMA
        # Should not raise

    def test_data_dir_override(self, reset_config: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test data directory can be overridden."""
        monkeypatch.setenv("RECALL_DATA_DIR", "/custom/path")

        settings = get_settings()

        assert settings.data_dir == Path("/custom/path")

    def test_settings_singleton(self, reset_config: None) -> None:
        """Test that get_settings returns singleton."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_reset_settings(self, reset_config: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that reset_settings clears singleton."""
        settings1 = get_settings()

        monkeypatch.setenv("RECALL_OLLAMA_MODEL", "different-model")
        reset_settings()

        settings2 = get_settings()

        assert settings1 is not settings2
        assert settings2.ollama_model == "different-model"
