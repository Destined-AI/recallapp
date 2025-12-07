"""Tests for embedding provider factory."""

import pytest

from recall_core.config.settings import EmbeddingProviderType, RecallSettings, reset_settings
from recall_core.embeddings.factory import create_embedding_provider
from recall_core.embeddings.ollama import OllamaProvider


class TestEmbeddingProviderFactory:
    """Tests for create_embedding_provider."""

    @pytest.fixture(autouse=True)
    def reset(self) -> None:
        """Reset settings before each test."""
        reset_settings()

    def test_create_ollama_provider_default(self) -> None:
        """Test creating Ollama provider with default settings."""
        provider = create_embedding_provider()

        assert isinstance(provider, OllamaProvider)
        assert provider.model_name == "nomic-embed-text"

    def test_create_ollama_provider_custom_settings(self) -> None:
        """Test creating Ollama provider with custom settings."""
        settings = RecallSettings(
            embedding_provider=EmbeddingProviderType.OLLAMA,
            ollama_model="mxbai-embed-large",
            ollama_base_url="http://custom:11434",
        )

        provider = create_embedding_provider(settings)

        assert isinstance(provider, OllamaProvider)
        assert provider.model_name == "mxbai-embed-large"
        assert provider._base_url == "http://custom:11434"

    def test_create_openai_provider_requires_key(self) -> None:
        """Test that OpenAI provider requires API key."""
        settings = RecallSettings(
            embedding_provider=EmbeddingProviderType.OLLAMA,  # Start with valid
        )
        # Manually set to OpenAI without key
        settings.embedding_provider = EmbeddingProviderType.OPENAI

        with pytest.raises(ValueError, match="openai_api_key required"):
            create_embedding_provider(settings)

    def test_create_anthropic_provider_requires_key(self) -> None:
        """Test that Anthropic provider requires API key."""
        settings = RecallSettings(
            embedding_provider=EmbeddingProviderType.OLLAMA,  # Start with valid
        )
        # Manually set to Anthropic without key
        settings.embedding_provider = EmbeddingProviderType.ANTHROPIC

        with pytest.raises(ValueError, match="anthropic_api_key required"):
            create_embedding_provider(settings)
