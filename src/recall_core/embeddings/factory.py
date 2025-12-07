"""Factory function to create embedding providers from configuration."""

from recall_core.config.settings import (
    EmbeddingProviderType,
    RecallSettings,
    get_settings,
)
from recall_core.embeddings.base import EmbeddingProvider
from recall_core.embeddings.ollama import OllamaProvider


def create_embedding_provider(
    settings: RecallSettings | None = None,
) -> EmbeddingProvider:
    """Create an embedding provider based on configuration.

    Args:
        settings: Optional settings override. Uses global settings if None.

    Returns:
        Configured EmbeddingProvider instance

    Raises:
        ProviderNotAvailableError: If required package is not installed
        ValueError: If API key is missing for cloud providers

    Example:
        # Use default settings
        provider = create_embedding_provider()

        # Override settings
        from recall_core.config import RecallSettings, EmbeddingProviderType
        settings = RecallSettings(embedding_provider=EmbeddingProviderType.OPENAI)
        provider = create_embedding_provider(settings)
    """
    if settings is None:
        settings = get_settings()

    match settings.embedding_provider:
        case EmbeddingProviderType.OLLAMA:
            return OllamaProvider(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
            )
        case EmbeddingProviderType.ANTHROPIC:
            # Lazy import to avoid requiring voyageai unless needed
            from recall_core.embeddings.anthropic import AnthropicProvider

            if not settings.anthropic_api_key:
                raise ValueError("anthropic_api_key required for Anthropic provider")
            return AnthropicProvider(
                api_key=settings.anthropic_api_key,
            )
        case EmbeddingProviderType.OPENAI:
            # Lazy import to avoid requiring openai unless needed
            from recall_core.embeddings.openai import OpenAIProvider

            if not settings.openai_api_key:
                raise ValueError("openai_api_key required for OpenAI provider")
            return OpenAIProvider(
                api_key=settings.openai_api_key,
            )
