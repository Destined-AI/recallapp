"""Type-safe configuration using pydantic-settings."""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class EmbeddingProviderType(str, Enum):
    """Supported embedding providers."""

    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


def _get_config_path() -> Path:
    """Get the configuration file path."""
    return Path.home() / ".config" / "recallapp" / "config.toml"


class RecallSettings(BaseSettings):
    """RecallApp configuration settings.

    Settings are loaded from (in order of precedence):
    1. Environment variables (prefix: RECALL_)
    2. TOML config file (~/.config/recallapp/config.toml)
    3. Default values
    """

    model_config = SettingsConfigDict(
        env_prefix="RECALL_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Embedding configuration
    embedding_provider: EmbeddingProviderType = Field(
        default=EmbeddingProviderType.OLLAMA,
        description="Which embedding provider to use",
    )

    # Ollama settings
    ollama_model: str = Field(
        default="nomic-embed-text",
        description="Ollama model for embeddings",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )

    # API Keys
    anthropic_api_key: str | None = Field(
        default=None,
        description="Voyage AI API key (Anthropic partner for embeddings)",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key",
    )

    # Storage paths
    data_dir: Path = Field(
        default_factory=lambda: Path.home() / ".recallapp",
        description="Directory for RecallApp data storage",
    )
    claude_code_dir: Path = Field(
        default_factory=lambda: Path.home() / ".claude",
        description="Claude Code storage directory",
    )

    @model_validator(mode="after")
    def validate_api_keys(self) -> "RecallSettings":
        """Ensure required API keys are present for selected provider."""
        if (
            self.embedding_provider == EmbeddingProviderType.ANTHROPIC
            and not self.anthropic_api_key
        ):
            raise ValueError("anthropic_api_key required when using Anthropic provider")
        if (
            self.embedding_provider == EmbeddingProviderType.OPENAI
            and not self.openai_api_key
        ):
            raise ValueError("openai_api_key required when using OpenAI provider")
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize settings sources to include TOML file."""
        config_path = _get_config_path()

        # Only include TOML source if file exists
        if config_path.exists():
            return (
                init_settings,
                env_settings,
                TomlConfigSettingsSource(settings_cls, toml_file=config_path),
            )
        return (
            init_settings,
            env_settings,
        )


# Singleton instance
_settings: RecallSettings | None = None


def get_settings(**kwargs: Any) -> RecallSettings:
    """Get or create settings singleton.

    Args:
        **kwargs: Override settings values (useful for testing)

    Returns:
        RecallSettings instance
    """
    global _settings
    if _settings is None or kwargs:
        _settings = RecallSettings(**kwargs)
    return _settings


def reset_settings() -> None:
    """Reset settings singleton (for testing)."""
    global _settings
    _settings = None
