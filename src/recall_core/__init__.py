"""RecallApp Core Library - Shared infrastructure for recall-search and recall-guard.

RecallApp is a developer companion for preserving context and protecting stability.
This core library provides:

- Configuration management with TOML + environment variable support
- Embedding providers (Ollama, OpenAI, Anthropic/Voyage AI)
- Vector storage via LanceDB
- Conversation storage with SQLite metadata indexing

Example:
    from recall_core import get_settings, create_embedding_provider, VectorStore

    # Get settings (from config file or env vars)
    settings = get_settings()

    # Create embedding provider based on settings
    async with create_embedding_provider() as provider:
        embedding = await provider.embed("Hello world")

    # Store and search vectors
    store = VectorStore(path=settings.data_dir / "vectors", dimension=provider.dimension)
    await store.add(document, embedding)
    results = await store.search(query_embedding)
"""

from recall_core.__about__ import __version__
from recall_core.config.settings import (
    EmbeddingProviderType,
    RecallSettings,
    get_settings,
    reset_settings,
)
from recall_core.embeddings.base import EmbeddingProvider
from recall_core.embeddings.factory import create_embedding_provider
from recall_core.embeddings.ollama import OllamaProvider
from recall_core.exceptions import (
    ConfigurationError,
    ConversationNotFoundError,
    DocumentNotFoundError,
    EmbeddingError,
    ProviderConnectionError,
    ProviderNotAvailableError,
    RecallError,
    StorageError,
)
from recall_core.storage.conversation import ConversationStore
from recall_core.storage.models import (
    Conversation,
    Document,
    DocumentMetadata,
    Message,
    SearchResult,
)
from recall_core.storage.vector import VectorStore

__all__ = [
    # Version
    "__version__",
    # Config
    "EmbeddingProviderType",
    "RecallSettings",
    "get_settings",
    "reset_settings",
    # Embeddings
    "EmbeddingProvider",
    "OllamaProvider",
    "create_embedding_provider",
    # Storage
    "ConversationStore",
    "VectorStore",
    "Conversation",
    "Document",
    "DocumentMetadata",
    "Message",
    "SearchResult",
    # Exceptions
    "RecallError",
    "ConfigurationError",
    "EmbeddingError",
    "ProviderConnectionError",
    "ProviderNotAvailableError",
    "StorageError",
    "ConversationNotFoundError",
    "DocumentNotFoundError",
]
