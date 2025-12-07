"""Embedding providers for recall-core."""

from recall_core.embeddings.base import EmbeddingProvider
from recall_core.embeddings.factory import create_embedding_provider
from recall_core.embeddings.ollama import OllamaProvider

__all__ = [
    "EmbeddingProvider",
    "OllamaProvider",
    "create_embedding_provider",
]
