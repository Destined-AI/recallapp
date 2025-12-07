"""Custom exceptions for recall-core."""


class RecallError(Exception):
    """Base exception for all RecallApp errors."""

    pass


class ConfigurationError(RecallError):
    """Error in configuration."""

    pass


class EmbeddingError(RecallError):
    """Error generating embeddings."""

    pass


class ProviderConnectionError(EmbeddingError):
    """Cannot connect to embedding provider."""

    pass


class ProviderNotAvailableError(EmbeddingError):
    """Embedding provider not installed or available."""

    pass


class StorageError(RecallError):
    """Error in storage operations."""

    pass


class ConversationNotFoundError(StorageError):
    """Conversation not found."""

    pass


class DocumentNotFoundError(StorageError):
    """Document not found in vector store."""

    pass
