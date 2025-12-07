"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    All embedding providers must implement this interface to ensure
    consistent behavior across different backends (Ollama, OpenAI, etc.).
    """

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
            ProviderConnectionError: If provider is unreachable
        """
        ...

    @abstractmethod
    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: Sequence of texts to embed

        Returns:
            List of embedding vectors, one per input text

        Raises:
            EmbeddingError: If embedding generation fails
            ProviderConnectionError: If provider is unreachable
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of embeddings produced by this provider.

        This is used to configure vector storage with the correct schema.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model being used."""
        ...

    async def close(self) -> None:
        """Clean up resources.

        Override in subclasses if cleanup is needed (e.g., closing HTTP clients).
        """
        pass

    async def __aenter__(self) -> "EmbeddingProvider":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit - ensures cleanup."""
        await self.close()
