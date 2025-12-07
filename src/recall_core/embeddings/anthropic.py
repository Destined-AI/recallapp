"""Anthropic embedding provider using Voyage AI.

Anthropic recommends Voyage AI for embeddings. This provider requires
the voyageai package: pip install recall-core[anthropic]
"""

from collections.abc import Sequence

from recall_core.embeddings.base import EmbeddingProvider
from recall_core.exceptions import EmbeddingError, ProviderNotAvailableError


class AnthropicProvider(EmbeddingProvider):
    """Embedding provider using Voyage AI (Anthropic's recommended partner).

    Requires the voyageai package to be installed.

    Example:
        provider = AnthropicProvider(api_key="your-voyage-api-key")
        async with provider:
            embedding = await provider.embed("Hello world")
    """

    MODEL_DIMENSIONS: dict[str, int] = {
        "voyage-3": 1024,
        "voyage-3-lite": 512,
        "voyage-code-3": 1024,
        "voyage-finance-2": 1024,
        "voyage-law-2": 1024,
        "voyage-large-2": 1536,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-3",
    ) -> None:
        """Initialize Voyage AI provider.

        Args:
            api_key: Voyage AI API key
            model: Model name (default: voyage-3)

        Raises:
            ProviderNotAvailableError: If voyageai package is not installed
        """
        try:
            import voyageai
        except ImportError as e:
            raise ProviderNotAvailableError(
                "voyageai package not installed. "
                "Install with: pip install recall-core[anthropic]"
            ) from e

        self._model = model
        self._client: voyageai.AsyncClient = voyageai.AsyncClient(api_key=api_key)
        self._dimension = self.MODEL_DIMENSIONS.get(model, 1024)

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        try:
            result = await self._client.embed(
                [text],
                model=self._model,
                input_type="document",
            )
            embedding: list[float] = result.embeddings[0]
            return embedding
        except Exception as e:
            raise EmbeddingError(f"Voyage AI error: {e}") from e

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        try:
            result = await self._client.embed(
                list(texts),
                model=self._model,
                input_type="document",
            )
            embeddings: list[list[float]] = result.embeddings
            return embeddings
        except Exception as e:
            raise EmbeddingError(f"Voyage AI error: {e}") from e

    async def close(self) -> None:
        """Close the client (no-op for Voyage AI)."""
        pass
