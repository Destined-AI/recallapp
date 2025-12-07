"""OpenAI embedding provider.

Requires the openai package: pip install recall-core[openai]
"""

from collections.abc import Sequence

from recall_core.embeddings.base import EmbeddingProvider
from recall_core.exceptions import EmbeddingError, ProviderNotAvailableError


class OpenAIProvider(EmbeddingProvider):
    """Embedding provider using OpenAI's API.

    Requires the openai package to be installed.

    Example:
        provider = OpenAIProvider(api_key="your-openai-api-key")
        async with provider:
            embedding = await provider.embed("Hello world")
    """

    MODEL_DIMENSIONS: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
    ) -> None:
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model name (default: text-embedding-3-small)

        Raises:
            ProviderNotAvailableError: If openai package is not installed
        """
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ProviderNotAvailableError(
                "openai package not installed. "
                "Install with: pip install recall-core[openai]"
            ) from e

        self._model = model
        self._client: AsyncOpenAI = AsyncOpenAI(api_key=api_key)
        self._dimension = self.MODEL_DIMENSIONS.get(model, 1536)

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
            response = await self._client.embeddings.create(
                model=self._model,
                input=text,
            )
            embedding: list[float] = response.data[0].embedding
            return embedding
        except Exception as e:
            raise EmbeddingError(f"OpenAI error: {e}") from e

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=list(texts),
            )
            embeddings: list[list[float]] = [item.embedding for item in response.data]
            return embeddings
        except Exception as e:
            raise EmbeddingError(f"OpenAI error: {e}") from e

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.close()
