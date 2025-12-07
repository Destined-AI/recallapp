"""Ollama embedding provider using httpx for async HTTP calls."""

from collections.abc import Sequence

import httpx

from recall_core.embeddings.base import EmbeddingProvider
from recall_core.exceptions import EmbeddingError, ProviderConnectionError


class OllamaProvider(EmbeddingProvider):
    """Embedding provider using Ollama's local API.

    Ollama runs locally and provides embeddings via HTTP API.
    Default model is nomic-embed-text which produces 768-dimensional embeddings.

    Example:
        async with OllamaProvider() as provider:
            embedding = await provider.embed("Hello world")
            print(f"Dimension: {len(embedding)}")
    """

    # Known model dimensions
    MODEL_DIMENSIONS: dict[str, int] = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
        "snowflake-arctic-embed": 1024,
    }
    DEFAULT_DIMENSION = 768

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
    ) -> None:
        """Initialize Ollama provider.

        Args:
            model: Ollama embedding model name
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
        """
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
        )
        self._detected_dimension: int | None = None

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    @property
    def dimension(self) -> int:
        """Return embedding dimension.

        Returns known dimension for common models, or detected dimension
        from first embedding call.
        """
        if self._detected_dimension is not None:
            return self._detected_dimension
        return self.MODEL_DIMENSIONS.get(self._model, self.DEFAULT_DIMENSION)

    async def _detect_dimension(self, embedding: list[float]) -> None:
        """Detect actual dimension from embedding response."""
        if self._detected_dimension is None:
            self._detected_dimension = len(embedding)

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            ProviderConnectionError: If Ollama is not reachable
            EmbeddingError: If API returns an error
        """
        try:
            response = await self._client.post(
                "/api/embeddings",
                json={"model": self._model, "prompt": text},
            )
            response.raise_for_status()
            data = response.json()
            embedding: list[float] = data["embedding"]
            await self._detect_dimension(embedding)
            return embedding
        except httpx.ConnectError as e:
            raise ProviderConnectionError(
                f"Failed to connect to Ollama at {self._base_url}. "
                "Ensure Ollama is running and accessible."
            ) from e
        except httpx.HTTPStatusError as e:
            raise EmbeddingError(f"Ollama API error: {e.response.text}") from e
        except KeyError as e:
            raise EmbeddingError(f"Unexpected Ollama response format: {e}") from e

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Ollama doesn't have a native batch endpoint, so we process sequentially.
        For better performance with many texts, consider using asyncio.gather
        with rate limiting.

        Args:
            texts: Sequence of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings: list[list[float]] = []
        for text in texts:
            embedding = await self.embed(text)
            embeddings.append(embedding)
        return embeddings

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
