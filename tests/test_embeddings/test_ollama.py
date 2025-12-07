"""Tests for Ollama embedding provider."""

import pytest
import respx
from httpx import Response

from recall_core.embeddings.ollama import OllamaProvider
from recall_core.exceptions import EmbeddingError, ProviderConnectionError


class TestOllamaProvider:
    """Tests for OllamaProvider."""

    @respx.mock
    async def test_embed_single_text(self) -> None:
        """Test embedding a single text."""
        # Mock Ollama API response
        mock_embedding = [0.1] * 768
        respx.post("http://localhost:11434/api/embeddings").mock(
            return_value=Response(200, json={"embedding": mock_embedding})
        )

        async with OllamaProvider() as provider:
            embedding = await provider.embed("Hello world")

        assert len(embedding) == 768
        assert embedding == mock_embedding

    @respx.mock
    async def test_embed_batch(self) -> None:
        """Test embedding multiple texts."""
        mock_embedding1 = [0.1] * 768
        mock_embedding2 = [0.2] * 768

        route = respx.post("http://localhost:11434/api/embeddings")
        route.side_effect = [
            Response(200, json={"embedding": mock_embedding1}),
            Response(200, json={"embedding": mock_embedding2}),
        ]

        async with OllamaProvider() as provider:
            embeddings = await provider.embed_batch(["Hello", "World"])

        assert len(embeddings) == 2
        assert embeddings[0] == mock_embedding1
        assert embeddings[1] == mock_embedding2

    def test_dimension_known_model(self) -> None:
        """Test dimension for known models."""
        provider = OllamaProvider(model="nomic-embed-text")
        assert provider.dimension == 768

        provider = OllamaProvider(model="mxbai-embed-large")
        assert provider.dimension == 1024

    def test_dimension_unknown_model(self) -> None:
        """Test dimension falls back for unknown models."""
        provider = OllamaProvider(model="unknown-model")
        assert provider.dimension == 768  # Default

    @respx.mock
    async def test_dimension_detection(self) -> None:
        """Test dimension is detected from first embedding."""
        mock_embedding = [0.1] * 512  # Non-standard dimension
        respx.post("http://localhost:11434/api/embeddings").mock(
            return_value=Response(200, json={"embedding": mock_embedding})
        )

        async with OllamaProvider(model="unknown-model") as provider:
            await provider.embed("test")
            assert provider.dimension == 512

    def test_model_name(self) -> None:
        """Test model name property."""
        provider = OllamaProvider(model="nomic-embed-text")
        assert provider.model_name == "nomic-embed-text"

    @respx.mock
    async def test_connection_error(self) -> None:
        """Test connection error handling."""
        import httpx

        respx.post("http://localhost:11434/api/embeddings").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        async with OllamaProvider() as provider:
            with pytest.raises(ProviderConnectionError):
                await provider.embed("test")

    @respx.mock
    async def test_api_error(self) -> None:
        """Test API error handling."""
        respx.post("http://localhost:11434/api/embeddings").mock(
            return_value=Response(500, text="Internal Server Error")
        )

        async with OllamaProvider() as provider:
            with pytest.raises(EmbeddingError):
                await provider.embed("test")

    def test_custom_base_url(self) -> None:
        """Test custom base URL."""
        provider = OllamaProvider(base_url="http://my-server:11434")
        assert provider._base_url == "http://my-server:11434"

    def test_base_url_strips_trailing_slash(self) -> None:
        """Test that trailing slash is stripped from base URL."""
        provider = OllamaProvider(base_url="http://my-server:11434/")
        assert provider._base_url == "http://my-server:11434"
