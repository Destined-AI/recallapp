"""Pytest fixtures for recall-core tests."""

from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest

from recall_core.config.settings import reset_settings
from recall_core.storage.conversation import ConversationStore
from recall_core.storage.vector import VectorStore


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Provide a temporary data directory."""
    data_dir = tmp_path / "recallapp"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def reset_config() -> Generator[None, None, None]:
    """Reset settings singleton before and after test."""
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
async def vector_store(temp_data_dir: Path) -> AsyncGenerator[VectorStore, None]:
    """Provide a temporary vector store."""
    store = VectorStore(path=temp_data_dir / "vectors", dimension=768)
    yield store
    await store.close()


@pytest.fixture
async def conversation_store(temp_data_dir: Path) -> AsyncGenerator[ConversationStore, None]:
    """Provide a temporary conversation store."""
    store = ConversationStore(data_dir=temp_data_dir)
    yield store
    await store.close()
