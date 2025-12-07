"""Tests for vector storage."""

from datetime import datetime

from recall_core.storage.models import Document, DocumentMetadata
from recall_core.storage.vector import VectorStore


class TestVectorStore:
    """Tests for VectorStore."""

    async def test_add_and_get_document(self, vector_store: VectorStore) -> None:
        """Test adding and retrieving a document."""
        doc = Document(
            id="test-1",
            text="Hello world",
            metadata=DocumentMetadata(source="test"),
        )
        embedding = [0.1] * 768

        await vector_store.add(doc, embedding)
        retrieved = await vector_store.get("test-1")

        assert retrieved is not None
        assert retrieved.id == "test-1"
        assert retrieved.text == "Hello world"
        assert retrieved.metadata.source == "test"

    async def test_add_batch(self, vector_store: VectorStore) -> None:
        """Test adding multiple documents at once."""
        docs = [
            Document(id=f"batch-{i}", text=f"Text {i}", metadata=DocumentMetadata(source="test"))
            for i in range(3)
        ]
        embeddings = [[0.1 * i] * 768 for i in range(3)]

        await vector_store.add_batch(docs, embeddings)

        for i in range(3):
            retrieved = await vector_store.get(f"batch-{i}")
            assert retrieved is not None
            assert retrieved.text == f"Text {i}"

    async def test_search(self, vector_store: VectorStore) -> None:
        """Test vector similarity search."""
        # Add some documents with different embeddings
        for i in range(5):
            doc = Document(
                id=f"search-{i}",
                text=f"Document {i}",
                metadata=DocumentMetadata(source="test"),
            )
            # Create embeddings that are more similar to query for lower indices
            embedding = [0.1 + (0.1 * i)] * 768
            await vector_store.add(doc, embedding)

        # Search with embedding similar to first document
        query_embedding = [0.1] * 768
        results = await vector_store.search(query_embedding, limit=3)

        assert len(results) == 3
        # Results should be ordered by similarity
        assert results[0].document.id == "search-0"
        assert results[0].score > results[1].score

    async def test_search_with_filter(self, vector_store: VectorStore) -> None:
        """Test search with metadata filter."""
        # Add documents with different sources
        for source in ["source-a", "source-b"]:
            for i in range(3):
                doc = Document(
                    id=f"{source}-{i}",
                    text=f"Text from {source}",
                    metadata=DocumentMetadata(source=source),
                )
                await vector_store.add(doc, [0.1] * 768)

        # Search only in source-a
        results = await vector_store.search(
            [0.1] * 768,
            limit=10,
            filter_expr="source = 'source-a'",
        )

        assert len(results) == 3
        assert all(r.document.metadata.source == "source-a" for r in results)

    async def test_upsert_updates_existing(self, vector_store: VectorStore) -> None:
        """Test that adding a document with same ID updates it."""
        doc1 = Document(
            id="upsert-test",
            text="Original text",
            metadata=DocumentMetadata(source="test"),
        )
        await vector_store.add(doc1, [0.1] * 768)

        doc2 = Document(
            id="upsert-test",
            text="Updated text",
            metadata=DocumentMetadata(source="test"),
        )
        await vector_store.add(doc2, [0.2] * 768)

        retrieved = await vector_store.get("upsert-test")
        assert retrieved is not None
        assert retrieved.text == "Updated text"

    async def test_delete_document(self, vector_store: VectorStore) -> None:
        """Test deleting a document."""
        doc = Document(
            id="delete-test",
            text="To be deleted",
            metadata=DocumentMetadata(source="test"),
        )
        await vector_store.add(doc, [0.1] * 768)

        await vector_store.delete("delete-test")

        retrieved = await vector_store.get("delete-test")
        assert retrieved is None

    async def test_delete_by_conversation(self, vector_store: VectorStore) -> None:
        """Test deleting all documents for a conversation."""
        # Add multiple documents for same conversation
        for i in range(5):
            doc = Document(
                id=f"conv-doc-{i}",
                text=f"Chunk {i}",
                metadata=DocumentMetadata(source="test", conversation_id="conv-123"),
            )
            await vector_store.add(doc, [0.1] * 768)

        # Add one from different conversation
        other_doc = Document(
            id="other-conv",
            text="Other conversation",
            metadata=DocumentMetadata(source="test", conversation_id="conv-456"),
        )
        await vector_store.add(other_doc, [0.1] * 768)

        deleted = await vector_store.delete_by_conversation("conv-123")

        assert deleted == 5
        assert await vector_store.get("conv-doc-0") is None
        assert await vector_store.get("other-conv") is not None

    async def test_get_nonexistent(self, vector_store: VectorStore) -> None:
        """Test getting a nonexistent document returns None."""
        retrieved = await vector_store.get("nonexistent")
        assert retrieved is None

    async def test_count(self, vector_store: VectorStore) -> None:
        """Test counting documents."""
        for i in range(10):
            doc = Document(
                id=f"count-{i}",
                text=f"Text {i}",
                metadata=DocumentMetadata(source="test"),
            )
            await vector_store.add(doc, [0.1] * 768)

        count = await vector_store.count()
        assert count == 10

    async def test_metadata_preserved(self, vector_store: VectorStore) -> None:
        """Test that all metadata fields are preserved."""
        now = datetime.utcnow()
        doc = Document(
            id="metadata-test",
            text="Test",
            metadata=DocumentMetadata(
                source="claude_code",
                project_path="/home/user/project",
                conversation_id="conv-abc",
                chunk_index=5,
                created_at=now,
                extra={"custom": "value"},
            ),
        )
        await vector_store.add(doc, [0.1] * 768)

        retrieved = await vector_store.get("metadata-test")
        assert retrieved is not None
        assert retrieved.metadata.source == "claude_code"
        assert retrieved.metadata.project_path == "/home/user/project"
        assert retrieved.metadata.conversation_id == "conv-abc"
        assert retrieved.metadata.chunk_index == 5
        assert retrieved.metadata.extra == {"custom": "value"}
