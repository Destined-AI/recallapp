"""Vector store backed by LanceDB."""

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import lancedb
import pyarrow as pa

from recall_core.storage.models import Document, DocumentMetadata, SearchResult


class VectorStore:
    """Vector store backed by LanceDB.

    LanceDB is a serverless vector database that stores data in a single
    directory. It's perfect for local development and small-to-medium workloads.

    Example:
        store = VectorStore(path=Path("./data/vectors"), dimension=768)
        await store.add(document, embedding)
        results = await store.search(query_embedding, limit=10)
    """

    TABLE_NAME = "documents"

    def __init__(self, path: Path, dimension: int) -> None:
        """Initialize vector store.

        Args:
            path: Directory path for LanceDB storage
            dimension: Embedding dimension (must match your embedding provider)
        """
        self._path = path
        self._dimension = dimension
        self._db: lancedb.DBConnection | None = None
        self._table: lancedb.table.Table | None = None

    @property
    def dimension(self) -> int:
        """Return the configured embedding dimension."""
        return self._dimension

    async def _ensure_connected(self) -> None:
        """Ensure database connection and table exist."""
        if self._db is None:
            self._path.mkdir(parents=True, exist_ok=True)
            self._db = await asyncio.to_thread(lancedb.connect, str(self._path))

        db = self._db
        assert db is not None  # Ensured by check above
        if self._table is None:
            table_names = await asyncio.to_thread(db.table_names)
            if self.TABLE_NAME in table_names:
                self._table = await asyncio.to_thread(db.open_table, self.TABLE_NAME)
            else:
                # Create table with schema
                schema = pa.schema(
                    [
                        pa.field("id", pa.string()),
                        pa.field("text", pa.string()),
                        pa.field("vector", pa.list_(pa.float32(), self._dimension)),
                        pa.field("source", pa.string()),
                        pa.field("project_path", pa.string()),
                        pa.field("conversation_id", pa.string()),
                        pa.field("chunk_index", pa.int32()),
                        pa.field("created_at", pa.string()),  # ISO format string
                        pa.field("extra", pa.string()),  # JSON string
                    ]
                )
                self._table = await asyncio.to_thread(
                    db.create_table,
                    self.TABLE_NAME,
                    schema=schema,
                )

    async def add(
        self,
        document: Document,
        embedding: list[float],
    ) -> None:
        """Add a document with its embedding.

        If a document with the same ID exists, it will be updated.

        Args:
            document: Document to add
            embedding: Embedding vector for the document
        """
        await self._ensure_connected()
        assert self._table is not None

        data = [
            {
                "id": document.id,
                "text": document.text,
                "vector": embedding,
                "source": document.metadata.source,
                "project_path": document.metadata.project_path or "",
                "conversation_id": document.metadata.conversation_id or "",
                "chunk_index": document.metadata.chunk_index,
                "created_at": document.metadata.created_at.isoformat(),
                "extra": json.dumps(document.metadata.extra),
            }
        ]

        # Use merge_insert for upsert behavior
        await asyncio.to_thread(
            lambda: self._table.merge_insert("id")  # type: ignore
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(data)
        )

    async def add_batch(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[list[float]],
    ) -> None:
        """Add multiple documents with embeddings.

        Args:
            documents: Documents to add
            embeddings: Embedding vectors (must match documents length)
        """
        await self._ensure_connected()
        assert self._table is not None

        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length")

        data = [
            {
                "id": doc.id,
                "text": doc.text,
                "vector": emb,
                "source": doc.metadata.source,
                "project_path": doc.metadata.project_path or "",
                "conversation_id": doc.metadata.conversation_id or "",
                "chunk_index": doc.metadata.chunk_index,
                "created_at": doc.metadata.created_at.isoformat(),
                "extra": json.dumps(doc.metadata.extra),
            }
            for doc, emb in zip(documents, embeddings)
        ]

        await asyncio.to_thread(
            lambda: self._table.merge_insert("id")  # type: ignore
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(data)
        )

    async def search(
        self,
        embedding: list[float],
        limit: int = 10,
        filter_expr: str | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            embedding: Query embedding vector
            limit: Maximum number of results
            filter_expr: Optional LanceDB filter expression
                         (e.g., "source = 'claude_code'")

        Returns:
            List of SearchResult objects sorted by similarity
        """
        await self._ensure_connected()
        assert self._table is not None

        query = self._table.search(embedding).limit(limit)
        if filter_expr:
            query = query.where(filter_expr)

        results = await asyncio.to_thread(query.to_list)

        search_results: list[SearchResult] = []
        for row in results:
            metadata = DocumentMetadata(
                source=row["source"],
                project_path=row["project_path"] or None,
                conversation_id=row["conversation_id"] or None,
                chunk_index=row["chunk_index"],
                created_at=row["created_at"],
                extra=json.loads(row["extra"]) if row["extra"] else {},
            )
            doc = Document(
                id=row["id"],
                text=row["text"],
                metadata=metadata,
            )
            # Convert distance to similarity score (higher is better)
            distance = row.get("_distance", 0.0)
            search_results.append(
                SearchResult(
                    document=doc,
                    score=1.0 / (1.0 + distance),
                    distance=distance,
                )
            )

        return search_results

    async def get(self, id: str) -> Document | None:
        """Get a document by ID.

        Args:
            id: Document ID

        Returns:
            Document if found, None otherwise
        """
        await self._ensure_connected()
        assert self._table is not None

        # Use a filter query to find by ID
        results = await asyncio.to_thread(
            lambda: self._table.search()  # type: ignore
            .where(f"id = '{id}'")
            .limit(1)
            .to_list()
        )

        if not results:
            return None

        row = results[0]
        metadata = DocumentMetadata(
            source=row["source"],
            project_path=row["project_path"] or None,
            conversation_id=row["conversation_id"] or None,
            chunk_index=row["chunk_index"],
            created_at=row["created_at"],
            extra=json.loads(row["extra"]) if row["extra"] else {},
        )
        return Document(id=row["id"], text=row["text"], metadata=metadata)

    async def delete(self, id: str) -> bool:
        """Delete a document by ID.

        Args:
            id: Document ID

        Returns:
            True (LanceDB delete doesn't return status)
        """
        await self._ensure_connected()
        assert self._table is not None

        await asyncio.to_thread(lambda: self._table.delete(f"id = '{id}'"))  # type: ignore
        return True

    async def delete_by_conversation(self, conversation_id: str) -> int:
        """Delete all documents for a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            Number of documents deleted (approximate)
        """
        await self._ensure_connected()
        assert self._table is not None

        # Count before delete
        count_results = await asyncio.to_thread(
            lambda: self._table.search()  # type: ignore
            .where(f"conversation_id = '{conversation_id}'")
            .to_list()
        )
        count = len(count_results)

        await asyncio.to_thread(
            lambda: self._table.delete(f"conversation_id = '{conversation_id}'")  # type: ignore
        )
        return count

    async def count(self) -> int:
        """Get total number of documents.

        Returns:
            Total document count
        """
        await self._ensure_connected()
        assert self._table is not None

        result = await asyncio.to_thread(lambda: len(self._table))  # type: ignore
        return result

    async def close(self) -> None:
        """Close the database connection."""
        # LanceDB doesn't require explicit close
        self._table = None
        self._db = None
