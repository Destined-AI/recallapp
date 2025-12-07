"""Conversation storage with SQLite metadata index and JSON files."""

import asyncio
from datetime import datetime
from pathlib import Path

import aiosqlite

from recall_core.storage.models import Conversation


class ConversationStore:
    """Store for raw conversation data.

    Uses SQLite for fast metadata queries and JSON files for full
    conversation content. This hybrid approach gives us:
    - Fast indexed queries by project, date, etc.
    - Human-readable conversation files for debugging
    - Atomic writes for reliability

    Example:
        store = ConversationStore(data_dir=Path("~/.recallapp").expanduser())
        await store.save(conversation)
        conv = await store.get(conversation_id)
    """

    def __init__(self, data_dir: Path) -> None:
        """Initialize conversation store.

        Args:
            data_dir: Base directory for RecallApp data
        """
        self._data_dir = data_dir
        self._conversations_dir = data_dir / "conversations"
        self._db_path = data_dir / "conversations.db"
        self._db: aiosqlite.Connection | None = None

    async def _ensure_initialized(self) -> None:
        """Ensure storage directories and database exist."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._conversations_dir.mkdir(exist_ok=True)

        if self._db is None:
            self._db = await aiosqlite.connect(str(self._db_path))
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    project_path TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    indexed_at TEXT,
                    title TEXT,
                    message_count INTEGER DEFAULT 0
                )
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_project_path
                ON conversations(project_path)
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON conversations(created_at)
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_source
                ON conversations(source)
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_indexed_at
                ON conversations(indexed_at)
            """)
            await self._db.commit()

    def _conversation_path(self, id: str) -> Path:
        """Get file path for a conversation.

        Uses first 2 chars of ID for directory sharding to avoid
        too many files in one directory.
        """
        shard = id[:2] if len(id) >= 2 else "00"
        return self._conversations_dir / shard / f"{id}.json"

    async def save(self, conversation: Conversation) -> None:
        """Save a conversation.

        Creates or updates both the JSON file and metadata index.

        Args:
            conversation: Conversation to save
        """
        await self._ensure_initialized()
        assert self._db is not None

        # Save JSON file with atomic write
        path = self._conversation_path(conversation.id)
        path.parent.mkdir(exist_ok=True)

        temp_path = path.with_suffix(".tmp")
        content = conversation.model_dump_json(indent=2)
        await asyncio.to_thread(temp_path.write_text, content)
        await asyncio.to_thread(temp_path.rename, path)

        # Update metadata index
        await self._db.execute(
            """
            INSERT OR REPLACE INTO conversations
            (id, source, project_path, created_at, updated_at, indexed_at, title, message_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                conversation.id,
                conversation.source,
                conversation.project_path,
                conversation.created_at.isoformat(),
                conversation.updated_at.isoformat(),
                conversation.indexed_at.isoformat() if conversation.indexed_at else None,
                conversation.title,
                len(conversation.messages),
            ),
        )
        await self._db.commit()

    async def get(self, id: str) -> Conversation | None:
        """Get a conversation by ID.

        Args:
            id: Conversation ID

        Returns:
            Conversation if found, None otherwise
        """
        await self._ensure_initialized()

        path = self._conversation_path(id)
        if not path.exists():
            return None

        content = await asyncio.to_thread(path.read_text)
        return Conversation.model_validate_json(content)

    async def delete(self, id: str) -> bool:
        """Delete a conversation.

        Args:
            id: Conversation ID

        Returns:
            True if deleted, False if not found
        """
        await self._ensure_initialized()
        assert self._db is not None

        path = self._conversation_path(id)
        existed = path.exists()
        if existed:
            await asyncio.to_thread(path.unlink)

        await self._db.execute("DELETE FROM conversations WHERE id = ?", (id,))
        await self._db.commit()
        return existed

    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Conversation]:
        """List all conversations.

        Args:
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of conversations ordered by updated_at descending
        """
        await self._ensure_initialized()
        assert self._db is not None

        async with self._db.execute(
            """
            SELECT id FROM conversations
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
        """,
            (limit, offset),
        ) as cursor:
            rows = await cursor.fetchall()

        conversations: list[Conversation] = []
        for (id,) in rows:
            conv = await self.get(id)
            if conv:
                conversations.append(conv)
        return conversations

    async def list_by_project(
        self,
        project_path: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Conversation]:
        """List conversations for a project.

        Args:
            project_path: Project path to filter by
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of conversations for the project
        """
        await self._ensure_initialized()
        assert self._db is not None

        async with self._db.execute(
            """
            SELECT id FROM conversations
            WHERE project_path = ?
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
        """,
            (project_path, limit, offset),
        ) as cursor:
            rows = await cursor.fetchall()

        conversations: list[Conversation] = []
        for (id,) in rows:
            conv = await self.get(id)
            if conv:
                conversations.append(conv)
        return conversations

    async def list_by_date_range(
        self,
        start: datetime,
        end: datetime,
        source: str | None = None,
        limit: int = 100,
    ) -> list[Conversation]:
        """List conversations in a date range.

        Args:
            start: Start of date range (inclusive)
            end: End of date range (inclusive)
            source: Optional source filter
            limit: Maximum number of results

        Returns:
            List of conversations in the date range
        """
        await self._ensure_initialized()
        assert self._db is not None

        query = """
            SELECT id FROM conversations
            WHERE created_at >= ? AND created_at <= ?
        """
        params: list[str | int] = [start.isoformat(), end.isoformat()]

        if source:
            query += " AND source = ?"
            params.append(source)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        conversations: list[Conversation] = []
        for (id,) in rows:
            conv = await self.get(id)
            if conv:
                conversations.append(conv)
        return conversations

    async def list_unindexed(self, limit: int = 100) -> list[Conversation]:
        """List conversations that haven't been indexed yet.

        Args:
            limit: Maximum number of results

        Returns:
            List of unindexed conversations (oldest first)
        """
        await self._ensure_initialized()
        assert self._db is not None

        async with self._db.execute(
            """
            SELECT id FROM conversations
            WHERE indexed_at IS NULL
            ORDER BY created_at ASC
            LIMIT ?
        """,
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()

        conversations: list[Conversation] = []
        for (id,) in rows:
            conv = await self.get(id)
            if conv:
                conversations.append(conv)
        return conversations

    async def mark_indexed(self, id: str, indexed_at: datetime | None = None) -> None:
        """Mark a conversation as indexed.

        Args:
            id: Conversation ID
            indexed_at: Timestamp (defaults to now)
        """
        await self._ensure_initialized()
        assert self._db is not None

        now = indexed_at or datetime.utcnow()

        await self._db.execute(
            "UPDATE conversations SET indexed_at = ? WHERE id = ?",
            (now.isoformat(), id),
        )
        await self._db.commit()

        # Also update the JSON file
        conv = await self.get(id)
        if conv:
            conv.indexed_at = now
            await self.save(conv)

    async def get_stats(self) -> dict[str, int]:
        """Get storage statistics.

        Returns:
            Dictionary with stats about stored conversations
        """
        await self._ensure_initialized()
        assert self._db is not None

        async with self._db.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN indexed_at IS NOT NULL THEN 1 END) as indexed,
                COUNT(DISTINCT project_path) as projects
            FROM conversations
        """) as cursor:
            row = await cursor.fetchone()

        assert row is not None
        return {
            "total_conversations": row[0],
            "indexed_conversations": row[1],
            "unique_projects": row[2],
        }

    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None
