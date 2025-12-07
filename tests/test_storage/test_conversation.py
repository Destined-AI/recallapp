"""Tests for conversation storage."""

from datetime import datetime, timedelta

from recall_core.storage.conversation import ConversationStore
from recall_core.storage.models import Conversation, Message


class TestConversationStore:
    """Tests for ConversationStore."""

    async def test_save_and_get(self, conversation_store: ConversationStore) -> None:
        """Test saving and retrieving a conversation."""
        conv = Conversation(
            id="test-conv-1",
            source="claude_code",
            project_path="/home/user/project",
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
            ],
        )

        await conversation_store.save(conv)
        retrieved = await conversation_store.get("test-conv-1")

        assert retrieved is not None
        assert retrieved.id == "test-conv-1"
        assert retrieved.source == "claude_code"
        assert retrieved.project_path == "/home/user/project"
        assert len(retrieved.messages) == 2
        assert retrieved.messages[0].content == "Hello"

    async def test_get_nonexistent(self, conversation_store: ConversationStore) -> None:
        """Test getting a nonexistent conversation returns None."""
        retrieved = await conversation_store.get("nonexistent")
        assert retrieved is None

    async def test_delete(self, conversation_store: ConversationStore) -> None:
        """Test deleting a conversation."""
        conv = Conversation(id="to-delete", source="test")
        await conversation_store.save(conv)

        existed = await conversation_store.delete("to-delete")
        assert existed is True

        retrieved = await conversation_store.get("to-delete")
        assert retrieved is None

    async def test_delete_nonexistent(self, conversation_store: ConversationStore) -> None:
        """Test deleting a nonexistent conversation returns False."""
        existed = await conversation_store.delete("nonexistent")
        assert existed is False

    async def test_list_all(self, conversation_store: ConversationStore) -> None:
        """Test listing all conversations."""
        for i in range(5):
            conv = Conversation(id=f"list-{i}", source="test")
            await conversation_store.save(conv)

        conversations = await conversation_store.list_all(limit=10)
        assert len(conversations) == 5

    async def test_list_all_with_pagination(self, conversation_store: ConversationStore) -> None:
        """Test pagination in list_all."""
        for i in range(10):
            conv = Conversation(id=f"page-{i}", source="test")
            await conversation_store.save(conv)

        page1 = await conversation_store.list_all(limit=3, offset=0)
        page2 = await conversation_store.list_all(limit=3, offset=3)

        assert len(page1) == 3
        assert len(page2) == 3
        # No overlap
        page1_ids = {c.id for c in page1}
        page2_ids = {c.id for c in page2}
        assert len(page1_ids & page2_ids) == 0

    async def test_list_by_project(self, conversation_store: ConversationStore) -> None:
        """Test listing conversations by project."""
        # Add conversations for different projects
        for i in range(3):
            conv = Conversation(
                id=f"proj-a-{i}",
                source="test",
                project_path="/project-a",
            )
            await conversation_store.save(conv)

        for i in range(2):
            conv = Conversation(
                id=f"proj-b-{i}",
                source="test",
                project_path="/project-b",
            )
            await conversation_store.save(conv)

        proj_a = await conversation_store.list_by_project("/project-a")
        proj_b = await conversation_store.list_by_project("/project-b")

        assert len(proj_a) == 3
        assert len(proj_b) == 2
        assert all(c.project_path == "/project-a" for c in proj_a)

    async def test_list_by_date_range(self, conversation_store: ConversationStore) -> None:
        """Test listing conversations by date range."""
        now = datetime.utcnow()

        # Create conversations at different times
        for i in range(5):
            conv = Conversation(
                id=f"date-{i}",
                source="test",
                created_at=now - timedelta(days=i),
            )
            await conversation_store.save(conv)

        # Query for last 2 days
        start = now - timedelta(days=2)
        end = now + timedelta(hours=1)  # Include today

        results = await conversation_store.list_by_date_range(start, end)

        # Should get conversations from today, yesterday, and 2 days ago
        assert len(results) == 3

    async def test_list_by_date_range_with_source(
        self, conversation_store: ConversationStore
    ) -> None:
        """Test date range filter with source filter."""
        now = datetime.utcnow()

        for source in ["claude_code", "other"]:
            for i in range(3):
                conv = Conversation(
                    id=f"{source}-{i}",
                    source=source,
                    created_at=now - timedelta(days=i),
                )
                await conversation_store.save(conv)

        start = now - timedelta(days=5)
        end = now + timedelta(hours=1)

        results = await conversation_store.list_by_date_range(start, end, source="claude_code")

        assert len(results) == 3
        assert all(c.source == "claude_code" for c in results)

    async def test_list_unindexed(self, conversation_store: ConversationStore) -> None:
        """Test listing unindexed conversations."""
        # Add some indexed and unindexed conversations
        for i in range(3):
            conv = Conversation(
                id=f"indexed-{i}",
                source="test",
                indexed_at=datetime.utcnow(),
            )
            await conversation_store.save(conv)

        for i in range(5):
            conv = Conversation(
                id=f"unindexed-{i}",
                source="test",
                indexed_at=None,
            )
            await conversation_store.save(conv)

        unindexed = await conversation_store.list_unindexed()

        assert len(unindexed) == 5
        assert all(c.indexed_at is None for c in unindexed)

    async def test_mark_indexed(self, conversation_store: ConversationStore) -> None:
        """Test marking a conversation as indexed."""
        conv = Conversation(id="to-index", source="test")
        await conversation_store.save(conv)

        await conversation_store.mark_indexed("to-index")

        retrieved = await conversation_store.get("to-index")
        assert retrieved is not None
        assert retrieved.indexed_at is not None

    async def test_get_stats(self, conversation_store: ConversationStore) -> None:
        """Test getting storage statistics."""
        # Add conversations with different states
        for i in range(5):
            conv = Conversation(
                id=f"stats-indexed-{i}",
                source="test",
                project_path="/project-a",
                indexed_at=datetime.utcnow(),
            )
            await conversation_store.save(conv)

        for i in range(3):
            conv = Conversation(
                id=f"stats-unindexed-{i}",
                source="test",
                project_path="/project-b",
            )
            await conversation_store.save(conv)

        stats = await conversation_store.get_stats()

        assert stats["total_conversations"] == 8
        assert stats["indexed_conversations"] == 5
        assert stats["unique_projects"] == 2

    async def test_update_existing(self, conversation_store: ConversationStore) -> None:
        """Test that saving updates existing conversation."""
        conv = Conversation(
            id="update-test",
            source="test",
            title="Original Title",
        )
        await conversation_store.save(conv)

        conv.title = "Updated Title"
        conv.messages.append(Message(role="user", content="New message"))
        await conversation_store.save(conv)

        retrieved = await conversation_store.get("update-test")
        assert retrieved is not None
        assert retrieved.title == "Updated Title"
        assert len(retrieved.messages) == 1

    async def test_messages_preserved(self, conversation_store: ConversationStore) -> None:
        """Test that message details are preserved."""
        now = datetime.utcnow()
        conv = Conversation(
            id="messages-test",
            source="test",
            messages=[
                Message(
                    role="user",
                    content="Hello",
                    timestamp=now,
                ),
                Message(
                    role="assistant",
                    content="Hi!",
                    timestamp=now,
                    tool_calls=[{"name": "read_file", "args": {"path": "/test"}}],
                ),
            ],
        )
        await conversation_store.save(conv)

        retrieved = await conversation_store.get("messages-test")
        assert retrieved is not None
        assert len(retrieved.messages) == 2
        assert retrieved.messages[1].tool_calls[0]["name"] == "read_file"
