"""Pydantic models for stored data structures."""

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata for a stored document."""

    source: str = Field(description="Origin of the document (e.g., 'claude_code')")
    project_path: str | None = Field(default=None, description="Associated project")
    conversation_id: str | None = Field(default=None, description="Parent conversation")
    chunk_index: int = Field(default=0, description="Index within chunked document")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """A document stored in the vector store."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    metadata: DocumentMetadata
    embedding: list[float] | None = Field(default=None, exclude=True)


class SearchResult(BaseModel):
    """Result from a vector search."""

    document: Document
    score: float = Field(description="Similarity score (higher is better)")
    distance: float = Field(description="Distance from query vector")


class Message(BaseModel):
    """A single message in a conversation."""

    role: str = Field(description="'user' or 'assistant'")
    content: str
    timestamp: datetime | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class Conversation(BaseModel):
    """A full conversation with messages and metadata."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    source: str = Field(default="claude_code", description="Origin of conversation")
    project_path: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    messages: list[Message] = Field(default_factory=list)
    indexed_at: datetime | None = None
    title: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
