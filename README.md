# RecallApp

**A developer companion for preserving context and protecting stability**

RecallApp solves two fundamental problems developers face when working with AI coding assistants:

1. **Lost Context** — Every conversation with Claude Code contains valuable decisions, explanations, and problem-solving approaches. But these conversations disappear into local storage, unsearchable and forgotten. RecallApp captures, indexes, and makes this knowledge retrievable.

2. **Breaking Changes** — As codebases grow, it becomes impossible to remember which parts are stable and battle-tested vs. actively under development. A small change can cascade into broken functionality. RecallApp watches your stable contracts and warns you before damage is done.

## Installation

```bash
pip install recall-core
```

For additional embedding providers:

```bash
# OpenAI embeddings
pip install recall-core[openai]

# Anthropic/Voyage AI embeddings
pip install recall-core[anthropic]

# All providers
pip install recall-core[all]
```

## Quick Start

### Configuration

RecallApp uses a TOML configuration file at `~/.config/recallapp/config.toml`:

```toml
# Embedding provider: ollama (default), openai, or anthropic
embedding_provider = "ollama"

# Ollama settings (default provider)
ollama_model = "nomic-embed-text"
ollama_base_url = "http://localhost:11434"

# API keys (required for cloud providers)
# openai_api_key = "sk-..."
# anthropic_api_key = "..."

# Storage paths
data_dir = "~/.recallapp"
claude_code_dir = "~/.claude"
```

Or use environment variables (prefix: `RECALL_`):

```bash
export RECALL_OLLAMA_BASE_URL="http://my-server:11434"
export RECALL_EMBEDDING_PROVIDER="openai"
export RECALL_OPENAI_API_KEY="sk-..."
```

### Using the Library

```python
from recall_core import (
    get_settings,
    create_embedding_provider,
    VectorStore,
    ConversationStore,
    Document,
    DocumentMetadata,
)

# Get settings
settings = get_settings()

# Create embedding provider
async with create_embedding_provider() as provider:
    # Generate embeddings
    embedding = await provider.embed("Hello world")
    print(f"Dimension: {provider.dimension}")

    # Create vector store
    store = VectorStore(
        path=settings.data_dir / "vectors",
        dimension=provider.dimension,
    )

    # Add a document
    doc = Document(
        text="This is searchable content",
        metadata=DocumentMetadata(
            source="claude_code",
            project_path="/home/user/myproject",
        ),
    )
    await store.add(doc, embedding)

    # Search for similar documents
    query_embedding = await provider.embed("search query")
    results = await store.search(query_embedding, limit=5)

    for result in results:
        print(f"{result.score:.3f}: {result.document.text}")
```

### Conversation Storage

```python
from recall_core import ConversationStore, Conversation, Message

store = ConversationStore(data_dir=settings.data_dir)

# Save a conversation
conv = Conversation(
    source="claude_code",
    project_path="/home/user/project",
    messages=[
        Message(role="user", content="How do I fix this bug?"),
        Message(role="assistant", content="Let me help you..."),
    ],
)
await store.save(conv)

# Query conversations
conversations = await store.list_by_project("/home/user/project")
unindexed = await store.list_unindexed()
```

## Architecture

```
recallapp/
├── recall-core/              # Shared Python library (this package)
│   ├── config/               # Settings management
│   ├── embeddings/           # Provider abstraction (Ollama, OpenAI, Anthropic)
│   └── storage/              # Vector store (LanceDB) + conversation store
│
├── recall-search/            # Chat history indexing service (coming soon)
├── recall-guard/             # Breaking changes detector (coming soon)
└── recall-vscode/            # VS Code extension (coming soon)
```

## Embedding Providers

| Provider | Model | Dimension | Notes |
|----------|-------|-----------|-------|
| Ollama | nomic-embed-text | 768 | Local, privacy-first (default) |
| Ollama | mxbai-embed-large | 1024 | Higher quality, local |
| OpenAI | text-embedding-3-small | 1536 | Cloud, fast |
| OpenAI | text-embedding-3-large | 3072 | Cloud, highest quality |
| Voyage AI | voyage-3 | 1024 | Anthropic's recommended |

## Development

```bash
# Clone the repo
git clone https://github.com/Destined-AI/recallapp
cd recallapp

# Install hatch
pip install hatch

# Create development environment
hatch env create

# Run tests
hatch run test

# Run linting
hatch run lint

# Run type checking
hatch run typecheck
```

## License

MIT License - see LICENSE file for details.

---

*RecallApp: Never lose context. Never break what works.*
