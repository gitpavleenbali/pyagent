# Vector Database

Semantic search and RAG with vector databases.

> **See [[VectorDB-Module]] for full documentation.**

## Quick Start

```python
from pyai.vectordb import ChromaDB

# Create store
db = ChromaDB(collection="knowledge")

# Add documents
db.add("PYAI is a Python SDK for AI agents")
db.add_documents(["doc1.txt", "doc2.pdf"])

# Search
results = db.search("What is PYAI?", top_k=5)
```

## Supported Databases

| Database | Description |
|----------|-------------|
| ChromaDB | Local, embedded vector store |
| Pinecone | Cloud-native, scalable |
| Qdrant | Self-hosted, feature-rich |
| Weaviate | GraphQL-based |

## Features

- Semantic similarity search
- Document ingestion
- Metadata filtering
- Hybrid search
- Multiple embedding models

## Related Pages

- [[VectorDB-Module]] - Full module documentation
- [[ChromaDB]] - ChromaDB integration
- [[Pinecone]] - Pinecone integration
- [[Qdrant]] - Qdrant integration
- [[Weaviate]] - Weaviate integration
- [[rag]] - RAG system
