# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
pyai Vector Database Connectors

Unified interface for vector databases with support for:
- ChromaDB (local/embedded)
- Pinecone (cloud)
- Weaviate (cloud/self-hosted)
- Qdrant (cloud/self-hosted)
- In-memory (testing)

Example:
    from pyai.vectordb import VectorStore, ChromaStore

    # Create in-memory store
    store = VectorStore()
    store.add("doc1", "Hello world", {"source": "example"})

    # Search
    results = store.search("greeting", k=5)

    # Use ChromaDB
    store = ChromaStore(collection="my_docs")
"""

from .base import (
    Document,
    SearchResult,
    VectorStore,
)
from .chroma import ChromaStore
from .memory import MemoryVectorStore
from .pinecone import PineconeStore
from .qdrant import QdrantStore
from .weaviate import WeaviateStore

__all__ = [
    # Base
    "VectorStore",
    "SearchResult",
    "Document",
    # Implementations
    "MemoryVectorStore",
    "ChromaStore",
    "PineconeStore",
    "WeaviateStore",
    "QdrantStore",
]
