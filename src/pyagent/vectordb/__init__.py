# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
PyAgent Vector Database Connectors

Unified interface for vector databases with support for:
- ChromaDB (local/embedded)
- Pinecone (cloud)
- Weaviate (cloud/self-hosted)
- Qdrant (cloud/self-hosted)
- In-memory (testing)

Example:
    from pyagent.vectordb import VectorStore, ChromaStore
    
    # Create in-memory store
    store = VectorStore()
    store.add("doc1", "Hello world", {"source": "example"})
    
    # Search
    results = store.search("greeting", k=5)
    
    # Use ChromaDB
    store = ChromaStore(collection="my_docs")
"""

from .base import (
    VectorStore,
    SearchResult,
    Document,
)

from .memory import MemoryVectorStore

from .chroma import ChromaStore

from .pinecone import PineconeStore

from .weaviate import WeaviateStore

from .qdrant import QdrantStore

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
