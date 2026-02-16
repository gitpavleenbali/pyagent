# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Pinecone vector store implementation.

Requires: pip install pinecone-client
"""

import os
from typing import Any, Dict, List, Optional

from .base import VectorStore, Document, SearchResult, EmbeddingFunction


class PineconeStore(VectorStore):
    """Pinecone vector store.
    
    Cloud-native vector database with automatic scaling.
    
    Example:
        store = PineconeStore(
            index_name="my-index",
            api_key="xxx",
            environment="us-west1-gcp"
        )
        
        # Add with embedding
        store.add(
            "doc1",
            "Hello world",
            embedding=[0.1, 0.2, ...]
        )
        
        # Search
        results = store.search("hello", k=5)
    """
    
    def __init__(
        self,
        index_name: str,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        embedding_function: Optional[EmbeddingFunction] = None,
        namespace: str = "",
        **kwargs
    ):
        """Initialize Pinecone store.
        
        Args:
            index_name: Pinecone index name
            api_key: Pinecone API key (or PINECONE_API_KEY env var)
            environment: Pinecone environment (or PINECONE_ENVIRONMENT)
            embedding_function: Function to compute embeddings
            namespace: Namespace within the index
            **kwargs: Additional Pinecone settings
        """
        self._index_name = index_name
        self._api_key = api_key or os.environ.get("PINECONE_API_KEY")
        self._environment = environment or os.environ.get("PINECONE_ENVIRONMENT")
        self._embedding_function = embedding_function
        self._namespace = namespace
        self._index = None
        self._kwargs = kwargs
        
        # Text storage (Pinecone only stores vectors)
        self._text_storage: Dict[str, str] = {}
    
    def _get_index(self):
        """Get or create Pinecone index."""
        if self._index is not None:
            return self._index
        
        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError(
                "pinecone-client package required. Install with: pip install pinecone-client"
            )
        
        pc = Pinecone(api_key=self._api_key)
        self._index = pc.Index(self._index_name)
        
        return self._index
    
    def add(
        self,
        id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Add a document to Pinecone."""
        if embedding is None:
            if self._embedding_function is None:
                raise ValueError(
                    "Pinecone requires embeddings. Provide embedding or embedding_function."
                )
            embedding = self._embedding_function.embed([content])[0]
        
        index = self._get_index()
        
        # Store text content (Pinecone doesn't store text)
        self._text_storage[id] = content
        
        # Add content to metadata for retrieval
        meta = metadata.copy() if metadata else {}
        meta["_content"] = content[:1000]  # Store truncated content in metadata
        
        index.upsert(
            vectors=[{
                "id": id,
                "values": embedding,
                "metadata": meta,
            }],
            namespace=self._namespace,
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add multiple documents efficiently."""
        if not documents:
            return
        
        # Compute embeddings for docs without them
        docs_without_emb = [d for d in documents if d.embedding is None]
        if docs_without_emb and self._embedding_function:
            contents = [d.content for d in docs_without_emb]
            embeddings = self._embedding_function.embed(contents)
            for doc, emb in zip(docs_without_emb, embeddings):
                doc.embedding = emb
        
        index = self._get_index()
        
        vectors = []
        for doc in documents:
            if doc.embedding is None:
                continue
            
            self._text_storage[doc.id] = doc.content
            
            meta = doc.metadata.copy()
            meta["_content"] = doc.content[:1000]
            
            vectors.append({
                "id": doc.id,
                "values": doc.embedding,
                "metadata": meta,
            })
        
        if vectors:
            # Batch upsert (Pinecone supports up to 100 vectors per call)
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                index.upsert(vectors=batch, namespace=self._namespace)
    
    def search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search Pinecone for similar documents."""
        if self._embedding_function is None:
            raise ValueError("Embedding function required for search")
        
        query_embedding = self._embedding_function.embed_query(query)
        
        index = self._get_index()
        
        results = index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            filter=filter,
            namespace=self._namespace,
        )
        
        search_results = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            content = self._text_storage.get(
                match["id"],
                metadata.pop("_content", "")
            )
            
            doc = Document(
                id=match["id"],
                content=content,
                metadata=metadata,
            )
            
            search_results.append(SearchResult(
                document=doc,
                score=match.get("score", 0.0),
                distance=1 - match.get("score", 0.0),
            ))
        
        return search_results
    
    def delete(self, id: str) -> bool:
        """Delete a document from Pinecone."""
        index = self._get_index()
        try:
            index.delete(ids=[id], namespace=self._namespace)
            self._text_storage.pop(id, None)
            return True
        except:
            return False
    
    def delete_many(self, ids: List[str]) -> int:
        """Delete multiple documents."""
        index = self._get_index()
        try:
            index.delete(ids=ids, namespace=self._namespace)
            for id in ids:
                self._text_storage.pop(id, None)
            return len(ids)
        except:
            return 0
    
    def get(self, id: str) -> Optional[Document]:
        """Get a document by ID."""
        index = self._get_index()
        
        result = index.fetch(ids=[id], namespace=self._namespace)
        
        vectors = result.get("vectors", {})
        if id in vectors:
            vec_data = vectors[id]
            metadata = vec_data.get("metadata", {})
            content = self._text_storage.get(
                id,
                metadata.pop("_content", "")
            )
            
            return Document(
                id=id,
                content=content,
                metadata=metadata,
                embedding=vec_data.get("values"),
            )
        
        return None
    
    def count(self) -> int:
        """Get approximate document count."""
        index = self._get_index()
        stats = index.describe_index_stats()
        
        if self._namespace:
            namespaces = stats.get("namespaces", {})
            ns_stats = namespaces.get(self._namespace, {})
            return ns_stats.get("vector_count", 0)
        
        return stats.get("total_vector_count", 0)
    
    def clear(self) -> None:
        """Clear all documents in the namespace."""
        index = self._get_index()
        index.delete(delete_all=True, namespace=self._namespace)
        self._text_storage.clear()
