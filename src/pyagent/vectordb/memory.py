# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
In-memory vector store implementation.

Simple vector store for testing and small datasets.
"""

import math
from typing import Any, Callable, Dict, List, Optional

from .base import VectorStore, Document, SearchResult, EmbeddingFunction


class MemoryVectorStore(VectorStore):
    """In-memory vector store.
    
    Useful for testing, prototyping, and small datasets.
    
    Example:
        store = MemoryVectorStore()
        store.add("doc1", "Hello world")
        store.add("doc2", "Goodbye world")
        
        results = store.search("hello", k=1)
        print(results[0].content)  # "Hello world"
    """
    
    def __init__(
        self,
        embedding_function: Optional[EmbeddingFunction] = None,
    ):
        """Initialize memory vector store.
        
        Args:
            embedding_function: Function to compute embeddings
        """
        self._documents: Dict[str, Document] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._embedding_function = embedding_function
    
    def add(
        self,
        id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Add a document to the store."""
        document = Document(
            id=id,
            content=content,
            metadata=metadata or {},
            embedding=embedding,
        )
        
        self._documents[id] = document
        
        # Compute or store embedding
        if embedding:
            self._embeddings[id] = embedding
        elif self._embedding_function:
            emb = self._embedding_function.embed([content])[0]
            self._embeddings[id] = emb
    
    def search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar documents."""
        if not self._documents:
            return []
        
        # If we have embeddings, use vector search
        if self._embeddings and self._embedding_function:
            return self._vector_search(query, k, filter)
        
        # Fall back to text similarity
        return self._text_search(query, k, filter)
    
    def _vector_search(
        self,
        query: str,
        k: int,
        filter: Optional[Dict[str, Any]],
    ) -> List[SearchResult]:
        """Search using vector similarity."""
        query_embedding = self._embedding_function.embed_query(query)
        
        results = []
        for id, doc in self._documents.items():
            # Apply filter
            if filter and not self._matches_filter(doc, filter):
                continue
            
            embedding = self._embeddings.get(id)
            if embedding:
                # Compute cosine similarity
                score = self._cosine_similarity(query_embedding, embedding)
                results.append(SearchResult(
                    document=doc,
                    score=score,
                    distance=1 - score,
                ))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]
    
    def _text_search(
        self,
        query: str,
        k: int,
        filter: Optional[Dict[str, Any]],
    ) -> List[SearchResult]:
        """Search using simple text matching."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        for id, doc in self._documents.items():
            # Apply filter
            if filter and not self._matches_filter(doc, filter):
                continue
            
            content_lower = doc.content.lower()
            content_words = set(content_lower.split())
            
            # Compute simple word overlap score
            overlap = len(query_words & content_words)
            if overlap > 0 or query_lower in content_lower:
                # Boost score if query is substring
                score = overlap / max(len(query_words), 1)
                if query_lower in content_lower:
                    score += 0.5
                
                results.append(SearchResult(
                    document=doc,
                    score=score,
                    distance=1 - score,
                ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]
    
    def _matches_filter(
        self,
        doc: Document,
        filter: Dict[str, Any],
    ) -> bool:
        """Check if document matches metadata filter."""
        for key, value in filter.items():
            if key not in doc.metadata:
                return False
            if doc.metadata[key] != value:
                return False
        return True
    
    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float],
    ) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def delete(self, id: str) -> bool:
        """Delete a document by ID."""
        if id in self._documents:
            del self._documents[id]
            self._embeddings.pop(id, None)
            return True
        return False
    
    def get(self, id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self._documents.get(id)
    
    def count(self) -> int:
        """Get document count."""
        return len(self._documents)
    
    def clear(self) -> None:
        """Clear all documents."""
        self._documents.clear()
        self._embeddings.clear()
    
    def list_ids(self) -> List[str]:
        """List all document IDs."""
        return list(self._documents.keys())
