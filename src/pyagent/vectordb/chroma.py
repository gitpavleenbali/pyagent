# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
ChromaDB vector store implementation.

Requires: pip install chromadb
"""

from typing import Any, Callable, Dict, List, Optional

from .base import VectorStore, Document, SearchResult, EmbeddingFunction


class ChromaStore(VectorStore):
    """ChromaDB vector store.
    
    Supports both ephemeral (in-memory) and persistent storage.
    
    Example:
        # Ephemeral (in-memory)
        store = ChromaStore(collection="my_docs")
        
        # Persistent
        store = ChromaStore(
            collection="my_docs",
            persist_directory="./chroma_db"
        )
        
        # Add documents
        store.add("doc1", "Hello world", {"source": "example"})
        
        # Search
        results = store.search("hello", k=5)
    """
    
    def __init__(
        self,
        collection: str = "default",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[EmbeddingFunction] = None,
        client: Optional[Any] = None,
        **kwargs
    ):
        """Initialize ChromaDB store.
        
        Args:
            collection: Collection name
            persist_directory: Path for persistent storage
            embedding_function: Custom embedding function
            client: Pre-configured ChromaDB client
            **kwargs: Additional ChromaDB settings
        """
        self._collection_name = collection
        self._persist_directory = persist_directory
        self._custom_embedding = embedding_function
        self._client = client
        self._collection = None
        self._kwargs = kwargs
    
    def _get_client(self):
        """Get or create ChromaDB client."""
        if self._client is not None:
            return self._client
        
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb package required. Install with: pip install chromadb"
            )
        
        if self._persist_directory:
            self._client = chromadb.PersistentClient(
                path=self._persist_directory,
                **self._kwargs
            )
        else:
            self._client = chromadb.Client(**self._kwargs)
        
        return self._client
    
    def _get_collection(self):
        """Get or create collection."""
        if self._collection is not None:
            return self._collection
        
        client = self._get_client()
        
        # Wrap custom embedding function if provided
        embedding_fn = None
        if self._custom_embedding:
            try:
                from chromadb import EmbeddingFunction as ChromaEF
                
                class WrappedEmbedding(ChromaEF):
                    def __init__(inner_self, fn):
                        inner_self._fn = fn
                    
                    def __call__(inner_self, input: List[str]) -> List[List[float]]:
                        return inner_self._fn.embed(input)
                
                embedding_fn = WrappedEmbedding(self._custom_embedding)
            except:
                pass
        
        self._collection = client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=embedding_fn,
        )
        
        return self._collection
    
    def add(
        self,
        id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Add a document to ChromaDB."""
        collection = self._get_collection()
        
        kwargs = {
            "ids": [id],
            "documents": [content],
        }
        
        if metadata:
            kwargs["metadatas"] = [metadata]
        
        if embedding:
            kwargs["embeddings"] = [embedding]
        
        collection.upsert(**kwargs)
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add multiple documents efficiently."""
        if not documents:
            return
        
        collection = self._get_collection()
        
        ids = [doc.id for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        kwargs = {
            "ids": ids,
            "documents": contents,
            "metadatas": metadatas,
        }
        
        embeddings = [doc.embedding for doc in documents if doc.embedding]
        if len(embeddings) == len(documents):
            kwargs["embeddings"] = embeddings
        
        collection.upsert(**kwargs)
    
    def search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search ChromaDB for similar documents."""
        collection = self._get_collection()
        
        kwargs = {
            "query_texts": [query],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        
        if filter:
            kwargs["where"] = filter
        
        results = collection.query(**kwargs)
        
        search_results = []
        
        if results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results["documents"] else [""] * len(ids)
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)
            
            for i, id in enumerate(ids):
                doc = Document(
                    id=id,
                    content=documents[i] if documents else "",
                    metadata=metadatas[i] if metadatas else {},
                )
                
                # Convert distance to similarity score
                distance = distances[i] if distances else 0.0
                score = 1.0 / (1.0 + distance)
                
                search_results.append(SearchResult(
                    document=doc,
                    score=score,
                    distance=distance,
                ))
        
        return search_results
    
    def delete(self, id: str) -> bool:
        """Delete a document from ChromaDB."""
        collection = self._get_collection()
        try:
            collection.delete(ids=[id])
            return True
        except:
            return False
    
    def delete_many(self, ids: List[str]) -> int:
        """Delete multiple documents."""
        collection = self._get_collection()
        try:
            collection.delete(ids=ids)
            return len(ids)
        except:
            return 0
    
    def get(self, id: str) -> Optional[Document]:
        """Get a document by ID."""
        collection = self._get_collection()
        
        result = collection.get(
            ids=[id],
            include=["documents", "metadatas"],
        )
        
        if result["ids"]:
            return Document(
                id=result["ids"][0],
                content=result["documents"][0] if result["documents"] else "",
                metadata=result["metadatas"][0] if result["metadatas"] else {},
            )
        
        return None
    
    def count(self) -> int:
        """Get document count."""
        collection = self._get_collection()
        return collection.count()
    
    def clear(self) -> None:
        """Clear all documents in the collection."""
        client = self._get_client()
        try:
            client.delete_collection(self._collection_name)
        except:
            pass
        self._collection = None
        # Recreate empty collection
        self._get_collection()
