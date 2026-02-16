# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Qdrant vector store implementation.

Requires: pip install qdrant-client
"""

import os
from typing import Any, Dict, List, Optional
import uuid as uuid_lib

from .base import VectorStore, Document, SearchResult, EmbeddingFunction


class QdrantStore(VectorStore):
    """Qdrant vector store.
    
    Supports cloud and self-hosted Qdrant instances.
    
    Example:
        # Cloud instance
        store = QdrantStore(
            url="https://xxx.qdrant.io",
            api_key="xxx",
            collection_name="documents"
        )
        
        # Local instance
        store = QdrantStore(
            url="http://localhost:6333",
            collection_name="documents"
        )
        
        # In-memory
        store = QdrantStore(
            location=":memory:",
            collection_name="documents"
        )
        
        store.add("doc1", "Hello world")
        results = store.search("hello", k=5)
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        location: Optional[str] = None,
        embedding_function: Optional[EmbeddingFunction] = None,
        vector_size: int = 1536,
        **kwargs
    ):
        """Initialize Qdrant store.
        
        Args:
            collection_name: Qdrant collection name
            url: Qdrant URL (or QDRANT_URL env var)
            api_key: Qdrant API key (or QDRANT_API_KEY)
            location: Local path or ":memory:" for in-memory
            embedding_function: Function to compute embeddings
            vector_size: Dimension of embeddings
            **kwargs: Additional Qdrant settings
        """
        self._collection_name = collection_name
        self._url = url or os.environ.get("QDRANT_URL")
        self._api_key = api_key or os.environ.get("QDRANT_API_KEY")
        self._location = location
        self._embedding_function = embedding_function
        self._vector_size = vector_size
        self._client = None
        self._kwargs = kwargs
        
        # ID mapping (Qdrant uses UUID)
        self._id_to_uuid: Dict[str, str] = {}
        self._uuid_to_id: Dict[str, str] = {}
    
    def _get_client(self):
        """Get or create Qdrant client."""
        if self._client is not None:
            return self._client
        
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError(
                "qdrant-client package required. Install with: pip install qdrant-client"
            )
        
        if self._location:
            # Local or in-memory
            self._client = QdrantClient(location=self._location)
        elif self._url:
            # Remote
            self._client = QdrantClient(
                url=self._url,
                api_key=self._api_key,
                **self._kwargs
            )
        else:
            # Default to in-memory
            self._client = QdrantClient(location=":memory:")
        
        # Ensure collection exists
        self._ensure_collection()
        
        return self._client
    
    def _ensure_collection(self):
        """Ensure collection exists."""
        try:
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            return
        
        collections = self._client.get_collections().collections
        exists = any(c.name == self._collection_name for c in collections)
        
        if not exists:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )
    
    def _get_uuid(self, id: str) -> str:
        """Get or create UUID for document ID."""
        if id in self._id_to_uuid:
            return self._id_to_uuid[id]
        
        uuid = str(uuid_lib.uuid4())
        self._id_to_uuid[id] = uuid
        self._uuid_to_id[uuid] = id
        return uuid
    
    def add(
        self,
        id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Add a document to Qdrant."""
        try:
            from qdrant_client.models import PointStruct
        except ImportError:
            raise ImportError("qdrant-client required")
        
        if embedding is None:
            if self._embedding_function is None:
                raise ValueError(
                    "Qdrant requires embeddings. Provide embedding or embedding_function."
                )
            embedding = self._embedding_function.embed([content])[0]
        
        client = self._get_client()
        uuid = self._get_uuid(id)
        
        payload = {
            "content": content,
            "doc_id": id,
            **(metadata or {}),
        }
        
        client.upsert(
            collection_name=self._collection_name,
            points=[
                PointStruct(
                    id=uuid,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add multiple documents efficiently."""
        try:
            from qdrant_client.models import PointStruct
        except ImportError:
            raise ImportError("qdrant-client required")
        
        if not documents:
            return
        
        # Compute embeddings for docs without them
        docs_without_emb = [d for d in documents if d.embedding is None]
        if docs_without_emb and self._embedding_function:
            contents = [d.content for d in docs_without_emb]
            embeddings = self._embedding_function.embed(contents)
            for doc, emb in zip(docs_without_emb, embeddings):
                doc.embedding = emb
        
        client = self._get_client()
        
        points = []
        for doc in documents:
            if doc.embedding is None:
                continue
            
            uuid = self._get_uuid(doc.id)
            payload = {
                "content": doc.content,
                "doc_id": doc.id,
                **doc.metadata,
            }
            
            points.append(PointStruct(
                id=uuid,
                vector=doc.embedding,
                payload=payload,
            ))
        
        if points:
            client.upsert(
                collection_name=self._collection_name,
                points=points,
            )
    
    def search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search Qdrant for similar documents."""
        if self._embedding_function is None:
            raise ValueError("Embedding function required for search")
        
        query_embedding = self._embedding_function.embed_query(query)
        
        client = self._get_client()
        
        # Build filter
        qdrant_filter = None
        if filter:
            try:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                
                conditions = [
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                    for key, value in filter.items()
                ]
                qdrant_filter = Filter(must=conditions)
            except ImportError:
                pass
        
        results = client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            limit=k,
            query_filter=qdrant_filter,
        )
        
        search_results = []
        for hit in results:
            payload = hit.payload or {}
            doc_id = payload.pop("doc_id", str(hit.id))
            content = payload.pop("content", "")
            
            doc = Document(
                id=doc_id,
                content=content,
                metadata=payload,
            )
            
            search_results.append(SearchResult(
                document=doc,
                score=hit.score,
                distance=1 - hit.score,
            ))
        
        return search_results
    
    def delete(self, id: str) -> bool:
        """Delete a document from Qdrant."""
        client = self._get_client()
        
        uuid = self._id_to_uuid.get(id)
        if uuid:
            try:
                client.delete(
                    collection_name=self._collection_name,
                    points_selector=[uuid],
                )
                del self._id_to_uuid[id]
                del self._uuid_to_id[uuid]
                return True
            except:
                pass
        return False
    
    def get(self, id: str) -> Optional[Document]:
        """Get a document by ID."""
        client = self._get_client()
        
        uuid = self._id_to_uuid.get(id)
        if uuid:
            results = client.retrieve(
                collection_name=self._collection_name,
                ids=[uuid],
            )
            
            if results:
                payload = results[0].payload or {}
                doc_id = payload.pop("doc_id", id)
                content = payload.pop("content", "")
                
                return Document(
                    id=doc_id,
                    content=content,
                    metadata=payload,
                    embedding=results[0].vector,
                )
        
        return None
    
    def count(self) -> int:
        """Get document count."""
        client = self._get_client()
        
        info = client.get_collection(self._collection_name)
        return info.points_count
    
    def clear(self) -> None:
        """Clear all documents."""
        client = self._get_client()
        try:
            client.delete_collection(self._collection_name)
        except:
            pass
        
        self._id_to_uuid.clear()
        self._uuid_to_id.clear()
        self._ensure_collection()
