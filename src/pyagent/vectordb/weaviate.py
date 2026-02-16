# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Weaviate vector store implementation.

Requires: pip install weaviate-client
"""

import os
from typing import Any, Dict, List, Optional

from .base import VectorStore, Document, SearchResult, EmbeddingFunction


class WeaviateStore(VectorStore):
    """Weaviate vector store.
    
    Supports cloud and self-hosted Weaviate instances.
    
    Example:
        # Cloud instance
        store = WeaviateStore(
            url="https://xxx.weaviate.network",
            api_key="xxx",
            class_name="Documents"
        )
        
        # Local instance
        store = WeaviateStore(
            url="http://localhost:8080",
            class_name="Documents"
        )
        
        store.add("doc1", "Hello world")
        results = store.search("hello", k=5)
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        class_name: str = "Document",
        embedding_function: Optional[EmbeddingFunction] = None,
        **kwargs
    ):
        """Initialize Weaviate store.
        
        Args:
            url: Weaviate URL (or WEAVIATE_URL env var)
            api_key: Weaviate API key (or WEAVIATE_API_KEY)
            class_name: Weaviate class name
            embedding_function: Function to compute embeddings
            **kwargs: Additional Weaviate settings
        """
        self._url = url or os.environ.get("WEAVIATE_URL", "http://localhost:8080")
        self._api_key = api_key or os.environ.get("WEAVIATE_API_KEY")
        self._class_name = class_name
        self._embedding_function = embedding_function
        self._client = None
        self._kwargs = kwargs
    
    def _get_client(self):
        """Get or create Weaviate client."""
        if self._client is not None:
            return self._client
        
        try:
            import weaviate
            from weaviate.auth import AuthApiKey
        except ImportError:
            raise ImportError(
                "weaviate-client package required. Install with: pip install weaviate-client"
            )
        
        auth_config = None
        if self._api_key:
            auth_config = AuthApiKey(api_key=self._api_key)
        
        self._client = weaviate.Client(
            url=self._url,
            auth_client_secret=auth_config,
            **self._kwargs
        )
        
        # Ensure class exists
        self._ensure_class()
        
        return self._client
    
    def _ensure_class(self):
        """Ensure the class exists in Weaviate."""
        if not self._client.schema.exists(self._class_name):
            class_obj = {
                "class": self._class_name,
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                    },
                    {
                        "name": "doc_id",
                        "dataType": ["text"],
                    },
                    {
                        "name": "metadata_json",
                        "dataType": ["text"],
                    },
                ],
                "vectorizer": "none" if self._embedding_function else "text2vec-openai",
            }
            self._client.schema.create_class(class_obj)
    
    def add(
        self,
        id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Add a document to Weaviate."""
        import json
        
        client = self._get_client()
        
        if embedding is None and self._embedding_function:
            embedding = self._embedding_function.embed([content])[0]
        
        properties = {
            "content": content,
            "doc_id": id,
            "metadata_json": json.dumps(metadata or {}),
        }
        
        # Check if document exists
        existing = self._find_by_doc_id(id)
        if existing:
            # Update
            client.data_object.update(
                data_object=properties,
                class_name=self._class_name,
                uuid=existing,
                vector=embedding,
            )
        else:
            # Create
            client.data_object.create(
                data_object=properties,
                class_name=self._class_name,
                vector=embedding,
            )
    
    def _find_by_doc_id(self, doc_id: str) -> Optional[str]:
        """Find Weaviate UUID by doc_id."""
        client = self._get_client()
        
        result = (
            client.query
            .get(self._class_name, ["doc_id"])
            .with_where({
                "path": ["doc_id"],
                "operator": "Equal",
                "valueText": doc_id,
            })
            .with_additional(["id"])
            .with_limit(1)
            .do()
        )
        
        data = result.get("data", {}).get("Get", {}).get(self._class_name, [])
        if data:
            return data[0]["_additional"]["id"]
        return None
    
    def search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search Weaviate for similar documents."""
        import json
        
        client = self._get_client()
        
        # Get embedding for query
        if self._embedding_function:
            query_embedding = self._embedding_function.embed_query(query)
            query_builder = (
                client.query
                .get(self._class_name, ["content", "doc_id", "metadata_json"])
                .with_near_vector({"vector": query_embedding})
            )
        else:
            # Use Weaviate's built-in vectorizer
            query_builder = (
                client.query
                .get(self._class_name, ["content", "doc_id", "metadata_json"])
                .with_near_text({"concepts": [query]})
            )
        
        query_builder = (
            query_builder
            .with_additional(["certainty", "distance"])
            .with_limit(k)
        )
        
        result = query_builder.do()
        
        search_results = []
        data = result.get("data", {}).get("Get", {}).get(self._class_name, [])
        
        for item in data:
            metadata = {}
            if item.get("metadata_json"):
                try:
                    metadata = json.loads(item["metadata_json"])
                except:
                    pass
            
            additional = item.get("_additional", {})
            certainty = additional.get("certainty", 0.0)
            distance = additional.get("distance", 1 - certainty)
            
            doc = Document(
                id=item.get("doc_id", ""),
                content=item.get("content", ""),
                metadata=metadata,
            )
            
            search_results.append(SearchResult(
                document=doc,
                score=certainty,
                distance=distance,
            ))
        
        return search_results
    
    def delete(self, id: str) -> bool:
        """Delete a document from Weaviate."""
        client = self._get_client()
        
        uuid = self._find_by_doc_id(id)
        if uuid:
            try:
                client.data_object.delete(
                    uuid=uuid,
                    class_name=self._class_name,
                )
                return True
            except:
                pass
        return False
    
    def get(self, id: str) -> Optional[Document]:
        """Get a document by ID."""
        import json
        
        client = self._get_client()
        
        uuid = self._find_by_doc_id(id)
        if uuid:
            result = client.data_object.get_by_id(
                uuid=uuid,
                class_name=self._class_name,
            )
            
            if result:
                properties = result.get("properties", {})
                metadata = {}
                if properties.get("metadata_json"):
                    try:
                        metadata = json.loads(properties["metadata_json"])
                    except:
                        pass
                
                return Document(
                    id=properties.get("doc_id", id),
                    content=properties.get("content", ""),
                    metadata=metadata,
                )
        
        return None
    
    def count(self) -> int:
        """Get document count."""
        client = self._get_client()
        
        result = (
            client.query
            .aggregate(self._class_name)
            .with_meta_count()
            .do()
        )
        
        data = result.get("data", {}).get("Aggregate", {}).get(self._class_name, [])
        if data:
            return data[0].get("meta", {}).get("count", 0)
        return 0
    
    def clear(self) -> None:
        """Clear all documents."""
        client = self._get_client()
        try:
            client.schema.delete_class(self._class_name)
        except:
            pass
        self._ensure_class()
