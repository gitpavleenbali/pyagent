# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Base classes for vector database connectors.
"""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """A document for storage in a vector database.

    Attributes:
        id: Unique document identifier
        content: Document text content
        metadata: Additional metadata
        embedding: Pre-computed embedding (optional)
    """

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @classmethod
    def create(cls, content: str, id: Optional[str] = None, **metadata) -> "Document":
        """Create a document with auto-generated ID.

        Args:
            content: Document content
            id: Optional ID (generated from content hash if not provided)
            **metadata: Additional metadata

        Returns:
            Document instance
        """
        if id is None:
            id = hashlib.md5(content.encode()).hexdigest()[:12]

        return cls(id=id, content=content, metadata=metadata)


@dataclass
class SearchResult:
    """A search result from vector database.

    Attributes:
        document: The matched document
        score: Similarity score (higher = more similar)
        distance: Distance metric (lower = more similar)
    """

    document: Document
    score: float = 1.0
    distance: float = 0.0

    @property
    def id(self) -> str:
        """Get document ID."""
        return self.document.id

    @property
    def content(self) -> str:
        """Get document content."""
        return self.document.content

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get document metadata."""
        return self.document.metadata


class VectorStore(ABC):
    """Abstract base class for vector stores.

    Provides a unified interface for vector database operations.

    Example:
        class MyVectorStore(VectorStore):
            def add(self, id, content, metadata=None, embedding=None):
                ...

            def search(self, query, k=10, filter=None):
                ...
    """

    @abstractmethod
    def add(
        self,
        id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Add a document to the store.

        Args:
            id: Document ID
            content: Document content
            metadata: Optional metadata
            embedding: Pre-computed embedding
        """
        pass

    def add_document(self, document: Document) -> None:
        """Add a Document object to the store.

        Args:
            document: Document to add
        """
        self.add(
            id=document.id,
            content=document.content,
            metadata=document.metadata,
            embedding=document.embedding,
        )

    def add_documents(self, documents: List[Document]) -> None:
        """Add multiple documents to the store.

        Args:
            documents: List of documents to add
        """
        for doc in documents:
            self.add_document(doc)

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add multiple texts to the store.

        Args:
            texts: List of text contents
            metadatas: List of metadata dicts
            ids: List of IDs (auto-generated if not provided)

        Returns:
            List of document IDs
        """
        if ids is None:
            ids = [hashlib.md5(t.encode()).hexdigest()[:12] for t in texts]

        if metadatas is None:
            metadatas = [{}] * len(texts)

        for id, text, metadata in zip(ids, texts, metadatas):
            self.add(id, text, metadata)

        return ids

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar documents.

        Args:
            query: Query text
            k: Number of results to return
            filter: Metadata filter

        Returns:
            List of search results
        """
        pass

    def similarity_search(self, query: str, k: int = 10, **kwargs) -> List[Document]:
        """Search and return documents (LangChain compatibility).

        Args:
            query: Query text
            k: Number of results
            **kwargs: Additional arguments

        Returns:
            List of documents
        """
        results = self.search(query, k=k, **kwargs)
        return [r.document for r in results]

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete a document by ID.

        Args:
            id: Document ID

        Returns:
            True if deleted, False if not found
        """
        pass

    def delete_many(self, ids: List[str]) -> int:
        """Delete multiple documents.

        Args:
            ids: List of document IDs

        Returns:
            Number of documents deleted
        """
        count = 0
        for id in ids:
            if self.delete(id):
                count += 1
        return count

    @abstractmethod
    def get(self, id: str) -> Optional[Document]:
        """Get a document by ID.

        Args:
            id: Document ID

        Returns:
            Document or None if not found
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Get the number of documents in the store.

        Returns:
            Document count
        """
        pass

    def clear(self) -> None:
        """Clear all documents from the store."""
        raise NotImplementedError("Clear not implemented for this store")

    def __len__(self) -> int:
        return self.count()


class EmbeddingFunction(ABC):
    """Abstract base class for embedding functions."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        pass

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query.

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        return self.embed([query])[0]

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed documents (alias for embed).

        Args:
            documents: List of document texts

        Returns:
            List of embeddings
        """
        return self.embed(documents)


class OpenAIEmbedding(EmbeddingFunction):
    """Embedding function using OpenAI API.

    Example:
        embedder = OpenAIEmbedding(model="text-embedding-3-small")
        embeddings = embedder.embed(["Hello", "World"])
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        self.model = model
        self.api_key = api_key
        self.dimensions = dimensions
        self._client = None

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using OpenAI API."""
        client = self._get_client()

        kwargs = {"model": self.model, "input": texts}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        response = client.embeddings.create(**kwargs)

        return [item.embedding for item in response.data]


class AzureOpenAIEmbedding(EmbeddingFunction):
    """Embedding function using Azure OpenAI.

    Example:
        embedder = AzureOpenAIEmbedding(
            deployment="my-embedding",
            endpoint="https://xxx.openai.azure.com/"
        )
    """

    def __init__(
        self,
        deployment: str,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-01",
        dimensions: Optional[int] = None,
    ):
        self.deployment = deployment
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.dimensions = dimensions
        self._client = None

    def _get_client(self):
        """Get or create Azure OpenAI client."""
        if self._client is None:
            try:
                from openai import AzureOpenAI
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")

            import os

            self._client = AzureOpenAI(
                api_key=self.api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
                api_version=self.api_version,
                azure_endpoint=self.endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
            )
        return self._client

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using Azure OpenAI."""
        client = self._get_client()

        kwargs = {"model": self.deployment, "input": texts}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        response = client.embeddings.create(**kwargs)

        return [item.embedding for item in response.data]
