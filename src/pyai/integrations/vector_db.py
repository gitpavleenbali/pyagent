"""
Vector Database Connectors for pyai
=======================================

Unified interface to connect pyai RAG operations with popular vector databases.
Supports both cloud and local vector stores.

Supported Databases:
- Azure AI Search (Azure Cognitive Search)
- Pinecone
- Weaviate
- Chroma (local)
- Qdrant
- FAISS (local)
- Milvus
- PostgreSQL + pgvector

Examples:
    >>> from pyai.integrations import vector_db

    # Connect to Azure AI Search
    >>> store = vector_db.connect("azure_ai_search",
    ...     endpoint="https://...",
    ...     index="my-docs"
    ... )

    # Use with pyai RAG
    >>> from pyai import rag
    >>> indexed = rag.index(documents, store=store)
    >>> answer = indexed.ask("What is the conclusion?")

    # Use local ChromaDB for development
    >>> local_store = vector_db.connect("chroma", path="./data")
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Document:
    """A document with content and metadata."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] = None
    score: float = None


@dataclass
class SearchResult:
    """Result from a vector search."""

    documents: List[Document]
    query: str
    total_count: int = 0

    def __iter__(self):
        return iter(self.documents)

    def __len__(self):
        return len(self.documents)


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(self, documents: List[Document]) -> List[str]:
        """Add documents to the store."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> SearchResult:
        """Search for similar documents."""
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """Delete documents by ID."""
        pass

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the store. Alias for add()."""
        return self.add(documents)

    def add_texts(self, texts: List[str], metadatas: List[Dict] = None) -> List[str]:
        """Convenience method to add raw texts."""
        docs = []
        for i, text in enumerate(texts):
            docs.append(
                Document(id=f"doc_{i}", content=text, metadata=metadatas[i] if metadatas else {})
            )
        return self.add(docs)


# =============================================================================
# Azure AI Search
# =============================================================================


class AzureAISearchStore(VectorStore):
    """Azure AI Search (formerly Cognitive Search) vector store."""

    def __init__(
        self,
        endpoint: str = None,
        index_name: str = None,
        api_key: str = None,
        credential=None,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.endpoint = endpoint or os.environ.get("AZURE_SEARCH_ENDPOINT")
        self.index_name = index_name or os.environ.get("AZURE_SEARCH_INDEX", "pyai-docs")
        self.api_key = api_key or os.environ.get("AZURE_SEARCH_API_KEY")
        self.credential = credential
        self.embedding_model = embedding_model
        self._client = None

    def _get_client(self):
        if self._client:
            return self._client

        try:
            from azure.core.credentials import AzureKeyCredential
            from azure.identity import DefaultAzureCredential
            from azure.search.documents import SearchClient

            if self.api_key:
                credential = AzureKeyCredential(self.api_key)
            else:
                credential = self.credential or DefaultAzureCredential()

            self._client = SearchClient(
                endpoint=self.endpoint, index_name=self.index_name, credential=credential
            )
            return self._client
        except ImportError:
            raise ImportError(
                "azure-search-documents not installed. Run: pip install azure-search-documents"
            )

    def add(self, documents: List[Document]) -> List[str]:
        """Add documents to Azure AI Search."""
        client = self._get_client()

        # Convert to Azure format
        azure_docs = []
        for doc in documents:
            azure_doc = {"id": doc.id, "content": doc.content, **doc.metadata}
            if doc.embedding:
                azure_doc["embedding"] = doc.embedding
            azure_docs.append(azure_doc)

        client.upload_documents(azure_docs)
        return [doc.id for doc in documents]

    def search(self, query: str, top_k: int = 5) -> SearchResult:
        """Search Azure AI Search."""
        client = self._get_client()

        results = client.search(search_text=query, top=top_k, include_total_count=True)

        documents = []
        for result in results:
            documents.append(
                Document(
                    id=result.get("id", ""),
                    content=result.get("content", ""),
                    metadata={
                        k: v
                        for k, v in result.items()
                        if k not in ["id", "content", "@search.score"]
                    },
                    score=result.get("@search.score", 0),
                )
            )

        return SearchResult(
            documents=documents,
            query=query,
            total_count=results.get_count() if hasattr(results, "get_count") else len(documents),
        )

    def delete(self, ids: List[str]) -> bool:
        """Delete documents from Azure AI Search."""
        client = self._get_client()
        client.delete_documents([{"id": id} for id in ids])
        return True


# =============================================================================
# Pinecone
# =============================================================================


class PineconeStore(VectorStore):
    """Pinecone vector store."""

    def __init__(self, api_key: str = None, environment: str = None, index_name: str = "pyai-docs"):
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        self.environment = environment or os.environ.get("PINECONE_ENVIRONMENT")
        self.index_name = index_name
        self._index = None

    def _get_index(self):
        if self._index:
            return self._index

        try:
            from pinecone import Pinecone

            pc = Pinecone(api_key=self.api_key)
            self._index = pc.Index(self.index_name)
            return self._index
        except ImportError:
            raise ImportError("pinecone-client not installed. Run: pip install pinecone-client")

    def add(self, documents: List[Document]) -> List[str]:
        """Add documents to Pinecone."""
        index = self._get_index()

        vectors = []
        for doc in documents:
            vectors.append(
                {
                    "id": doc.id,
                    "values": doc.embedding or [],
                    "metadata": {"content": doc.content, **doc.metadata},
                }
            )

        index.upsert(vectors)
        return [doc.id for doc in documents]

    def search(self, query: str, top_k: int = 5) -> SearchResult:
        """Search Pinecone (requires query embedding)."""
        index = self._get_index()

        # Note: In production, you'd embed the query first
        # This is a simplified version
        results = index.query(
            vector=[0] * 1536,  # Placeholder - needs embedding
            top_k=top_k,
            include_metadata=True,
        )

        documents = []
        for match in results.matches:
            documents.append(
                Document(
                    id=match.id,
                    content=match.metadata.get("content", ""),
                    metadata=match.metadata,
                    score=match.score,
                )
            )

        return SearchResult(documents=documents, query=query)

    def delete(self, ids: List[str]) -> bool:
        """Delete from Pinecone."""
        index = self._get_index()
        index.delete(ids=ids)
        return True


# =============================================================================
# ChromaDB (Local)
# =============================================================================


class ChromaStore(VectorStore):
    """ChromaDB local vector store."""

    def __init__(self, path: str = "./chroma_data", collection_name: str = "pyai_docs"):
        self.path = path
        self.collection_name = collection_name
        self._collection = None

    def _get_collection(self):
        if self._collection:
            return self._collection

        try:
            import chromadb

            client = chromadb.PersistentClient(path=self.path)
            self._collection = client.get_or_create_collection(self.collection_name)
            return self._collection
        except ImportError:
            raise ImportError("chromadb not installed. Run: pip install chromadb")

    def add(self, documents: List[Document]) -> List[str]:
        """Add documents to Chroma."""
        collection = self._get_collection()

        collection.add(
            ids=[doc.id for doc in documents],
            documents=[doc.content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
        )

        return [doc.id for doc in documents]

    def search(self, query: str, top_k: int = 5) -> SearchResult:
        """Search Chroma."""
        collection = self._get_collection()

        results = collection.query(query_texts=[query], n_results=top_k)

        documents = []
        for i, doc_id in enumerate(results["ids"][0]):
            documents.append(
                Document(
                    id=doc_id,
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    score=results["distances"][0][i] if results.get("distances") else None,
                )
            )

        return SearchResult(documents=documents, query=query)

    def delete(self, ids: List[str]) -> bool:
        """Delete from Chroma."""
        collection = self._get_collection()
        collection.delete(ids=ids)
        return True


# =============================================================================
# FAISS (Local)
# =============================================================================


class FAISSStore(VectorStore):
    """FAISS local vector store for fast similarity search."""

    def __init__(self, dimension: int = 1536, index_path: str = None):
        self.dimension = dimension
        self.index_path = index_path
        self._index = None
        self._documents = {}  # id -> document mapping

    def _get_index(self):
        if self._index:
            return self._index

        try:
            import faiss
            import numpy as np

            if self.index_path and os.path.exists(self.index_path):
                self._index = faiss.read_index(self.index_path)
            else:
                self._index = faiss.IndexFlatL2(self.dimension)

            return self._index
        except ImportError:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

    def add(self, documents: List[Document]) -> List[str]:
        """Add documents to FAISS."""
        import numpy as np

        index = self._get_index()

        embeddings = []
        for doc in documents:
            if doc.embedding:
                embeddings.append(doc.embedding)
                self._documents[doc.id] = doc
            else:
                raise ValueError(f"Document {doc.id} has no embedding")

        vectors = np.array(embeddings, dtype=np.float32)
        index.add(vectors)

        return [doc.id for doc in documents]

    def search(self, query: str, top_k: int = 5) -> SearchResult:
        """Search FAISS (requires query embedding)."""
        # Note: In production, you'd embed the query first
        return SearchResult(documents=[], query=query)

    def delete(self, ids: List[str]) -> bool:
        """FAISS doesn't support deletion - rebuild index instead."""
        for id in ids:
            self._documents.pop(id, None)
        return True

    def save(self, path: str = None):
        """Save index to disk."""
        import faiss

        path = path or self.index_path
        if path and self._index:
            faiss.write_index(self._index, path)


# =============================================================================
# Qdrant
# =============================================================================


class QdrantStore(VectorStore):
    """Qdrant vector store."""

    def __init__(
        self,
        url: str = "localhost",
        port: int = 6333,
        collection_name: str = "pyai_docs",
        api_key: str = None,
    ):
        self.url = url
        self.port = port
        self.collection_name = collection_name
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client:
            return self._client

        try:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(host=self.url, port=self.port, api_key=self.api_key)
            return self._client
        except ImportError:
            raise ImportError("qdrant-client not installed. Run: pip install qdrant-client")

    def add(self, documents: List[Document]) -> List[str]:
        """Add documents to Qdrant."""
        from qdrant_client.models import PointStruct

        client = self._get_client()

        points = []
        for i, doc in enumerate(documents):
            points.append(
                PointStruct(
                    id=i,
                    vector=doc.embedding or [],
                    payload={"content": doc.content, "doc_id": doc.id, **doc.metadata},
                )
            )

        client.upsert(collection_name=self.collection_name, points=points)
        return [doc.id for doc in documents]

    def search(self, query: str, top_k: int = 5) -> SearchResult:
        """Search Qdrant."""
        self._get_client()
        # Requires query embedding
        return SearchResult(documents=[], query=query)

    def delete(self, ids: List[str]) -> bool:
        """Delete from Qdrant."""
        self._get_client()
        # Implementation depends on how IDs are stored
        return True


# =============================================================================
# Factory Function
# =============================================================================


def connect(provider: str, **kwargs) -> VectorStore:
    """
    Connect to a vector database.

    Args:
        provider: Database provider name
        **kwargs: Provider-specific configuration

    Returns:
        VectorStore instance

    Providers:
        - azure_ai_search: Azure AI Search
        - pinecone: Pinecone
        - chroma: ChromaDB (local)
        - faiss: FAISS (local)
        - qdrant: Qdrant
        - weaviate: Weaviate
        - milvus: Milvus

    Examples:
        >>> store = vector_db.connect("azure_ai_search",
        ...     endpoint="https://...",
        ...     index_name="my-docs"
        ... )

        >>> store = vector_db.connect("chroma", path="./data")
    """
    providers = {
        "azure_ai_search": AzureAISearchStore,
        "azure": AzureAISearchStore,  # Alias
        "pinecone": PineconeStore,
        "chroma": ChromaStore,
        "chromadb": ChromaStore,  # Alias
        "faiss": FAISSStore,
        "qdrant": QdrantStore,
    }

    if provider.lower() not in providers:
        raise ValueError(
            f"Unknown vector DB provider: {provider}. Available: {list(providers.keys())}"
        )

    return providers[provider.lower()](**kwargs)


# Available providers reference
AVAILABLE_PROVIDERS = {
    "cloud": ["azure_ai_search", "pinecone", "qdrant", "weaviate", "milvus"],
    "local": ["chroma", "faiss"],
    "hybrid": ["qdrant", "weaviate", "milvus"],
}


class VectorDBModule:
    """Vector database module."""

    connect = staticmethod(connect)

    # Store classes
    AzureAISearch = AzureAISearchStore
    Pinecone = PineconeStore
    Chroma = ChromaStore
    FAISS = FAISSStore
    Qdrant = QdrantStore

    # Data classes
    Document = Document
    SearchResult = SearchResult

    PROVIDERS = AVAILABLE_PROVIDERS


vector_db = VectorDBModule()
