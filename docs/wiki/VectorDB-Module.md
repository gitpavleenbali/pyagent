# VectorDB Module

The VectorDB module provides connectors and abstractions for vector databases, enabling semantic search and RAG capabilities.

## Overview

```python
from pyai.vectordb import connect
from pyai.integrations.vector_db import VectorStore, Document
```

## Supported Databases

| Database | Description |
|----------|-------------|
| [ChromaDB](ChromaDB) | Open-source, lightweight |
| [Pinecone](Pinecone) | Managed cloud service |
| [Qdrant](Qdrant) | High-performance, Rust-based |
| [Weaviate](Weaviate) | GraphQL-based, multi-modal |

## Quick Start

### Connect to Database

```python
from pyai.vectordb import connect

# ChromaDB (local)
db = connect("chroma", path="./my_db")

# Pinecone (cloud)
db = connect("pinecone", api_key="...", index_name="my-index")

# Qdrant (local or cloud)
db = connect("qdrant", url="http://localhost:6333")

# Weaviate
db = connect("weaviate", url="http://localhost:8080")
```

### Store Documents

```python
# Store text documents
db.add([
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming industries.",
    "Python is a versatile programming language."
])

# Store with metadata
db.add(
    documents=["Document content here..."],
    metadatas=[{"source": "wiki", "category": "tech"}],
    ids=["doc-001"]
)
```

### Search

```python
# Semantic search
results = db.search("What is machine learning?", n=5)

for doc in results:
    print(f"Score: {doc.score:.3f} - {doc.content[:100]}")
```

### With Filters

```python
results = db.search(
    query="Python programming",
    n=10,
    filter={"category": "tech"}
)
```

## Document Class

```python
from pyai.integrations.vector_db import Document

doc = Document(
    content="Document text content",
    metadata={"author": "John", "date": "2024-01-15"},
    embedding=[0.1, 0.2, ...],  # Optional
    id="doc-123"
)
```

## Embedding Functions

```python
# Use default OpenAI embeddings
db = connect("chroma", embedding_model="text-embedding-3-small")

# Custom embedding function
def my_embeddings(texts: list[str]) -> list[list[float]]:
    # Your embedding logic
    return embeddings

db = connect("chroma", embedding_function=my_embeddings)
```

## RAG Integration

```python
from pyai.easy import rag

# Index documents
index = rag.index("./documents")

# Query with RAG
answer = rag.ask(
    index,
    "What is the main conclusion?",
    n_results=5
)
```

## Operations

### Add Documents

```python
db.add(
    documents=["text1", "text2"],
    ids=["id1", "id2"],
    metadatas=[{}, {}]
)
```

### Update Documents

```python
db.update(
    ids=["id1"],
    documents=["Updated text"],
    metadatas=[{"updated": True}]
)
```

### Delete Documents

```python
db.delete(ids=["id1", "id2"])
db.delete(filter={"category": "outdated"})
```

### Count

```python
count = db.count()
print(f"Total documents: {count}")
```

## See Also

- [ChromaDB](ChromaDB) - ChromaDB connector
- [Pinecone](Pinecone) - Pinecone connector
- [Qdrant](Qdrant) - Qdrant connector
- [Weaviate](Weaviate) - Weaviate connector
