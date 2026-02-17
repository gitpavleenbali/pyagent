# ChromaDB

ChromaDB is an open-source embedding database that runs locally or in-memory.

## Installation

```bash
pip install pyagent[vectordb]
# or specifically
pip install chromadb
```

## Connection

```python
from pyagent.vectordb import connect

# Persistent storage
db = connect("chroma", path="./chroma_db")

# In-memory (for testing)
db = connect("chroma", persist=False)

# With custom settings
db = connect(
    "chroma",
    path="./data",
    collection_name="my_collection"
)
```

## Configuration

```python
from pyagent.vectordb.chroma import ChromaStore

store = ChromaStore(
    path="./chroma_db",                    # Storage path
    collection_name="documents",            # Collection name
    embedding_model="text-embedding-3-small" # Embedding model
)
```

## Basic Operations

### Add Documents

```python
# Simple add
db.add([
    "First document text",
    "Second document text"
])

# With metadata and IDs
db.add(
    documents=["Document content"],
    metadatas=[{"source": "web", "date": "2024-01-15"}],
    ids=["doc-001"]
)
```

### Search

```python
results = db.search("query text", n=5)

for result in results:
    print(f"ID: {result.id}")
    print(f"Content: {result.content}")
    print(f"Score: {result.score}")
    print(f"Metadata: {result.metadata}")
```

### Search with Filters

```python
# Exact match
results = db.search(
    "query",
    n=10,
    filter={"source": "web"}
)

# Multiple conditions
results = db.search(
    "query",
    filter={
        "$and": [
            {"source": "web"},
            {"date": {"$gte": "2024-01-01"}}
        ]
    }
)
```

### Update

```python
db.update(
    ids=["doc-001"],
    documents=["Updated content"],
    metadatas=[{"updated": True}]
)
```

### Delete

```python
# By ID
db.delete(ids=["doc-001", "doc-002"])

# By filter
db.delete(filter={"source": "outdated"})

# Clear collection
db.delete(all=True)
```

## Collections

```python
# Create new collection
db.create_collection("new_collection")

# List collections
collections = db.list_collections()

# Switch collection
db.use_collection("other_collection")

# Delete collection
db.delete_collection("old_collection")
```

## Embeddings

### Default (OpenAI)

```python
db = connect("chroma", embedding_model="text-embedding-3-small")
```

### Custom Embedding Function

```python
def my_embed(texts: list[str]) -> list[list[float]]:
    # Your embedding logic
    return embeddings

db = connect("chroma", embedding_function=my_embed)
```

### Local Embeddings

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

db = connect(
    "chroma",
    embedding_function=lambda texts: model.encode(texts).tolist()
)
```

## See Also

- [VectorDB-Module](VectorDB-Module) - Module overview
- [Pinecone](Pinecone) - Cloud vector database
- [Qdrant](Qdrant) - High-performance option
