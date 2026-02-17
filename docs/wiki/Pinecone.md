# Pinecone

Pinecone is a managed cloud vector database optimized for production workloads.

## Installation

```bash
pip install pyagent[vectordb]
# or specifically
pip install pinecone-client
```

## Connection

```python
from pyagent.vectordb import connect

db = connect(
    "pinecone",
    api_key="your-api-key",
    index_name="my-index",
    environment="us-east-1"  # Your Pinecone environment
)
```

## Configuration

```python
from pyagent.vectordb.pinecone import PineconeStore

store = PineconeStore(
    api_key="your-api-key",
    index_name="my-index",
    environment="us-east-1",
    namespace="default",           # Optional namespace
    embedding_model="text-embedding-3-small"
)
```

## Creating an Index

Indexes must be created in Pinecone console or via API:

```python
import pinecone

pinecone.init(api_key="your-api-key", environment="us-east-1")

# Create index (do once)
pinecone.create_index(
    name="my-index",
    dimension=1536,  # OpenAI embedding dimension
    metric="cosine"
)
```

## Basic Operations

### Add Documents

```python
# Simple add
db.add([
    "First document",
    "Second document"
])

# With metadata and IDs
db.add(
    documents=["Document content"],
    metadatas=[{"source": "web", "category": "tech"}],
    ids=["doc-001"]
)
```

### Search

```python
results = db.search("query text", n=5)

for result in results:
    print(f"ID: {result.id}")
    print(f"Score: {result.score}")
    print(f"Metadata: {result.metadata}")
```

### Search with Filters

```python
results = db.search(
    "query",
    n=10,
    filter={
        "category": {"$eq": "tech"},
        "year": {"$gte": 2023}
    }
)
```

### Update

```python
db.update(
    ids=["doc-001"],
    metadatas=[{"updated": True}]
)
```

### Delete

```python
# By ID
db.delete(ids=["doc-001", "doc-002"])

# By filter
db.delete(filter={"category": "outdated"})

# Delete all
db.delete(delete_all=True)
```

## Namespaces

Organize data within an index:

```python
# Use specific namespace
db = connect(
    "pinecone",
    api_key="...",
    index_name="my-index",
    namespace="production"
)

# Query across namespaces
results = db.search("query", namespace=None)  # All namespaces
```

## Filter Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` | Equal | `{"field": {"$eq": "value"}}` |
| `$ne` | Not equal | `{"field": {"$ne": "value"}}` |
| `$gt` | Greater than | `{"field": {"$gt": 10}}` |
| `$gte` | Greater or equal | `{"field": {"$gte": 10}}` |
| `$lt` | Less than | `{"field": {"$lt": 10}}` |
| `$lte` | Less or equal | `{"field": {"$lte": 10}}` |
| `$in` | In array | `{"field": {"$in": ["a", "b"]}}` |

## Statistics

```python
stats = db.describe_index()
print(f"Total vectors: {stats['total_vector_count']}")
print(f"Dimension: {stats['dimension']}")
```

## See Also

- [VectorDB-Module](VectorDB-Module) - Module overview
- [ChromaDB](ChromaDB) - Local option
- [Qdrant](Qdrant) - Self-hosted option
