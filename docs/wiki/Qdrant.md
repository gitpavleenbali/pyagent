# Qdrant

Qdrant is a high-performance, Rust-based vector database with rich filtering capabilities.

## Installation

```bash
pip install pyai[vectordb]
# or specifically
pip install qdrant-client
```

## Running Qdrant

### Docker

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Qdrant Cloud

Sign up at [cloud.qdrant.io](https://cloud.qdrant.io)

## Connection

```python
from pyai.vectordb import connect

# Local instance
db = connect("qdrant", url="http://localhost:6333")

# Qdrant Cloud
db = connect(
    "qdrant",
    url="https://your-cluster.qdrant.io",
    api_key="your-api-key"
)

# With collection name
db = connect(
    "qdrant",
    url="http://localhost:6333",
    collection_name="my_documents"
)
```

## Configuration

```python
from pyai.vectordb.qdrant import QdrantStore

store = QdrantStore(
    url="http://localhost:6333",
    collection_name="documents",
    api_key=None,                          # Optional
    embedding_model="text-embedding-3-small"
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
    metadatas=[{"source": "web", "tags": ["python", "ai"]}],
    ids=["doc-001"]
)
```

### Search

```python
results = db.search("query text", n=5)

for result in results:
    print(f"ID: {result.id}")
    print(f"Score: {result.score}")
    print(f"Content: {result.content}")
```

### Search with Filters

```python
# Exact match
results = db.search(
    "query",
    n=10,
    filter={
        "must": [{"key": "source", "match": {"value": "web"}}]
    }
)

# Range filter
results = db.search(
    "query",
    filter={
        "must": [{"key": "year", "range": {"gte": 2023}}]
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
db.delete(filter={"must": [{"key": "archived", "match": {"value": True}}]})
```

## Collections

```python
# Create collection
db.create_collection(
    name="new_collection",
    dimension=1536,
    distance="cosine"  # or "euclid", "dot"
)

# List collections
collections = db.list_collections()

# Delete collection
db.delete_collection("old_collection")
```

## Advanced Filtering

Qdrant supports complex filtering:

```python
filter = {
    "must": [
        {"key": "category", "match": {"value": "tech"}}
    ],
    "should": [
        {"key": "language", "match": {"value": "python"}},
        {"key": "language", "match": {"value": "rust"}}
    ],
    "must_not": [
        {"key": "archived", "match": {"value": True}}
    ]
}

results = db.search("query", filter=filter)
```

### Filter Types

| Type | Description |
|------|-------------|
| `match` | Exact value match |
| `range` | Numeric range (gte, lte, gt, lt) |
| `geo_bounding_box` | Geographic bounds |
| `geo_radius` | Geographic radius |

## Performance Features

### Quantization

```python
# Enable scalar quantization for faster search
db = connect(
    "qdrant",
    url="...",
    quantization="scalar"
)
```

### Sharding

```python
# Create collection with sharding
db.create_collection(
    name="large_collection",
    shard_number=3
)
```

## See Also

- [VectorDB-Module](VectorDB-Module) - Module overview
- [ChromaDB](ChromaDB) - Simple local option
- [Pinecone](Pinecone) - Managed cloud option
