# Weaviate

Weaviate is an open-source vector database with GraphQL API and multi-modal support.

## Installation

```bash
pip install pyagent[vectordb]
# or specifically
pip install weaviate-client
```

## Running Weaviate

### Docker

```bash
docker run -p 8080:8080 semitechnologies/weaviate
```

### Docker Compose

```yaml
version: '3.4'
services:
  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
```

## Connection

```python
from pyagent.vectordb import connect

# Local instance
db = connect("weaviate", url="http://localhost:8080")

# Weaviate Cloud
db = connect(
    "weaviate",
    url="https://your-cluster.weaviate.network",
    api_key="your-api-key"
)

# With class name
db = connect(
    "weaviate",
    url="http://localhost:8080",
    class_name="Document"
)
```

## Configuration

```python
from pyagent.vectordb.weaviate import WeaviateStore

store = WeaviateStore(
    url="http://localhost:8080",
    class_name="Document",
    api_key=None,
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

# With metadata
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
    print(f"Content: {result.content}")
```

### Search with Filters

```python
results = db.search(
    "query",
    n=10,
    filter={
        "path": ["category"],
        "operator": "Equal",
        "valueText": "tech"
    }
)
```

### GraphQL Queries

```python
# Direct GraphQL query
result = db.graphql_query("""
{
    Get {
        Document(
            nearText: {concepts: ["machine learning"]}
            limit: 5
        ) {
            content
            category
            _additional {
                certainty
            }
        }
    }
}
""")
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
db.delete(ids=["doc-001"])

# By filter
db.delete(filter={"path": ["archived"], "operator": "Equal", "valueBoolean": True})
```

## Schema / Classes

```python
# Create class (schema)
db.create_class(
    name="Article",
    properties=[
        {"name": "content", "dataType": ["text"]},
        {"name": "title", "dataType": ["string"]},
        {"name": "published", "dataType": ["date"]}
    ]
)

# List classes
classes = db.list_classes()

# Delete class
db.delete_class("OldClass")
```

## Filter Operators

| Operator | Description |
|----------|-------------|
| `Equal` | Exact match |
| `NotEqual` | Not equal |
| `GreaterThan` | Greater than |
| `GreaterThanEqual` | Greater or equal |
| `LessThan` | Less than |
| `LessThanEqual` | Less or equal |
| `Like` | Pattern match (wildcards) |

## Multi-modal Support

Weaviate supports images and other media:

```python
# Add image with text
db.add(
    documents=["A photo of a sunset"],
    images=["base64_encoded_image..."],
    metadatas=[{"type": "photo"}]
)

# Search by image
results = db.search_by_image(image_data)
```

## Hybrid Search

Combine vector and keyword search:

```python
results = db.hybrid_search(
    query="machine learning",
    alpha=0.5,  # 0 = keyword only, 1 = vector only
    n=10
)
```

## See Also

- [VectorDB-Module](VectorDB-Module) - Module overview
- [ChromaDB](ChromaDB) - Simple local option
- [Pinecone](Pinecone) - Managed cloud option
- [Qdrant](Qdrant) - High-performance option
