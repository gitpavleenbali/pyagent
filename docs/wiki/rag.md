# rag

The `rag` module provides a 2-line RAG (Retrieval-Augmented Generation) system.

## Import

```python
from pyai.easy import rag
```

## Basic Usage

```python
# Create RAG system with documents
docs = rag.load("./documents/")

# Query the documents
answer = docs.query("What is the main topic?")
```

## Quick Start

```python
from pyai.easy import rag

# Load documents from directory
knowledge = rag.load("./knowledge_base/")

# Ask questions
answer = knowledge.query("How do I configure the system?")
print(answer)

# With source citations
result = knowledge.query("What are the features?", return_sources=True)
print(result.answer)
print(result.sources)
```

## Loading Documents

### From Directory

```python
# Load all supported files from directory
docs = rag.load("./docs/")

# With file filter
docs = rag.load("./docs/", pattern="*.md")
```

### From Files

```python
# Load specific files
docs = rag.load(["doc1.pdf", "doc2.txt", "doc3.md"])
```

### From URLs

```python
# Load from web pages
docs = rag.load([
    "https://example.com/page1",
    "https://example.com/page2"
])
```

## Query Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | required | Question to ask |
| `top_k` | int | 5 | Number of chunks to retrieve |
| `return_sources` | bool | False | Include source citations |
| `temperature` | float | 0.7 | Response creativity |

## Examples

### Full Example

```python
from pyai.easy import rag

# Create knowledge base
kb = rag.load("./company_docs/")

# Simple query
answer = kb.query("What is our refund policy?")

# Query with sources
result = kb.query(
    "What are the product features?",
    return_sources=True,
    top_k=3
)

for source in result.sources:
    print(f"- {source.file}: {source.chunk}")
```

### Async Usage

```python
import asyncio
from pyai.easy import rag

async def main():
    kb = await rag.load_async("./docs/")
    answer = await kb.query_async("What is PYAI?")
    print(answer)

asyncio.run(main())
```

## Supported Formats

- PDF (`.pdf`)
- Text (`.txt`)
- Markdown (`.md`)
- Word (`.docx`)
- HTML (`.html`)
- JSON (`.json`)

## Configuration

```python
# Custom embedding model
rag.configure(
    embedding_model="text-embedding-3-small",
    chunk_size=500,
    chunk_overlap=50
)
```

## See Also

- [[VectorDB-Module]] - Vector database backends
- [[ChromaDB]] - ChromaDB integration
- [[ask]] - Simple question answering
