# PyAgent Integrations

**Connect PyAgent to the AI Ecosystem**

The `integrations` module provides seamless connections to popular AI frameworks, vector databases, and external services. Build on top of existing infrastructure without starting from scratch.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         PyAgent                                  │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  LangChain   │  │   Semantic   │  │   Vector     │          │
│  │   Adapter    │  │    Kernel    │  │  Databases   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                    │
└─────────┼─────────────────┼─────────────────┼────────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
   ┌──────────┐      ┌──────────┐      ┌──────────┐
   │LangChain │      │ Semantic │      │ Azure AI │
   │   Tools  │      │  Kernel  │      │  Search  │
   └──────────┘      └──────────┘      │ Pinecone │
                                       │  Chroma  │
                                       │  FAISS   │
                                       │  Qdrant  │
                                       └──────────┘
```

## LangChain Adapter

Import tools, chains, and retrievers from LangChain:

### Import LangChain Tools

```python
from pyagent.integrations import langchain

# Import search tools
search_tool = langchain.import_tool("serpapi")
wiki_tool = langchain.import_tool("wikipedia")

# Use with PyAgent
from pyagent import agent
researcher = agent("Research assistant", skills=[search_tool, wiki_tool])
```

### Import LangChain Chains

```python
# Import an existing chain
summarizer = langchain.import_chain(my_langchain_chain)

# Use like a PyAgent skill
result = summarizer("Long document text...")
```

### Export PyAgent to LangChain

```python
# Create a PyAgent agent
my_agent = agent("You are a helpful assistant")

# Export as LangChain tool
lc_tool = langchain.export_agent(my_agent, name="helper")

# Use in LangChain workflows
from langchain.agents import AgentExecutor
langchain_agent = AgentExecutor(tools=[lc_tool], ...)
```

### Import Retrievers

```python
# Import LangChain retriever
retriever = langchain.import_retriever(my_langchain_retriever)

# Use with PyAgent RAG
from pyagent import rag
response = rag("question", retriever=retriever)
```

## Semantic Kernel Adapter

Integrate with Microsoft's Semantic Kernel:

### Create Kernel

```python
from pyagent.integrations import semantic_kernel

# Create with Azure OpenAI
kernel = semantic_kernel.create_kernel(
    provider="azure",
    deployment="gpt-4o",
    endpoint="https://your-resource.openai.azure.com"
)

# Create with OpenAI
kernel = semantic_kernel.create_kernel(
    provider="openai",
    model="gpt-4"
)
```

### Import SK Plugins

```python
# Import existing SK plugin
plugin = semantic_kernel.import_plugin(
    kernel,
    "path/to/plugin",
    name="MyPlugin"
)

# Use functions from the plugin
result = plugin.function("input")
```

### Create Plans

```python
# Create a sequential plan
plan = semantic_kernel.create_plan(
    kernel,
    goal="Summarize this document and translate to Spanish",
    available_functions=["summarize", "translate"]
)

# Execute the plan
result = semantic_kernel.execute_plan(kernel, plan)
```

### Export to SK

```python
# Export PyAgent agent as SK function
sk_function = semantic_kernel.export_to_kernel(
    kernel,
    my_agent,
    name="PyAgentHelper"
)
```

### Create Memory

```python
# Create semantic memory
memory = semantic_kernel.create_memory(
    kernel,
    collection="knowledge_base"
)

# Save facts
memory.save("Paris is the capital of France", "geography")
memory.save("The Eiffel Tower is 330m tall", "landmarks")

# Search memory
results = memory.search("France capital", top=3)
```

## Vector Database Connectors

Unified interface for vector storage:

### Connection Factory

```python
from pyagent.integrations import vector_db

# Azure AI Search
store = vector_db.connect(
    "azure_ai_search",
    endpoint="https://search.windows.net",
    index_name="my-index",
    credential="api-key"  # or DefaultAzureCredential()
)

# Pinecone
store = vector_db.connect(
    "pinecone",
    api_key="...",
    index_name="my-index"
)

# ChromaDB (local)
store = vector_db.connect(
    "chroma",
    path="./vectors"
)

# FAISS (in-memory)
store = vector_db.connect(
    "faiss",
    dimension=1536
)

# Qdrant
store = vector_db.connect(
    "qdrant",
    host="localhost",
    port=6333,
    collection="my-collection"
)
```

### Unified Operations

```python
# Add documents
store.add_documents([
    {"id": "1", "content": "First document", "metadata": {"category": "A"}},
    {"id": "2", "content": "Second document", "metadata": {"category": "B"}}
])

# Search
results = store.search(
    query="find relevant documents",
    top_k=5,
    filter={"category": "A"}
)

# Get by ID
doc = store.get("1")

# Delete
store.delete("1")
```

### With Embeddings

```python
# Use custom embedding model
from pyagent.core import embeddings

store = vector_db.connect(
    "chroma",
    path="./vectors",
    embedding_function=embeddings.get("text-embedding-3-small")
)
```

### Use with RAG

```python
from pyagent import rag
from pyagent.integrations import vector_db

# Connect to your vector store
store = vector_db.connect("azure_ai_search", ...)

# Use directly in RAG
response = rag(
    "What is the return policy?",
    store=store,
    top_k=3
)
```

## Module Structure

```
integrations/
├── __init__.py              # Module exports
├── langchain_adapter.py     # LangChain integration
├── semantic_kernel_adapter.py # SK integration
└── vector_db.py             # Vector database connectors
```

## Supported Backends

### LangChain
- Tools: SerpAPI, Wikipedia, ArXiv, PubMed, Python REPL, etc.
- Chains: Any LangChain chain/LCEL runnable
- Retrievers: All LangChain retrievers

### Semantic Kernel
- Providers: OpenAI, Azure OpenAI
- Features: Plugins, Planners, Memory

### Vector Databases
| Database | Type | Best For |
|----------|------|----------|
| Azure AI Search | Cloud | Enterprise, hybrid search |
| Pinecone | Cloud | Scalable production |
| ChromaDB | Local/Cloud | Development, small-medium |
| FAISS | In-memory | Fast prototyping |
| Qdrant | Self-hosted | Full control |

## Authentication

### Azure Services (Recommended)

```python
from azure.identity import DefaultAzureCredential

# Uses managed identity, Azure CLI, environment vars, etc.
credential = DefaultAzureCredential()

store = vector_db.connect(
    "azure_ai_search",
    endpoint="...",
    credential=credential
)
```

### API Keys

```python
store = vector_db.connect(
    "pinecone",
    api_key=os.environ["PINECONE_API_KEY"],
    ...
)
```

## Best Practices

1. **Choose the Right Store**: 
   - Development: ChromaDB or FAISS
   - Production: Azure AI Search or Pinecone
   - Self-hosted: Qdrant

2. **Use Azure Identity**: Prefer `DefaultAzureCredential` over API keys

3. **Batch Operations**: Add documents in batches for performance

4. **Index Wisely**: Create indexes for frequently filtered fields

5. **Monitor Costs**: Cloud vector stores charge per operation

## See Also

- [Azure AI Search Docs](https://docs.microsoft.com/azure/search/)
- [Semantic Kernel Docs](https://learn.microsoft.com/semantic-kernel/)
- [LangChain Docs](https://python.langchain.com/)
