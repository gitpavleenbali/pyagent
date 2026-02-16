"""
PyAgent Integrations Module
============================

Connect PyAgent to the broader AI ecosystem. Adapters for popular frameworks,
vector databases, and external services.

Supported Integrations:
- LangChain: Use LangChain tools/chains within PyAgent
- Semantic Kernel: Microsoft's AI orchestration framework
- LlamaIndex: Advanced RAG and indexing
- Vector Databases: Pinecone, Weaviate, Chroma, Qdrant, FAISS
- MCP Servers: External MCP tool servers
- OpenAI Plugins: OpenAI plugin ecosystem
- Hugging Face: Models and datasets

Usage:
    >>> from pyagent.integrations import langchain, semantic_kernel, vector_db
    
    # Use LangChain tools in PyAgent
    >>> tool = langchain.import_tool("serpapi")
    >>> agent = pyagent.agent("researcher", tools=[tool])
    
    # Connect to vector database
    >>> store = vector_db.connect("pinecone", index="my-index")
    >>> rag = pyagent.rag.with_store(store)
"""

from typing import Dict, Any, List, Optional

# Import submodules for easy access
from pyagent.integrations import langchain_adapter as langchain
from pyagent.integrations import semantic_kernel_adapter as semantic_kernel
from pyagent.integrations import vector_db

__all__ = [
    "langchain",
    "semantic_kernel", 
    "vector_db",
]
