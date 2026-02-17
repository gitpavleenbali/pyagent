# Kernel Registry

The Kernel provides Semantic Kernel-style service management for enterprise deployments.

## Overview

The Kernel pattern allows:
- Centralized service registration
- Dependency injection
- Multi-provider management
- Plugin architecture

## Basic Usage

```python
from pyai.kernel import Kernel

# Create kernel
kernel = Kernel()

# Register services
kernel.add_service(openai_provider, "llm")
kernel.add_service(vector_db, "memory")

# Use services
llm = kernel.get_service("llm")
memory = kernel.get_service("memory")
```

## Service Registration

### Add LLM Providers

```python
from pyai.kernel import Kernel
from pyai.core import OpenAIProvider, AzureOpenAIProvider

kernel = Kernel()

# Register OpenAI
kernel.add_service(
    OpenAIProvider(api_key="sk-..."),
    service_id="openai"
)

# Register Azure
kernel.add_service(
    AzureOpenAIProvider(
        endpoint="https://...",
        deployment="gpt-4o-mini"
    ),
    service_id="azure"
)
```

### Add Memory

```python
from pyai.vectordb import ChromaDB

kernel.add_service(
    ChromaDB(collection="knowledge"),
    service_id="memory"
)
```

### Add Plugins

```python
from pyai.plugins import WebPlugin, FilePlugin

kernel.add_plugin(WebPlugin())
kernel.add_plugin(FilePlugin())
```

## Using with Agents

```python
from pyai import Agent
from pyai.kernel import Kernel

kernel = Kernel()
kernel.add_service(provider, "llm")
kernel.add_service(memory, "memory")

# Agent uses kernel services
agent = Agent(
    name="Enterprise Agent",
    kernel=kernel
)
```

## Service Discovery

```python
# List all services
services = kernel.list_services()
print(services)  # ["llm", "memory", "cache"]

# Check service exists
if kernel.has_service("llm"):
    llm = kernel.get_service("llm")
```

## Default Services

```python
# Set default LLM
kernel.set_default_service("llm", "azure")

# Get default
default_llm = kernel.get_default_service("llm")
```

## Dependency Injection

```python
from pyai.kernel import inject

class MyAgent:
    @inject("llm")
    def __init__(self, llm=None):
        self.llm = llm
    
    async def run(self, message):
        return await self.llm.generate([message])

# Kernel injects dependencies
agent = kernel.create(MyAgent)
```

## Configuration

```python
# From config file
kernel = Kernel.from_config("kernel.yaml")
```

```yaml
# kernel.yaml
services:
  llm:
    type: azure_openai
    endpoint: ${AZURE_OPENAI_ENDPOINT}
    deployment: gpt-4o-mini
  
  memory:
    type: chromadb
    collection: knowledge

plugins:
  - web
  - file
  - math
```

## Examples

### Enterprise Setup

```python
from pyai.kernel import Kernel
from pyai.core import AzureOpenAIProvider
from pyai.vectordb import Qdrant
from pyai.sessions import RedisSession

kernel = Kernel()

# LLM
kernel.add_service(
    AzureOpenAIProvider(
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        deployment="gpt-4o-mini",
        use_azure_ad=True
    ),
    service_id="llm"
)

# Vector memory
kernel.add_service(
    Qdrant(url="http://localhost:6333"),
    service_id="memory"
)

# Session storage
kernel.add_service(
    RedisSession(host="redis"),
    service_id="sessions"
)

# Create enterprise agent
agent = Agent(
    name="Enterprise Assistant",
    kernel=kernel
)
```

### Multi-tenant

```python
# Per-tenant kernels
tenants = {}

def get_kernel(tenant_id: str) -> Kernel:
    if tenant_id not in tenants:
        kernel = Kernel()
        kernel.add_service(
            get_tenant_provider(tenant_id),
            "llm"
        )
        tenants[tenant_id] = kernel
    return tenants[tenant_id]
```

## See Also

- [[Sessions]] - Session management
- [[Plugins-Module]] - Plugin system
- [[Agent]] - Agent class
