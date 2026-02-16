# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Kernel Module

Central orchestration layer providing unified service management,
plugin registration, and component orchestration - similar to 
Microsoft Semantic Kernel's architectural pattern.

The Kernel serves as the central nervous system for AI applications,
managing:
- Services (LLM providers, memory, vector stores)
- Plugins (tools and skills as reusable components)
- Filters (middleware for processing)
- Configuration (unified settings)

Example:
    from pyagent import Kernel
    
    # Create kernel with services
    kernel = Kernel()
    kernel.add_service("llm", AzureOpenAI())
    kernel.add_service("memory", ConversationMemory())
    
    # Register plugins
    kernel.add_plugin(WeatherPlugin())
    kernel.add_plugin(SearchPlugin())
    
    # Create agent with kernel
    agent = kernel.create_agent(
        name="assistant",
        instructions="You are a helpful assistant"
    )
    
    # Invoke function
    result = await kernel.invoke("weather", "get_forecast", city="NYC")
"""

from .kernel import Kernel, KernelBuilder
from .services import (
    ServiceRegistry,
    Service,
    ServiceType,
    LLMService,
    MemoryService,
    VectorService,
)
from .filters import (
    FilterRegistry,
    Filter,
    FilterType,
    PromptFilter,
    FunctionFilter,
    FilterContext,
)
from .context import KernelContext, InvocationContext

__all__ = [
    # Core
    "Kernel",
    "KernelBuilder",
    # Services
    "ServiceRegistry",
    "Service",
    "ServiceType",
    "LLMService",
    "MemoryService",
    "VectorService",
    # Filters
    "FilterRegistry",
    "Filter",
    "FilterType",
    "PromptFilter",
    "FunctionFilter",
    "FilterContext",
    # Context
    "KernelContext",
    "InvocationContext",
]
