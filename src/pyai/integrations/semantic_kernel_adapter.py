"""
Semantic Kernel Integration for pyai
========================================

Bridge between pyai and Microsoft's Semantic Kernel framework.
Import SK plugins/functions into pyai, or use pyai within SK.

Features:
- Import Semantic Kernel plugins as pyai skills
- Use SK's planning capabilities with pyai agents
- Export pyai functions as SK plugins
- Memory/embeddings interoperability
- Azure AI integration via SK connectors

Examples:
    >>> from pyai.integrations import semantic_kernel as sk

    # Import SK plugin
    >>> writer = sk.import_plugin("WriterPlugin")
    >>> agent = pyai.agent("writer", plugins=[writer])

    # Use SK planner with pyai
    >>> plan = sk.create_plan("Write a blog post about AI")
    >>> result = sk.execute_plan(plan)

    # Export pyai to SK
    >>> sk_kernel = sk.export_to_kernel(my_agent)
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class SKPlugin:
    """Wrapper for Semantic Kernel plugins in pyai."""

    name: str
    description: str
    functions: Dict[str, Callable] = field(default_factory=dict)
    _sk_plugin: Any = None

    def __call__(self, function_name: str, *args, **kwargs) -> Any:
        """Execute a plugin function."""
        if function_name in self.functions:
            return self.functions[function_name](*args, **kwargs)
        raise ValueError(f"Function {function_name} not found in plugin {self.name}")

    def list_functions(self) -> List[str]:
        """List available functions in this plugin."""
        return list(self.functions.keys())


@dataclass
class SKPlan:
    """Wrapper for Semantic Kernel plans."""

    goal: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    _sk_plan: Any = None

    def __str__(self) -> str:
        return f"Plan(goal='{self.goal}', steps={len(self.steps)})"


def create_kernel(
    *, provider: str = "azure", deployment: str = None, endpoint: str = None, api_key: str = None
):
    """
    Create a Semantic Kernel instance configured with pyai settings.

    Args:
        provider: AI provider (azure, openai)
        deployment: Model deployment name
        endpoint: API endpoint
        api_key: API key (optional for Azure AD auth)

    Returns:
        Configured Semantic Kernel

    Examples:
        >>> kernel = sk.create_kernel(provider="azure")
        >>> result = kernel.run_async(...)
    """
    try:
        import semantic_kernel as sk
        from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion

        kernel = sk.Kernel()

        if provider == "azure":
            deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
            endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")

            if not endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT not set")

            # Try Azure AD auth first
            if not api_key:
                try:
                    from azure.identity import DefaultAzureCredential

                    credential = DefaultAzureCredential()

                    kernel.add_service(
                        AzureChatCompletion(
                            deployment_name=deployment,
                            endpoint=endpoint,
                            ad_token_provider=lambda: (
                                credential.get_token(
                                    "https://cognitiveservices.azure.com/.default"
                                ).token
                            ),
                        )
                    )
                except ImportError:
                    raise ImportError("azure-identity not installed for Azure AD auth")
            else:
                kernel.add_service(
                    AzureChatCompletion(
                        deployment_name=deployment, endpoint=endpoint, api_key=api_key
                    )
                )
        else:
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            kernel.add_service(
                OpenAIChatCompletion(model_id=deployment or "gpt-4o-mini", api_key=api_key)
            )

        return kernel

    except ImportError:
        raise ImportError("Semantic Kernel not installed. Run: pip install semantic-kernel")


def import_plugin(plugin_name: str, *, plugin_path: str = None, kernel=None) -> SKPlugin:
    """
    Import a Semantic Kernel plugin for use in pyai.

    Args:
        plugin_name: Name of the plugin
        plugin_path: Path to plugin directory (for custom plugins)
        kernel: Existing SK kernel (creates new if not provided)

    Returns:
        SKPlugin wrapper

    Built-in plugins:
        - ConversationSummaryPlugin
        - HttpPlugin
        - MathPlugin
        - TextPlugin
        - TimePlugin
        - FileIOPlugin
        - WebSearchPlugin
    """
    try:
        import semantic_kernel as sk
        from semantic_kernel.core_plugins import (
            ConversationSummaryPlugin,
            HttpPlugin,
            MathPlugin,
            TextPlugin,
            TimePlugin,
        )

        kernel = kernel or create_kernel()

        # Map plugin names to classes
        builtin_plugins = {
            "ConversationSummaryPlugin": ConversationSummaryPlugin,
            "HttpPlugin": HttpPlugin,
            "MathPlugin": MathPlugin,
            "TextPlugin": TextPlugin,
            "TimePlugin": TimePlugin,
        }

        if plugin_name in builtin_plugins:
            plugin_class = builtin_plugins[plugin_name]
            sk_plugin = kernel.add_plugin(plugin_class(), plugin_name)

            # Extract functions
            functions = {}
            for func_name in dir(sk_plugin):
                if not func_name.startswith("_"):
                    func = getattr(sk_plugin, func_name)
                    if callable(func):
                        functions[func_name] = func

            return SKPlugin(
                name=plugin_name,
                description=f"Semantic Kernel {plugin_name}",
                functions=functions,
                _sk_plugin=sk_plugin,
            )
        elif plugin_path:
            # Load from directory
            sk_plugin = kernel.import_plugin_from_prompt_directory(plugin_path, plugin_name)
            return SKPlugin(
                name=plugin_name,
                description=f"Custom plugin from {plugin_path}",
                _sk_plugin=sk_plugin,
            )
        else:
            raise ValueError(f"Unknown plugin: {plugin_name}")

    except ImportError:
        raise ImportError("Semantic Kernel not installed. Run: pip install semantic-kernel")


def create_plan(goal: str, *, planner_type: str = "sequential", kernel=None) -> SKPlan:
    """
    Create a plan using Semantic Kernel's planning capabilities.

    Args:
        goal: The goal to achieve
        planner_type: Type of planner ("sequential", "stepwise", "basic")
        kernel: SK kernel to use

    Returns:
        SKPlan that can be executed

    Examples:
        >>> plan = sk.create_plan("Write a blog post and tweet about it")
        >>> result = sk.execute_plan(plan)
    """
    try:
        import semantic_kernel as sk
        from semantic_kernel.planners import BasicPlanner, SequentialPlanner

        kernel = kernel or create_kernel()

        if planner_type == "sequential":
            planner = SequentialPlanner(kernel)
        else:
            planner = BasicPlanner()

        # Create plan (async in SK, we wrap it)
        import asyncio

        loop = asyncio.get_event_loop()
        sk_plan = loop.run_until_complete(planner.create_plan(goal))

        return SKPlan(
            goal=goal, steps=[{"description": str(s)} for s in sk_plan._steps], _sk_plan=sk_plan
        )

    except ImportError:
        raise ImportError("Semantic Kernel not installed")


def execute_plan(plan: SKPlan, kernel=None) -> str:
    """
    Execute a Semantic Kernel plan.

    Args:
        plan: SKPlan to execute
        kernel: SK kernel to use

    Returns:
        Plan execution result
    """
    try:
        import asyncio

        kernel = kernel or create_kernel()

        if plan._sk_plan:
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(kernel.run_async(plan._sk_plan))
            return str(result)
        else:
            raise ValueError("Plan has no SK plan object")

    except ImportError:
        raise ImportError("Semantic Kernel not installed")


def export_to_kernel(pyai_agent, kernel=None):
    """
    Export a pyai agent as a Semantic Kernel plugin.

    Args:
        pyai_agent: pyai SimpleAgent instance
        kernel: SK kernel to add plugin to

    Returns:
        SK kernel with pyai plugin added
    """
    try:
        import semantic_kernel as sk
        from semantic_kernel.functions import kernel_function

        kernel = kernel or create_kernel()

        # Create a wrapper plugin
        class pyaiPlugin:
            def __init__(self, agent):
                self._agent = agent

            @kernel_function(description="Run the pyai agent")
            def run(self, input: str) -> str:
                return self._agent(input)

        plugin = pyaiPlugin(pyai_agent)
        kernel.add_plugin(plugin, pyai_agent.name)

        return kernel

    except ImportError:
        raise ImportError("Semantic Kernel not installed")


def create_memory(*, provider: str = "azure_ai_search", **kwargs):
    """
    Create a Semantic Kernel memory store.

    Args:
        provider: Memory provider type
        **kwargs: Provider configuration

    Supported providers:
        - azure_ai_search: Azure AI Search
        - azure_cosmos_db: Azure Cosmos DB
        - chroma: ChromaDB
        - pinecone: Pinecone
        - volatile: In-memory (for testing)
    """
    try:
        import semantic_kernel as sk

        if provider == "azure_ai_search":
            from semantic_kernel.connectors.memory.azure_ai_search import AzureAISearchMemoryStore

            return AzureAISearchMemoryStore(**kwargs)
        elif provider == "volatile":
            from semantic_kernel.memory import VolatileMemoryStore

            return VolatileMemoryStore()
        else:
            raise ValueError(f"Unknown memory provider: {provider}")

    except ImportError:
        raise ImportError("Semantic Kernel not installed")


# Available plugins reference
AVAILABLE_PLUGINS = {
    "core": [
        "ConversationSummaryPlugin",
        "HttpPlugin",
        "MathPlugin",
        "TextPlugin",
        "TimePlugin",
    ],
    "azure": [
        "AzureSearchPlugin",
        "AzureBlobStoragePlugin",
        "CosmosDBPlugin",
    ],
}


class SemanticKernelModule:
    """Semantic Kernel integration module."""

    create_kernel = staticmethod(create_kernel)
    import_plugin = staticmethod(import_plugin)
    create_plan = staticmethod(create_plan)
    execute_plan = staticmethod(execute_plan)
    export_to_kernel = staticmethod(export_to_kernel)
    create_memory = staticmethod(create_memory)

    PLUGINS = AVAILABLE_PLUGINS


semantic_kernel = SemanticKernelModule()
