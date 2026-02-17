# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Kernel

Central orchestration layer for AI applications.
Manages services, plugins, filters, and provides unified execution.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .context import KernelContext
from .filters import Filter, FilterContext, FilterRegistry
from .services import Service, ServiceRegistry, ServiceType

if TYPE_CHECKING:
    from ..plugins import PluginRegistry


class Kernel:
    """Central kernel for AI application orchestration.

    The Kernel serves as the central nervous system, managing:
    - Services (LLM providers, memory, vector stores)
    - Plugins (tools and skills as reusable components)
    - Filters (middleware for processing)
    - Execution context

    This pattern is inspired by Microsoft Semantic Kernel and provides
    a unified, enterprise-grade foundation for AI applications.

    Example:
        # Create kernel
        kernel = Kernel()

        # Add services
        kernel.add_service(LLMService(
            name="gpt4",
            instance=openai_client,
            is_default=True
        ))

        # Add plugins
        kernel.add_plugin(WeatherPlugin())
        kernel.add_plugin(SearchPlugin())

        # Add filters
        kernel.add_filter(LoggingFilter())

        # Invoke function
        result = await kernel.invoke("weather", "get_forecast", city="NYC")

        # Create agent using kernel services
        agent = kernel.create_agent(
            name="assistant",
            instructions="You are helpful"
        )
    """

    def __init__(
        self,
        *,
        services: Optional[ServiceRegistry] = None,
        filters: Optional[FilterRegistry] = None,
        plugins: Optional["PluginRegistry"] = None,
    ):
        """Initialize kernel.

        Args:
            services: Pre-configured service registry
            filters: Pre-configured filter registry
            plugins: Pre-configured plugin registry
        """
        self._services = services or ServiceRegistry()
        self._filters = filters or FilterRegistry()

        # Import here to avoid circular dependency
        from ..plugins import PluginRegistry

        self._plugins = plugins or PluginRegistry()

        self._context = KernelContext()
        self._agents: Dict[str, Any] = {}

    # ========== Service Management ==========

    @property
    def services(self) -> ServiceRegistry:
        """Get the service registry."""
        return self._services

    def add_service(
        self,
        service: Union[Service, Any],
        name: Optional[str] = None,
        service_type: ServiceType = ServiceType.CUSTOM,
        is_default: bool = False,
    ) -> "Kernel":
        """Add a service to the kernel.

        Args:
            service: Service descriptor or raw instance
            name: Service name (required if service is raw instance)
            service_type: Type of service (for raw instances)
            is_default: Whether this is the default (for raw instances)

        Returns:
            Self for chaining
        """
        if isinstance(service, Service):
            self._services.add(service)
        else:
            service_name: str = name if name else service.__class__.__name__
            self._services.add_instance(
                name=service_name,
                instance=service,
                service_type=service_type,
                is_default=is_default,
            )
        return self

    def get_service(
        self, name: Optional[str] = None, service_type: Optional[ServiceType] = None
    ) -> Optional[Any]:
        """Get a service.

        Args:
            name: Service name (if specific service needed)
            service_type: Get default service of this type

        Returns:
            Service instance or None
        """
        if name:
            return self._services.get(name)
        if service_type:
            return self._services.get_default(service_type)
        return None

    def get_llm(self, name: Optional[str] = None) -> Optional[Any]:
        """Get an LLM service.

        Args:
            name: Specific LLM name, or None for default

        Returns:
            LLM service instance
        """
        if name:
            return self._services.get(name)
        return self._services.get_default(ServiceType.LLM)

    # ========== Plugin Management ==========

    @property
    def plugins(self) -> "PluginRegistry":
        """Get the plugin registry."""
        return self._plugins

    def add_plugin(self, plugin: Any, name: Optional[str] = None) -> "Kernel":
        """Add a plugin to the kernel.

        Args:
            plugin: Plugin instance
            name: Override plugin name

        Returns:
            Self for chaining
        """
        self._plugins.register(plugin, name=name)
        return self

    def get_plugin(self, name: str) -> Optional[Any]:
        """Get a plugin by name.

        Args:
            name: Plugin name

        Returns:
            Plugin instance or None
        """
        return self._plugins.get_plugin(name)

    def get_function(self, plugin_name: str, function_name: str) -> Optional[Any]:
        """Get a function from a plugin.

        Args:
            plugin_name: Plugin name
            function_name: Function name

        Returns:
            Function or None
        """
        return self._plugins.get_function(plugin_name, function_name)

    # ========== Filter Management ==========

    @property
    def filters(self) -> FilterRegistry:
        """Get the filter registry."""
        return self._filters

    def add_filter(self, filter_instance: Filter, priority: int = 100) -> "Kernel":
        """Add a filter to the kernel.

        Args:
            filter_instance: Filter instance
            priority: Execution priority (lower = earlier)

        Returns:
            Self for chaining
        """
        self._filters.add(filter_instance, priority=priority)
        return self

    # ========== Context Management ==========

    @property
    def context(self) -> KernelContext:
        """Get the current execution context."""
        return self._context

    def create_context(self) -> KernelContext:
        """Create a fresh execution context.

        Returns:
            New context (also sets as current)
        """
        self._context = KernelContext()
        return self._context

    def set_variable(self, name: str, value: Any) -> "Kernel":
        """Set a context variable.

        Args:
            name: Variable name
            value: Variable value

        Returns:
            Self for chaining
        """
        self._context.set_variable(name, value)
        return self

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a context variable.

        Args:
            name: Variable name
            default: Default value if not found

        Returns:
            Variable value
        """
        return self._context.get_variable(name, default)

    # ========== Invocation ==========

    async def invoke_async(self, plugin_name: str, function_name: str, **arguments) -> Any:
        """Invoke a plugin function asynchronously.

        Args:
            plugin_name: Name of the plugin
            function_name: Name of the function
            **arguments: Function arguments

        Returns:
            Function result

        Raises:
            ValueError: If plugin/function not found
        """
        # Get function
        func = self.get_function(plugin_name, function_name)
        if func is None:
            raise ValueError(f"Function '{function_name}' not found in plugin '{plugin_name}'")

        # Create invocation context
        inv = self._context.create_invocation(
            plugin_name=plugin_name, function_name=function_name, arguments=arguments
        )

        # Create filter context
        filter_ctx = FilterContext(
            kernel=self, plugin_name=plugin_name, function_name=function_name, arguments=arguments
        )

        try:
            inv.start()

            # Apply pre-invocation filters
            modified_args = self._filters.apply_function_invoking(filter_ctx, arguments)

            # Invoke function
            if asyncio.iscoroutinefunction(func):
                result = await func(**modified_args)
            else:
                result = func(**modified_args)

            # Apply post-invocation filters
            result = self._filters.apply_function_invoked(filter_ctx, result)

            inv.complete(result)
            return result

        except Exception as e:
            inv.fail(e)
            raise

    def invoke(self, plugin_name: str, function_name: str, **arguments) -> Any:
        """Invoke a plugin function (sync wrapper).

        For async invocation, use invoke_async().

        Args:
            plugin_name: Name of the plugin
            function_name: Name of the function
            **arguments: Function arguments

        Returns:
            Function result
        """
        try:
            asyncio.get_running_loop()
            # Already in async context
            return asyncio.create_task(self.invoke_async(plugin_name, function_name, **arguments))
        except RuntimeError:
            # No running loop, run synchronously
            return asyncio.run(self.invoke_async(plugin_name, function_name, **arguments))

    # ========== Agent Creation ==========

    def create_agent(
        self,
        name: str,
        instructions: str,
        *,
        model: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        plugins: Optional[List[str]] = None,
        **kwargs,
    ) -> Any:
        """Create an agent using kernel services.

        Args:
            name: Agent name
            instructions: Agent instructions
            model: Model name (uses default LLM if not specified)
            tools: List of tools
            plugins: List of plugin names to include
            **kwargs: Additional agent configuration

        Returns:
            Agent instance
        """
        from ..core import Agent

        # Collect tools from specified plugins
        all_tools = list(tools or [])
        for plugin_name in plugins or []:
            plugin = self.get_plugin(plugin_name)
            if plugin:
                # Get all functions from plugin as tools
                for func_name in plugin.list_functions():
                    func = plugin.get_function(func_name)
                    if func:
                        all_tools.append(func)

        # Get LLM service
        self.get_llm()

        # Create agent
        agent = Agent(
            name=name,
            instructions=instructions,
            model=model,
            tools=all_tools if all_tools else None,
            **kwargs,
        )

        # Register agent
        self._agents[name] = agent

        return agent

    def get_agent(self, name: str) -> Optional[Any]:
        """Get a registered agent.

        Args:
            name: Agent name

        Returns:
            Agent instance or None
        """
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self._agents.keys())

    # ========== Prompt Execution ==========

    async def invoke_prompt_async(
        self, prompt: str, *, model: Optional[str] = None, **kwargs
    ) -> str:
        """Invoke a prompt on the default LLM.

        Args:
            prompt: The prompt to send
            model: Specific model name (optional)
            **kwargs: Additional parameters

        Returns:
            LLM response
        """
        # Create filter context
        filter_ctx = FilterContext(kernel=self, metadata={"type": "prompt"})

        # Apply prompt filters
        modified_prompt = self._filters.apply_prompt_rendering(filter_ctx, prompt)

        # Get LLM service
        llm = self.get_llm(model)
        if llm is None:
            # Use easy.ask as fallback
            from ..easy.ask import ask as ask_func

            result = ask_func(modified_prompt)
        else:
            # Use LLM service
            if hasattr(llm, "complete"):
                result = await llm.complete(modified_prompt, **kwargs)
            elif hasattr(llm, "chat"):
                result = await llm.chat([{"role": "user", "content": modified_prompt}], **kwargs)
            elif callable(llm):
                result = llm(modified_prompt, **kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
            else:
                raise ValueError(f"LLM service {llm} not callable")

        # Apply post-prompt filters
        result_str: str = str(result) if not isinstance(result, str) else result
        result_str = self._filters.apply_prompt_rendered(filter_ctx, prompt, result_str)

        return result_str

    def invoke_prompt(self, prompt: str, **kwargs) -> Union[str, "asyncio.Task[str]"]:
        """Invoke a prompt (sync wrapper).

        Args:
            prompt: The prompt to send
            **kwargs: Additional parameters

        Returns:
            LLM response (str if sync, Task if called from async context)
        """
        try:
            asyncio.get_running_loop()
            return asyncio.create_task(self.invoke_prompt_async(prompt, **kwargs))
        except RuntimeError:
            return asyncio.run(self.invoke_prompt_async(prompt, **kwargs))

    # ========== Utilities ==========

    def to_dict(self) -> Dict[str, Any]:
        """Convert kernel state to dictionary.

        Returns:
            Dictionary with kernel state
        """
        return {
            "services": self._services.list_services(),
            "plugins": self._plugins.list_plugins(),
            "agents": list(self._agents.keys()),
            "context": self._context.to_dict(),
        }

    def __repr__(self) -> str:
        return (
            f"Kernel(services={len(self._services.list_services())}, "
            f"plugins={len(self._plugins.list_plugins())}, "
            f"agents={len(self._agents)})"
        )


class KernelBuilder:
    """Builder for creating Kernel instances.

    Provides a fluent API for configuring kernels.

    Example:
        kernel = (
            KernelBuilder()
            .add_service(LLMService("gpt4", client, is_default=True))
            .add_plugin(WeatherPlugin())
            .add_plugin(SearchPlugin())
            .add_filter(LoggingFilter())
            .build()
        )
    """

    def __init__(self):
        """Initialize builder."""
        self._services = ServiceRegistry()
        self._filters = FilterRegistry()

        from ..plugins import PluginRegistry

        self._plugins = PluginRegistry()

    def add_service(self, service: Service) -> "KernelBuilder":
        """Add a service to the kernel.

        Args:
            service: Service descriptor

        Returns:
            Self for chaining
        """
        self._services.add(service)
        return self

    def add_plugin(self, plugin: Any, name: Optional[str] = None) -> "KernelBuilder":
        """Add a plugin to the kernel.

        Args:
            plugin: Plugin instance
            name: Override plugin name

        Returns:
            Self for chaining
        """
        self._plugins.register(plugin, name=name)
        return self

    def add_filter(self, filter_instance: Filter, priority: int = 100) -> "KernelBuilder":
        """Add a filter to the kernel.

        Args:
            filter_instance: Filter instance
            priority: Execution priority

        Returns:
            Self for chaining
        """
        self._filters.add(filter_instance, priority=priority)
        return self

    def build(self) -> Kernel:
        """Build the kernel.

        Returns:
            Configured Kernel instance
        """
        return Kernel(services=self._services, filters=self._filters, plugins=self._plugins)
