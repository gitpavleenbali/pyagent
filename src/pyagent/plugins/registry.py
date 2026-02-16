# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Plugin Registry

Central registry for managing plugins.
"""

from typing import Any, Callable, Dict, List, Optional, Type
from .base import Plugin, PluginFunction


class PluginRegistry:
    """Registry for plugins.
    
    Provides centralized management of plugins with lookup,
    registration, and discovery capabilities.
    
    Example:
        registry = PluginRegistry()
        
        # Register a plugin
        registry.register(WeatherPlugin())
        
        # Get a function
        func = registry.get_function("weather", "get_weather")
        result = func(city="NYC")
        
        # List all plugins
        for name in registry.list_plugins():
            print(f"Plugin: {name}")
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._plugins: Dict[str, Plugin] = {}
    
    def register(self, plugin: Plugin, name: Optional[str] = None) -> None:
        """Register a plugin.
        
        Args:
            plugin: Plugin instance to register
            name: Override plugin name
        """
        plugin_name = name or plugin.name or plugin.__class__.__name__
        self._plugins[plugin_name] = plugin
    
    def unregister(self, name: str) -> bool:
        """Unregister a plugin.
        
        Args:
            name: Plugin name to unregister
            
        Returns:
            True if plugin was removed
        """
        if name in self._plugins:
            del self._plugins[name]
            return True
        return False
    
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None
        """
        return self._plugins.get(name)
    
    def get_function(
        self,
        plugin_name: str,
        function_name: str
    ) -> Optional[PluginFunction]:
        """Get a function from a plugin.
        
        Args:
            plugin_name: Name of the plugin
            function_name: Name of the function
            
        Returns:
            PluginFunction or None
        """
        plugin = self.get_plugin(plugin_name)
        if plugin:
            return plugin.get_function(function_name)
        return None
    
    def list_plugins(self) -> List[str]:
        """List all registered plugin names."""
        return list(self._plugins.keys())
    
    def list_all_functions(self) -> Dict[str, List[str]]:
        """List all functions grouped by plugin.
        
        Returns:
            Dict mapping plugin names to function names
        """
        result = {}
        for name, plugin in self._plugins.items():
            result[name] = plugin.list_functions()
        return result
    
    def call(
        self,
        plugin_name: str,
        function_name: str,
        *args,
        **kwargs
    ) -> Any:
        """Call a function by plugin and function name.
        
        Args:
            plugin_name: Plugin name
            function_name: Function name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        func = self.get_function(plugin_name, function_name)
        if func is None:
            raise ValueError(
                f"Function not found: {plugin_name}.{function_name}"
            )
        return func(*args, **kwargs)
    
    def to_tools(self) -> List[Dict[str, Any]]:
        """Convert all plugin functions to tool schemas.
        
        Returns:
            List of tool schemas for all functions
        """
        tools = []
        for plugin in self._plugins.values():
            tools.extend(plugin.to_tools())
        return tools
    
    def __contains__(self, name: str) -> bool:
        """Check if plugin exists."""
        return name in self._plugins
    
    def __len__(self) -> int:
        """Get number of registered plugins."""
        return len(self._plugins)


# Global default registry
global_registry = PluginRegistry()


def register_plugin(plugin: Plugin, name: Optional[str] = None) -> None:
    """Register a plugin to the global registry.
    
    Convenience function for quick registration.
    
    Args:
        plugin: Plugin to register
        name: Optional name override
    """
    global_registry.register(plugin, name)


def get_plugin(name: str) -> Optional[Plugin]:
    """Get a plugin from the global registry.
    
    Args:
        name: Plugin name
        
    Returns:
        Plugin or None
    """
    return global_registry.get_plugin(name)


def call_function(
    plugin_name: str,
    function_name: str,
    *args,
    **kwargs
) -> Any:
    """Call a function from the global registry.
    
    Args:
        plugin_name: Plugin name
        function_name: Function name
        *args: Arguments
        **kwargs: Keyword arguments
        
    Returns:
        Function result
    """
    return global_registry.call(plugin_name, function_name, *args, **kwargs)
