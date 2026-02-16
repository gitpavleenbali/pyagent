# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Plugin Base Classes

Core classes for the plugin system.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type
import inspect


@dataclass
class PluginFunction:
    """A function within a plugin.
    
    Represents a single callable function that can be used as a tool
    or action within an agent.
    
    Attributes:
        name: Function name
        description: What the function does
        func: The actual callable
        parameters: Parameter definitions
        returns: Return type description
        is_async: Whether function is async
    """
    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    returns: Optional[str] = None
    is_async: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_function(
        cls,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> "PluginFunction":
        """Create a PluginFunction from a Python function.
        
        Automatically extracts parameters and docstring.
        
        Args:
            func: The function to wrap
            name: Override function name
            description: Override description
            
        Returns:
            PluginFunction instance
        """
        func_name = name or func.__name__
        func_doc = description or func.__doc__ or ""
        
        # Extract first line of docstring as description
        if func_doc:
            func_doc = func_doc.strip().split("\n")[0]
        
        # Get parameters from signature
        sig = inspect.signature(func)
        parameters = {}
        
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            
            param_info = {"type": "any"}
            
            # Try to get type annotation
            if param.annotation != inspect.Parameter.empty:
                type_name = getattr(param.annotation, "__name__", str(param.annotation))
                param_info["type"] = type_name
            
            # Check if required
            if param.default == inspect.Parameter.empty:
                param_info["required"] = True
            else:
                param_info["default"] = param.default
            
            parameters[param_name] = param_info
        
        # Check if async
        is_async = inspect.iscoroutinefunction(func)
        
        return cls(
            name=func_name,
            description=func_doc,
            func=func,
            parameters=parameters,
            is_async=is_async,
        )
    
    def __call__(self, *args, **kwargs) -> Any:
        """Call the underlying function."""
        return self.func(*args, **kwargs)
    
    async def call_async(self, *args, **kwargs) -> Any:
        """Call the function asynchronously."""
        if self.is_async:
            return await self.func(*args, **kwargs)
        else:
            return self.func(*args, **kwargs)
    
    def to_tool_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        
        for param_name, param_info in self.parameters.items():
            prop = {"type": param_info.get("type", "string")}
            if param_info.get("description"):
                prop["description"] = param_info["description"]
            properties[param_name] = prop
            
            if param_info.get("required", False):
                required.append(param_name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class Plugin:
    """A collection of related functions.
    
    Plugins group related functionality together, similar to
    Microsoft Semantic Kernel's plugin pattern.
    
    Example:
        class WeatherPlugin(Plugin):
            name = "weather"
            description = "Weather-related functions"
            
            @function
            def get_current_weather(self, city: str) -> str:
                '''Get current weather for a city.'''
                return f"Weather in {city}: Sunny, 72°F"
            
            @function
            def get_forecast(self, city: str, days: int = 5) -> str:
                '''Get weather forecast.'''
                return f"Forecast for {city}: Sunny for {days} days"
        
        # Register and use
        plugin = WeatherPlugin()
        registry.register(plugin)
    """
    
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    
    def __init__(self):
        """Initialize plugin and discover functions."""
        self._functions: Dict[str, PluginFunction] = {}
        self._discover_functions()
    
    def _discover_functions(self) -> None:
        """Discover decorated functions in this plugin."""
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            
            attr = getattr(self, attr_name)
            
            # Check if decorated with @function
            if hasattr(attr, "_plugin_function"):
                plugin_func = PluginFunction.from_function(
                    attr,
                    name=getattr(attr, "_function_name", attr_name),
                    description=getattr(attr, "_function_description", None),
                )
                self._functions[plugin_func.name] = plugin_func
    
    @property
    def functions(self) -> Dict[str, PluginFunction]:
        """Get all functions in this plugin."""
        return self._functions
    
    def get_function(self, name: str) -> Optional[PluginFunction]:
        """Get a specific function by name."""
        return self._functions.get(name)
    
    def list_functions(self) -> List[str]:
        """List all function names."""
        return list(self._functions.keys())
    
    def add_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> None:
        """Manually add a function to the plugin.
        
        Args:
            func: Function to add
            name: Override name
            description: Override description
        """
        plugin_func = PluginFunction.from_function(func, name, description)
        self._functions[plugin_func.name] = plugin_func
    
    def __call__(self, function_name: str, *args, **kwargs) -> Any:
        """Call a function by name."""
        func = self.get_function(function_name)
        if func is None:
            raise ValueError(f"Function not found: {function_name}")
        return func(*args, **kwargs)
    
    def to_tools(self) -> List[Dict[str, Any]]:
        """Convert all functions to tool schemas."""
        return [func.to_tool_schema() for func in self._functions.values()]


class PluginFromFunctions(Plugin):
    """Create a plugin from a collection of functions.
    
    Example:
        def add(a: int, b: int) -> int:
            '''Add two numbers.'''
            return a + b
        
        def multiply(a: int, b: int) -> int:
            '''Multiply two numbers.'''
            return a * b
        
        math_plugin = PluginFromFunctions(
            name="math",
            description="Math operations",
            functions=[add, multiply]
        )
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        functions: List[Callable] = None,
        version: str = "1.0.0"
    ):
        """Initialize plugin with functions.
        
        Args:
            name: Plugin name
            description: Plugin description
            functions: List of functions to include
            version: Plugin version
        """
        self.name = name
        self.description = description
        self.version = version
        self._functions = {}
        
        # Add provided functions
        for func in (functions or []):
            self.add_function(func)
    
    def _discover_functions(self) -> None:
        """Override - don't auto-discover for this class."""
        pass
