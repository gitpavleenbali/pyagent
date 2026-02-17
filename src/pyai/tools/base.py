# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Tool Base Classes

Base classes and decorators for creating tools.
"""

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union


@dataclass
class ToolResult:
    """Result of a tool execution.

    Attributes:
        success: Whether the tool executed successfully
        data: The result data
        error: Error message if failed
        metadata: Additional metadata
    """

    success: bool = True
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.success:
            return str(self.data)
        return f"Error: {self.error}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class Tool:
    """A callable tool for agents.

    Attributes:
        name: Tool name
        description: Tool description
        func: The callable function
        parameters: Parameter schema
        returns: Return type description
    """

    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    returns: str = ""
    tags: List[str] = field(default_factory=list)

    def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool."""
        return self.func(*args, **kwargs)

    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool and wrap result in ToolResult.

        Args:
            **kwargs: Arguments to pass to the tool function

        Returns:
            ToolResult with success status and data
        """
        try:
            result = self.func(**kwargs)
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
                or {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "returns": self.returns,
            "tags": self.tags,
        }

    @classmethod
    def from_function(
        cls,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> "Tool":
        """Create a Tool from a function.

        Args:
            func: The function to wrap
            name: Override function name
            description: Override docstring
            tags: Tags for categorization

        Returns:
            Tool instance
        """
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"Tool: {tool_name}"

        # Extract parameter schema from type hints
        sig = inspect.signature(func)
        hints = getattr(func, "__annotations__", {})

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = hints.get(param_name, str)
            json_type = _python_type_to_json(param_type)

            properties[param_name] = {
                "type": json_type,
                "description": f"The {param_name} parameter",
            }

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        return cls(
            name=tool_name,
            description=tool_desc.strip(),
            func=func,
            parameters=parameters,
            returns=str(hints.get("return", "")),
            tags=tags or [],
        )


def tool(
    name: Optional[str] = None, description: Optional[str] = None, tags: Optional[List[str]] = None
) -> Callable:
    """Decorator to create a tool from a function.

    Args:
        name: Override function name
        description: Override docstring
        tags: Tags for categorization

    Returns:
        Decorated function as a Tool

    Example:
        @tool(name="get_weather", description="Get weather info")
        def weather(city: str) -> str:
            return f"Weather in {city}: sunny"
    """

    def decorator(func: Callable) -> Tool:
        return Tool.from_function(func, name, description, tags)

    # Handle @tool without parentheses
    if callable(name):
        func = name
        return Tool.from_function(func, None, None, None)

    return decorator


def _python_type_to_json(py_type: type) -> str:
    """Convert Python type to JSON schema type."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    # Handle typing module types
    origin = getattr(py_type, "__origin__", None)
    if origin is not None:
        if origin in (list, List):
            return "array"
        if origin in (dict, Dict):
            return "object"
        if origin is Union:
            return "string"  # Simplified

    return type_map.get(py_type, "string")
