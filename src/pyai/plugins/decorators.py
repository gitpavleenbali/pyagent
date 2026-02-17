# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Plugin Decorators

Decorators for defining plugins and functions.
"""

from functools import wraps
from typing import Callable, Optional, Type


def function(
    name: Optional[str] = None, description: Optional[str] = None, *, enabled: bool = True
) -> Callable:
    """Decorator to mark a method as a plugin function.

    Use this decorator on methods within a Plugin class to
    expose them as callable functions.

    Args:
        name: Override function name
        description: Override description
        enabled: Whether function is enabled

    Example:
        class MyPlugin(Plugin):
            @function
            def hello(self, name: str) -> str:
                '''Say hello to someone.'''
                return f"Hello, {name}!"

            @function(name="add_numbers", description="Add two nums")
            def add(self, a: int, b: int) -> int:
                return a + b
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Mark as plugin function
        wrapper._plugin_function = True
        wrapper._function_enabled = enabled
        wrapper._function_name = name
        wrapper._function_description = description

        return wrapper

    # Handle both @function and @function() syntax
    if callable(name):
        # Called as @function without parentheses
        func = name
        name = None  # Reset since it's actually the function

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._plugin_function = True
        wrapper._function_enabled = True
        wrapper._function_name = None
        wrapper._function_description = None

        return wrapper

    return decorator


def plugin(
    name: Optional[str] = None, description: Optional[str] = None, version: str = "1.0.0"
) -> Callable[[Type], Type]:
    """Class decorator to define a plugin.

    Automatically sets plugin metadata on the class.

    Args:
        name: Plugin name
        description: Plugin description
        version: Plugin version

    Example:
        @plugin(name="weather", description="Weather functions")
        class WeatherPlugin(Plugin):
            @function
            def get_weather(self, city: str) -> str:
                return f"Weather in {city}: Sunny"
    """

    def decorator(cls: Type) -> Type:
        # Set class attributes
        if name:
            cls.name = name
        elif not hasattr(cls, "name") or not cls.name:
            cls.name = cls.__name__.replace("Plugin", "").lower()

        if description:
            cls.description = description
        elif not hasattr(cls, "description"):
            cls.description = cls.__doc__ or ""

        cls.version = version

        return cls

    return decorator


# Alias for Semantic Kernel compatibility
kernel_function = function
