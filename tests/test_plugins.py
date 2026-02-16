# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Tests for the plugins module.

Tests plugin system including:
- Plugin base class
- Plugin functions
- Plugin registry
- Decorators
"""

import pytest
import tempfile
import os
from pathlib import Path


class TestPluginFunction:
    """Tests for PluginFunction."""
    
    def test_plugin_function_import(self):
        """Test that PluginFunction can be imported."""
        from pyagent.plugins import PluginFunction
        assert PluginFunction is not None
    
    def test_plugin_function_from_function(self):
        """Test creating PluginFunction from a regular function."""
        from pyagent.plugins.base import PluginFunction
        
        def my_func(name: str, count: int = 1) -> str:
            """Say hello."""
            return f"Hello, {name}!" * count
        
        pf = PluginFunction.from_function(my_func)
        
        assert pf.name == "my_func"
        assert "name" in pf.parameters
        assert pf.parameters["name"]["required"] == True
        assert pf.parameters["count"]["default"] == 1
    
    def test_plugin_function_call(self):
        """Test calling a plugin function."""
        from pyagent.plugins.base import PluginFunction
        
        def add(a: int, b: int) -> int:
            return a + b
        
        pf = PluginFunction.from_function(add)
        result = pf(2, 3)
        
        assert result == 5
    
    def test_plugin_function_to_tool_schema(self):
        """Test converting to tool schema."""
        from pyagent.plugins.base import PluginFunction
        
        def search(query: str) -> str:
            """Search for something."""
            return f"Results for: {query}"
        
        pf = PluginFunction.from_function(search)
        schema = pf.to_tool_schema()
        
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert "query" in schema["function"]["parameters"]["properties"]


class TestPlugin:
    """Tests for Plugin class."""
    
    def test_plugin_import(self):
        """Test that Plugin can be imported."""
        from pyagent.plugins import Plugin
        assert Plugin is not None
    
    def test_plugin_creation(self):
        """Test creating a plugin."""
        from pyagent.plugins import Plugin
        from pyagent.plugins.decorators import function
        
        class TestPlugin(Plugin):
            name = "test"
            description = "A test plugin"
            
            @function
            def hello(self, name: str) -> str:
                """Say hello."""
                return f"Hello, {name}!"
        
        plugin = TestPlugin()
        
        assert plugin.name == "test"
        assert "hello" in plugin.list_functions()
    
    def test_plugin_call_function(self):
        """Test calling a function through plugin."""
        from pyagent.plugins import Plugin
        from pyagent.plugins.decorators import function
        
        class MathPlugin(Plugin):
            name = "math"
            
            @function
            def add(self, a: int, b: int) -> int:
                """Add numbers."""
                return a + b
        
        plugin = MathPlugin()
        result = plugin("add", 5, 3)
        
        assert result == 8
    
    def test_plugin_to_tools(self):
        """Test converting plugin to tools."""
        from pyagent.plugins import Plugin
        from pyagent.plugins.decorators import function
        
        class ToolPlugin(Plugin):
            name = "tools"
            
            @function
            def func1(self, x: str) -> str:
                """Function 1."""
                return x
            
            @function
            def func2(self, y: int) -> int:
                """Function 2."""
                return y
        
        plugin = ToolPlugin()
        tools = plugin.to_tools()
        
        assert len(tools) == 2


class TestPluginFromFunctions:
    """Tests for PluginFromFunctions."""
    
    def test_plugin_from_functions(self):
        """Test creating plugin from function list."""
        from pyagent.plugins.base import PluginFromFunctions
        
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b
        
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b
        
        plugin = PluginFromFunctions(
            name="math",
            description="Math operations",
            functions=[add, multiply]
        )
        
        assert plugin.name == "math"
        assert "add" in plugin.list_functions()
        assert "multiply" in plugin.list_functions()


class TestPluginRegistry:
    """Tests for PluginRegistry."""
    
    def test_registry_import(self):
        """Test that PluginRegistry can be imported."""
        from pyagent.plugins import PluginRegistry
        assert PluginRegistry is not None
    
    def test_registry_register(self):
        """Test registering a plugin."""
        from pyagent.plugins import PluginRegistry, Plugin
        from pyagent.plugins.decorators import function
        
        class TestPlugin(Plugin):
            name = "test"
        
        registry = PluginRegistry()
        registry.register(TestPlugin())
        
        assert "test" in registry
        assert len(registry) == 1
    
    def test_registry_get_plugin(self):
        """Test getting a plugin."""
        from pyagent.plugins import PluginRegistry, Plugin
        
        class MyPlugin(Plugin):
            name = "my_plugin"
        
        registry = PluginRegistry()
        plugin = MyPlugin()
        registry.register(plugin)
        
        retrieved = registry.get_plugin("my_plugin")
        assert retrieved is plugin
    
    def test_registry_get_function(self):
        """Test getting a function from registry."""
        from pyagent.plugins import PluginRegistry, Plugin
        from pyagent.plugins.decorators import function
        
        class CalcPlugin(Plugin):
            name = "calc"
            
            @function
            def double(self, x: int) -> int:
                """Double a number."""
                return x * 2
        
        registry = PluginRegistry()
        registry.register(CalcPlugin())
        
        func = registry.get_function("calc", "double")
        assert func is not None
        assert func(5) == 10
    
    def test_registry_call(self):
        """Test calling function through registry."""
        from pyagent.plugins import PluginRegistry, Plugin
        from pyagent.plugins.decorators import function
        
        class StringPlugin(Plugin):
            name = "string"
            
            @function
            def upper(self, text: str) -> str:
                """Uppercase text."""
                return text.upper()
        
        registry = PluginRegistry()
        registry.register(StringPlugin())
        
        result = registry.call("string", "upper", "hello")
        assert result == "HELLO"
    
    def test_registry_list_all_functions(self):
        """Test listing all functions."""
        from pyagent.plugins import PluginRegistry, Plugin
        from pyagent.plugins.decorators import function
        
        class Plugin1(Plugin):
            name = "p1"
            
            @function
            def f1(self):
                pass
        
        class Plugin2(Plugin):
            name = "p2"
            
            @function
            def f2(self):
                pass
            
            @function
            def f3(self):
                pass
        
        registry = PluginRegistry()
        registry.register(Plugin1())
        registry.register(Plugin2())
        
        all_funcs = registry.list_all_functions()
        
        assert "p1" in all_funcs
        assert "p2" in all_funcs
        assert "f1" in all_funcs["p1"]
        assert "f2" in all_funcs["p2"]


class TestDecorators:
    """Tests for plugin decorators."""
    
    def test_function_decorator(self):
        """Test @function decorator."""
        from pyagent.plugins.decorators import function
        
        @function
        def my_func():
            pass
        
        assert my_func._plugin_function == True
    
    def test_function_decorator_with_args(self):
        """Test @function decorator with arguments."""
        from pyagent.plugins.decorators import function
        
        @function(name="custom_name", description="Custom desc")
        def my_func():
            pass
        
        assert my_func._function_name == "custom_name"
        assert my_func._function_description == "Custom desc"
    
    def test_plugin_decorator(self):
        """Test @plugin class decorator."""
        from pyagent.plugins.decorators import plugin
        from pyagent.plugins import Plugin
        
        @plugin(name="decorated", description="A decorated plugin")
        class MyPlugin(Plugin):
            pass
        
        assert MyPlugin.name == "decorated"
        assert MyPlugin.description == "A decorated plugin"


class TestGlobalRegistry:
    """Tests for global registry."""
    
    def test_global_registry_exists(self):
        """Test that global registry exists."""
        from pyagent.plugins import global_registry
        assert global_registry is not None


class TestPluginIntegration:
    """Integration tests for plugins module."""
    
    def test_module_exports(self):
        """Test that all expected exports are available."""
        from pyagent import plugins
        
        assert hasattr(plugins, "Plugin")
        assert hasattr(plugins, "PluginFunction")
        assert hasattr(plugins, "PluginRegistry")
        assert hasattr(plugins, "global_registry")
        assert hasattr(plugins, "function")
        assert hasattr(plugins, "plugin")
    
    def test_main_init_exports(self):
        """Test that plugins is exported from main pyagent."""
        import pyagent
        
        assert hasattr(pyagent, "plugins")
        assert hasattr(pyagent, "Plugin")
        assert hasattr(pyagent, "PluginRegistry")
    
    def test_full_plugin_workflow(self):
        """Test complete plugin workflow."""
        from pyagent.plugins import Plugin, PluginRegistry
        from pyagent.plugins.decorators import function, plugin
        
        @plugin(name="weather", description="Weather functions")
        class WeatherPlugin(Plugin):
            
            @function
            def get_weather(self, city: str) -> str:
                """Get current weather for a city."""
                return f"Weather in {city}: Sunny, 72°F"
            
            @function
            def get_forecast(self, city: str, days: int = 5) -> str:
                """Get weather forecast."""
                return f"Forecast for {city}: Sunny for {days} days"
        
        # Create and register
        registry = PluginRegistry()
        registry.register(WeatherPlugin())
        
        # Call functions
        weather = registry.call("weather", "get_weather", "NYC")
        forecast = registry.call("weather", "get_forecast", "LA", days=3)
        
        assert "NYC" in weather
        assert "LA" in forecast
        assert "3 days" in forecast
        
        # Get tools for agent
        tools = registry.to_tools()
        assert len(tools) == 2
