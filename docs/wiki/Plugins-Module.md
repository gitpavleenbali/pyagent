# Plugins Module

The Plugins module provides a flexible system for extending agent capabilities through reusable, modular components.

## Overview

```python
from pyagent.plugins import Plugin, PluginRegistry, plugin_function
```

## Key Components

| Component | Description |
|-----------|-------------|
| [PluginBase](PluginBase) | Base class for plugins |
| [PluginRegistry](PluginRegistry) | Plugin management and discovery |
| plugin_function | Decorator for plugin functions |

## Quick Start

### Using Built-in Plugins

```python
from pyagent.plugins import PluginRegistry

# Get available plugins
registry = PluginRegistry()
print(registry.list_plugins())

# Load a plugin
calculator = registry.get("calculator")
agent.add_plugin(calculator)
```

### Creating a Simple Plugin

```python
from pyagent.plugins import Plugin, plugin_function

class CalculatorPlugin(Plugin):
    name = "calculator"
    description = "Performs mathematical calculations"
    
    @plugin_function
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
    
    @plugin_function
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

# Register and use
agent.add_plugin(CalculatorPlugin())
```

### Using Decorators

```python
from pyagent.plugins import plugin_function

@plugin_function(name="get_weather", description="Get current weather")
def get_weather(location: str) -> str:
    """Fetch weather for a location."""
    # Implementation
    return f"Weather in {location}: Sunny, 72°F"

# Add to agent
agent.add_function(get_weather)
```

## Plugin Configuration

### With Settings

```python
class DatabasePlugin(Plugin):
    name = "database"
    
    def __init__(self, connection_string: str):
        self.connection = connect(connection_string)
    
    @plugin_function
    def query(self, sql: str) -> list:
        """Execute a SQL query."""
        return self.connection.execute(sql)

# Initialize with config
plugin = DatabasePlugin("postgresql://...")
agent.add_plugin(plugin)
```

### From Configuration File

```yaml
# plugins.yaml
plugins:
  - name: database
    config:
      connection_string: "postgresql://..."
  - name: cache
    config:
      backend: redis
      url: "redis://localhost"
```

```python
registry = PluginRegistry()
registry.load_from_yaml("plugins.yaml")
```

## Built-in Plugins

| Plugin | Description |
|--------|-------------|
| `web_search` | Search the web |
| `calculator` | Math calculations |
| `code_executor` | Execute Python code |
| `file_system` | File operations |
| `http_client` | HTTP requests |

## Plugin Lifecycle

```python
class MyPlugin(Plugin):
    def on_load(self):
        """Called when plugin is loaded."""
        print("Plugin loaded!")
    
    def on_unload(self):
        """Called when plugin is unloaded."""
        print("Plugin unloaded!")
    
    def on_agent_start(self, agent):
        """Called when agent starts."""
        pass
    
    def on_agent_stop(self, agent):
        """Called when agent stops."""
        pass
```

## Enabling/Disabling Functions

```python
@plugin_function(enabled=False)
def experimental_feature():
    """This function is disabled by default."""
    pass

# Enable at runtime
plugin.enable_function("experimental_feature")
```

## See Also

- [PluginBase](PluginBase) - Creating plugins
- [PluginRegistry](PluginRegistry) - Managing plugins
