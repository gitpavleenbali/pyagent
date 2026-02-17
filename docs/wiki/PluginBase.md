# PluginBase

The `Plugin` base class provides the foundation for creating custom agent plugins.

## Import

```python
from pyagent.plugins import Plugin, plugin_function
```

## Creating a Plugin

### Basic Structure

```python
from pyagent.plugins import Plugin, plugin_function

class MyPlugin(Plugin):
    name = "my_plugin"
    description = "Description of what the plugin does"
    version = "1.0.0"
    
    @plugin_function
    def my_function(self, arg1: str, arg2: int = 10) -> str:
        """Function documentation becomes the tool description."""
        return f"Result: {arg1} x {arg2}"
```

### Class Attributes

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | str | Yes | Unique plugin identifier |
| `description` | str | Yes | Plugin description |
| `version` | str | No | Version string |
| `author` | str | No | Author name |
| `dependencies` | list | No | Required packages |

## Plugin Functions

### The @plugin_function Decorator

```python
from pyagent.plugins import plugin_function

@plugin_function
def search(query: str, limit: int = 10) -> list[str]:
    """Search for items matching the query.
    
    Args:
        query: The search query
        limit: Maximum results to return
    
    Returns:
        List of matching items
    """
    return ["item1", "item2"]
```

### Decorator Options

```python
@plugin_function(
    name="custom_name",              # Override function name
    description="Custom description", # Override docstring
    enabled=True,                     # Enable/disable
    requires_auth=False               # Require authentication
)
def my_function():
    pass
```

## Lifecycle Methods

```python
class MyPlugin(Plugin):
    name = "my_plugin"
    
    def on_load(self):
        """Called when plugin is loaded into agent."""
        self.setup_connections()
    
    def on_unload(self):
        """Called when plugin is removed from agent."""
        self.cleanup_connections()
    
    def on_agent_start(self, agent):
        """Called when the agent starts running."""
        pass
    
    def on_agent_stop(self, agent):
        """Called when the agent stops."""
        pass
    
    def on_function_call(self, function_name: str, args: dict):
        """Called before a function is executed."""
        logging.info(f"Calling {function_name}")
    
    def on_function_result(self, function_name: str, result):
        """Called after a function returns."""
        logging.info(f"{function_name} returned: {result}")
```

## Configuration

### Constructor Configuration

```python
class ConfigurablePlugin(Plugin):
    name = "configurable"
    
    def __init__(self, api_key: str, endpoint: str = "default"):
        super().__init__()
        self.api_key = api_key
        self.endpoint = endpoint
    
    @plugin_function
    def call_api(self, data: dict) -> dict:
        # Use self.api_key and self.endpoint
        pass

# Usage
plugin = ConfigurablePlugin(api_key="xxx", endpoint="production")
agent.add_plugin(plugin)
```

### Settings Schema

```python
class MyPlugin(Plugin):
    name = "my_plugin"
    
    settings_schema = {
        "api_key": {"type": "string", "required": True},
        "timeout": {"type": "integer", "default": 30}
    }
    
    def configure(self, settings: dict):
        """Called with validated settings."""
        self.api_key = settings["api_key"]
        self.timeout = settings.get("timeout", 30)
```

## Async Functions

```python
class AsyncPlugin(Plugin):
    name = "async_plugin"
    
    @plugin_function
    async def fetch_data(self, url: str) -> dict:
        """Fetch data asynchronously."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()
```

## Using with Agents

```python
from pyagent import Agent

# Create plugin instance
plugin = MyPlugin()

# Add to agent
agent = Agent(
    name="Assistant",
    plugins=[plugin]
)

# Or add later
agent.add_plugin(plugin)
```

## See Also

- [Plugins-Module](Plugins-Module) - Module overview
- [PluginRegistry](PluginRegistry) - Plugin management
