# PluginRegistry

The `PluginRegistry` manages plugin discovery, loading, and lifecycle in PyAgent.

## Import

```python
from pyagent.plugins import PluginRegistry
```

## Basic Usage

```python
from pyagent.plugins import PluginRegistry, Plugin

# Create registry
registry = PluginRegistry()

# Register plugins
registry.register(MyPlugin())
registry.register(AnotherPlugin())

# Use with agent
agent = Agent(plugins=registry.get_all())
```

## Class Reference

### Constructor

```python
PluginRegistry(
    auto_discover: bool = False,  # Auto-discover plugins in PYTHONPATH
    plugin_dirs: list[str] = None # Additional directories to scan
)
```

## Registration Methods

### register()

Register a plugin instance.

```python
registry.register(plugin: Plugin) -> None
```

**Example:**
```python
plugin = WeatherPlugin(api_key="xxx")
registry.register(plugin)
```

### register_class()

Register a plugin class for lazy instantiation.

```python
registry.register_class(
    plugin_class: type[Plugin],
    config: dict = None
) -> None
```

**Example:**
```python
registry.register_class(
    WeatherPlugin,
    config={"api_key": "xxx"}
)
```

### unregister()

Remove a plugin from the registry.

```python
registry.unregister(name: str) -> bool
```

**Example:**
```python
if registry.unregister("weather"):
    print("Plugin removed")
```

## Discovery Methods

### discover()

Auto-discover plugins in specified paths.

```python
registry.discover(
    paths: list[str] = None,  # Directories to scan
    pattern: str = "*_plugin.py"  # File pattern
) -> int  # Number of plugins found
```

**Example:**
```python
count = registry.discover(paths=["./plugins"])
print(f"Found {count} plugins")
```

### discover_entry_points()

Load plugins from package entry points.

```python
registry.discover_entry_points(group: str = "pyagent.plugins") -> int
```

**Example:**
```python
# In plugin package's pyproject.toml:
# [project.entry-points."pyagent.plugins"]
# my_plugin = "my_package:MyPlugin"

registry.discover_entry_points()
```

## Query Methods

### get()

Get a plugin by name.

```python
plugin = registry.get(name: str) -> Plugin | None
```

### get_all()

Get all registered plugins.

```python
plugins = registry.get_all() -> list[Plugin]
```

### has()

Check if a plugin is registered.

```python
if registry.has("weather"):
    print("Weather plugin available")
```

### list_names()

Get all plugin names.

```python
names = registry.list_names() -> list[str]
```

### list_functions()

Get all functions across all plugins.

```python
functions = registry.list_functions() -> dict[str, callable]
# Returns {"plugin.function": callable, ...}
```

## Configuration

### configure()

Configure a plugin with settings.

```python
registry.configure(
    name: str,
    settings: dict
) -> None
```

**Example:**
```python
registry.configure("weather", {
    "api_key": "new_key",
    "timeout": 60
})
```

### configure_all()

Configure multiple plugins at once.

```python
registry.configure_all(config: dict) -> None
```

**Example:**
```python
registry.configure_all({
    "weather": {"api_key": "xxx"},
    "search": {"endpoint": "production"}
})
```

## Lifecycle Management

### load_all()

Initialize all registered plugins.

```python
registry.load_all() -> None
```

### unload_all()

Clean up all plugins.

```python
registry.unload_all() -> None
```

## Context Manager

```python
with PluginRegistry(auto_discover=True) as registry:
    agent = Agent(plugins=registry.get_all())
    # ... use agent
# Plugins automatically unloaded
```

## Events

### Callbacks

```python
registry.on_plugin_loaded = lambda plugin: print(f"Loaded: {plugin.name}")
registry.on_plugin_error = lambda plugin, error: print(f"Error: {error}")
```

## Global Registry

For convenience, a global registry is available:

```python
from pyagent.plugins import global_registry

# Register globally
global_registry.register(MyPlugin())

# Access from anywhere
plugin = global_registry.get("my_plugin")
```

## Example: Plugin Directory Structure

```
my_project/
├── plugins/
│   ├── __init__.py
│   ├── weather_plugin.py
│   ├── search_plugin.py
│   └── custom_plugin.py
└── main.py
```

```python
# main.py
from pyagent.plugins import PluginRegistry
from pyagent import Agent

registry = PluginRegistry()
registry.discover(paths=["./plugins"])

agent = Agent(
    name="Assistant",
    plugins=registry.get_all()
)
```

## See Also

- [Plugins-Module](Plugins-Module) - Module overview
- [PluginBase](PluginBase) - Creating plugins
