# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Plugin Loader

Load plugins from files and directories.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import List, Union

from .base import Plugin


def load_plugin(path: Union[str, Path]) -> Plugin:
    """Load a plugin from a Python file.

    The file should define a class that inherits from Plugin.

    Args:
        path: Path to the plugin Python file

    Returns:
        Plugin instance

    Example:
        plugin = load_plugin("plugins/weather_plugin.py")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Plugin file not found: {path}")

    if not path.suffix == ".py":
        raise ValueError(f"Plugin file must be a Python file: {path}")

    # Generate unique module name
    module_name = f"pyai_plugin_{path.stem}"

    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Find Plugin subclass
    plugin_class = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, Plugin) and attr is not Plugin:
            plugin_class = attr
            break

    if plugin_class is None:
        raise ValueError(f"No Plugin subclass found in: {path}")

    # Instantiate and return
    return plugin_class()


def load_plugins_from_dir(directory: Union[str, Path], recursive: bool = False) -> List[Plugin]:
    """Load all plugins from a directory.

    Args:
        directory: Path to directory containing plugin files
        recursive: Whether to search subdirectories

    Returns:
        List of Plugin instances

    Example:
        plugins = load_plugins_from_dir("plugins/")
        for p in plugins:
            print(f"Loaded: {p.name}")
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    plugins = []

    if recursive:
        files = directory.rglob("*.py")
    else:
        files = directory.glob("*.py")

    for file_path in files:
        # Skip __init__.py and other special files
        if file_path.name.startswith("_"):
            continue

        try:
            plugin = load_plugin(file_path)
            plugins.append(plugin)
        except (ImportError, ValueError):
            # Skip files that don't contain valid plugins
            continue

    return plugins


def load_plugin_module(module_path: str) -> Plugin:
    """Load a plugin from an installed Python module.

    Args:
        module_path: Dotted module path (e.g., "mypackage.plugins.weather")

    Returns:
        Plugin instance

    Example:
        plugin = load_plugin_module("myapp.plugins.weather")
    """
    module = importlib.import_module(module_path)

    # Find Plugin subclass
    plugin_class = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, Plugin) and attr is not Plugin:
            plugin_class = attr
            break

    if plugin_class is None:
        raise ValueError(f"No Plugin subclass found in: {module_path}")

    return plugin_class()
