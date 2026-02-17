# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Plugins Module

Support for agents and tools as reusable plugins.
Similar to Microsoft Semantic Kernel's plugin pattern.
"""

from .base import Plugin, PluginFunction
from .decorators import function, plugin
from .loader import load_plugin, load_plugins_from_dir
from .registry import PluginRegistry, global_registry

__all__ = [
    "Plugin",
    "PluginFunction",
    "PluginRegistry",
    "global_registry",
    "load_plugin",
    "load_plugins_from_dir",
    "plugin",
    "function",
]
