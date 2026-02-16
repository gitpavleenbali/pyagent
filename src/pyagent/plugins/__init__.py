# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Plugins Module

Support for agents and tools as reusable plugins.
Similar to Microsoft Semantic Kernel's plugin pattern.
"""

from .base import Plugin, PluginFunction
from .registry import PluginRegistry, global_registry
from .loader import load_plugin, load_plugins_from_dir
from .decorators import plugin, function

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
