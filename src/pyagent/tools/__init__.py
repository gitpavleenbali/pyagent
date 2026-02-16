# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Tools Module

Tool discovery, management, and utilities.
"""

from .discovery import (
    discover_tools,
    load_tools_from_directory,
    ToolDiscovery,
)
from .watcher import (
    ToolWatcher,
    watch_tools,
)
from .base import (
    Tool,
    ToolResult,
    tool,
)

__all__ = [
    # Discovery
    "discover_tools",
    "load_tools_from_directory",
    "ToolDiscovery",
    # Watcher
    "ToolWatcher",
    "watch_tools",
    # Base
    "Tool",
    "ToolResult",
    "tool",
]
