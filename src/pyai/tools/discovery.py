# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Tool Discovery

Automatically discover and load tools from directories.
Like Strands Agents' load_tools_from_directory.
"""

import importlib.util
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Union

from .base import Tool


class ToolDiscovery:
    """Discover and manage tools from directories.

    Example:
        discovery = ToolDiscovery()
        discovery.scan("./my_tools/")
        tools = discovery.get_all_tools()

        # Or watch for changes
        discovery.watch("./my_tools/", on_change=reload_agent)
    """

    def __init__(self):
        """Initialize tool discovery."""
        self._tools: Dict[str, Tool] = {}
        self._watched_dirs: Dict[str, Path] = {}
        self._tool_sources: Dict[str, str] = {}  # tool_name -> source_file

    def scan(
        self, directory: Union[str, Path], recursive: bool = True, pattern: str = "*.py"
    ) -> List[Tool]:
        """Scan directory for tools.

        Args:
            directory: Directory to scan
            recursive: Scan subdirectories
            pattern: File pattern to match

        Returns:
            List of discovered tools
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        discovered = []

        # Find Python files
        if recursive:
            files = directory.rglob(pattern)
        else:
            files = directory.glob(pattern)

        for file_path in files:
            # Skip __pycache__ and __init__
            if "__pycache__" in str(file_path):
                continue
            if file_path.name == "__init__.py":
                continue

            tools = self._load_tools_from_file(file_path)
            discovered.extend(tools)

        return discovered

    def _load_tools_from_file(self, file_path: Path) -> List[Tool]:
        """Load tools from a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            List of tools found in file
        """
        tools = []

        try:
            # Load module from file
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if spec is None or spec.loader is None:
                return []

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find tools in module
            for name, obj in inspect.getmembers(module):
                # Skip private/dunder names
                if name.startswith("_"):
                    continue

                # Check if it's already a Tool
                if isinstance(obj, Tool):
                    self._tools[obj.name] = obj
                    self._tool_sources[obj.name] = str(file_path)
                    tools.append(obj)

                # Check if it's a decorated function with tool marker
                elif callable(obj) and hasattr(obj, "_is_tool"):
                    tool = Tool.from_function(obj)
                    self._tools[tool.name] = tool
                    self._tool_sources[tool.name] = str(file_path)
                    tools.append(tool)

                # Check if it's a function that looks like a tool
                elif callable(obj) and inspect.isfunction(obj) and obj.__doc__ is not None:
                    # Has docstring, could be a tool
                    tool = Tool.from_function(obj)
                    self._tools[tool.name] = tool
                    self._tool_sources[tool.name] = str(file_path)
                    tools.append(tool)

        except Exception as e:
            # Log but don't fail on individual file errors
            print(f"Warning: Error loading tools from {file_path}: {e}")

        return tools

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all_tools(self) -> List[Tool]:
        """Get all discovered tools."""
        return list(self._tools.values())

    def list_tools(self) -> List[str]:
        """List all tool names."""
        return list(self._tools.keys())

    def reload(self, tool_name: Optional[str] = None) -> List[Tool]:
        """Reload tools from their source files.

        Args:
            tool_name: Specific tool to reload, or None for all

        Returns:
            List of reloaded tools
        """
        reloaded = []

        if tool_name:
            source = self._tool_sources.get(tool_name)
            if source:
                tools = self._load_tools_from_file(Path(source))
                reloaded.extend(tools)
        else:
            # Reload all
            sources = set(self._tool_sources.values())
            for source in sources:
                tools = self._load_tools_from_file(Path(source))
                reloaded.extend(tools)

        return reloaded

    def remove(self, tool_name: str) -> bool:
        """Remove a tool."""
        if tool_name in self._tools:
            del self._tools[tool_name]
            if tool_name in self._tool_sources:
                del self._tool_sources[tool_name]
            return True
        return False

    def clear(self):
        """Clear all discovered tools."""
        self._tools.clear()
        self._tool_sources.clear()


def discover_tools(
    directory: Union[str, Path], recursive: bool = True, pattern: str = "*.py"
) -> List[Tool]:
    """Discover tools from a directory.

    Simple function interface for tool discovery.

    Args:
        directory: Directory to scan
        recursive: Scan subdirectories
        pattern: File pattern to match

    Returns:
        List of discovered tools

    Example:
        tools = discover_tools("./my_tools/")
        agent = Agent("helper", tools=tools)
    """
    discovery = ToolDiscovery()
    return discovery.scan(directory, recursive, pattern)


def load_tools_from_directory(
    directory: Union[str, Path],
    recursive: bool = True,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> List[Tool]:
    """Load all tools from a directory.

    Compatible with Strands Agents pattern.

    Args:
        directory: Directory to load from
        recursive: Scan subdirectories
        include_patterns: Only include matching files
        exclude_patterns: Exclude matching files

    Returns:
        List of tools

    Example:
        tools = load_tools_from_directory(
            "./tools/",
            exclude_patterns=["test_*", "*_old.py"]
        )
    """
    directory = Path(directory)
    discovery = ToolDiscovery()

    all_tools = []
    patterns = include_patterns or ["*.py"]

    for pattern in patterns:
        tools = discovery.scan(directory, recursive, pattern)
        all_tools.extend(tools)

    # Apply exclusions
    if exclude_patterns:
        import fnmatch

        filtered = []
        for tool in all_tools:
            source = discovery._tool_sources.get(tool.name, "")
            exclude = False
            for excl_pattern in exclude_patterns:
                if fnmatch.fnmatch(Path(source).name, excl_pattern):
                    exclude = True
                    break
            if not exclude:
                filtered.append(tool)
        all_tools = filtered

    return all_tools
