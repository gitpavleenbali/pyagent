# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Tool Watcher

Watch directories for tool changes and hot-reload.
"""

import os
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Union

from .base import Tool
from .discovery import ToolDiscovery


class ToolWatcher:
    """Watch directories for tool changes.
    
    Monitors tool files and triggers reload on changes.
    
    Example:
        watcher = ToolWatcher("./my_tools/")
        watcher.on_change = lambda tools: print(f"Reloaded {len(tools)} tools")
        watcher.start()
        
        # Later...
        watcher.stop()
    """
    
    def __init__(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        poll_interval: float = 1.0,
        on_change: Optional[Callable[[List[Tool]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None
    ):
        """Initialize watcher.
        
        Args:
            directory: Directory to watch
            recursive: Watch subdirectories
            poll_interval: Seconds between checks
            on_change: Callback when tools change
            on_error: Callback on errors
        """
        self.directory = Path(directory)
        self.recursive = recursive
        self.poll_interval = poll_interval
        self.on_change = on_change
        self.on_error = on_error
        
        self._discovery = ToolDiscovery()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._file_mtimes: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def start(self, daemon: bool = True):
        """Start watching for changes.
        
        Args:
            daemon: Run as daemon thread (stops with main thread)
        """
        if self._running:
            return
        
        self._running = True
        
        # Initial scan
        self._scan_files()
        self._discovery.scan(self.directory, self.recursive)
        
        # Start watch thread
        self._thread = threading.Thread(target=self._watch_loop, daemon=daemon)
        self._thread.start()
    
    def stop(self):
        """Stop watching."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
    
    def _watch_loop(self):
        """Main watch loop."""
        while self._running:
            try:
                changed_files = self._check_changes()
                
                if changed_files:
                    # Reload changed files
                    new_tools = []
                    for file_path in changed_files:
                        tools = self._discovery._load_tools_from_file(Path(file_path))
                        new_tools.extend(tools)
                    
                    # Trigger callback
                    if self.on_change and new_tools:
                        try:
                            self.on_change(new_tools)
                        except Exception as e:
                            if self.on_error:
                                self.on_error(e)
            
            except Exception as e:
                if self.on_error:
                    self.on_error(e)
            
            time.sleep(self.poll_interval)
    
    def _scan_files(self) -> Set[str]:
        """Scan directory for Python files."""
        files = set()
        
        if self.recursive:
            for file_path in self.directory.rglob("*.py"):
                if "__pycache__" not in str(file_path):
                    files.add(str(file_path))
                    self._file_mtimes[str(file_path)] = os.path.getmtime(file_path)
        else:
            for file_path in self.directory.glob("*.py"):
                files.add(str(file_path))
                self._file_mtimes[str(file_path)] = os.path.getmtime(file_path)
        
        return files
    
    def _check_changes(self) -> List[str]:
        """Check for file changes.
        
        Returns:
            List of changed file paths
        """
        changed = []
        current_files = set()
        
        # Scan current files
        if self.recursive:
            iterator = self.directory.rglob("*.py")
        else:
            iterator = self.directory.glob("*.py")
        
        for file_path in iterator:
            if "__pycache__" in str(file_path):
                continue
            
            file_str = str(file_path)
            current_files.add(file_str)
            
            try:
                mtime = os.path.getmtime(file_path)
                
                if file_str not in self._file_mtimes:
                    # New file
                    self._file_mtimes[file_str] = mtime
                    changed.append(file_str)
                elif mtime > self._file_mtimes[file_str]:
                    # Modified file
                    self._file_mtimes[file_str] = mtime
                    changed.append(file_str)
            
            except OSError:
                # File might have been deleted
                pass
        
        # Check for deleted files
        with self._lock:
            deleted = set(self._file_mtimes.keys()) - current_files
            for file_str in deleted:
                del self._file_mtimes[file_str]
        
        return changed
    
    @property
    def tools(self) -> List[Tool]:
        """Get current tools."""
        return self._discovery.get_all_tools()
    
    @property
    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running


def watch_tools(
    directory: Union[str, Path],
    on_change: Callable[[List[Tool]], None],
    recursive: bool = True,
    poll_interval: float = 1.0,
    background: bool = True
) -> ToolWatcher:
    """Watch a directory for tool changes.
    
    Simple function interface for tool watching.
    
    Args:
        directory: Directory to watch
        on_change: Callback when tools change
        recursive: Watch subdirectories
        poll_interval: Seconds between checks
        background: Start in background
        
    Returns:
        ToolWatcher instance
        
    Example:
        def reload_agent(tools):
            agent.tools = tools
            print(f"Reloaded {len(tools)} tools")
        
        watcher = watch_tools("./my_tools/", on_change=reload_agent)
        # watcher.stop() when done
    """
    watcher = ToolWatcher(
        directory=directory,
        recursive=recursive,
        poll_interval=poll_interval,
        on_change=on_change
    )
    
    if background:
        watcher.start(daemon=True)
    
    return watcher
