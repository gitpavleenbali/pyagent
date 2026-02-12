"""
mcp - Model Context Protocol server support

Create and connect to MCP servers for tool discovery and execution.
Inspired by Strands' MCP support but with PyAgent's simplicity.

Examples:
    >>> from pyagent import mcp
    
    # Create an MCP server with tools
    >>> @mcp.tool("calculator")
    ... def add(a: int, b: int) -> int:
    ...     '''Add two numbers'''
    ...     return a + b
    
    # Start server
    >>> server = mcp.server("my-tools", tools=[add])
    >>> server.start()
    
    # Or use as context
    >>> with mcp.server("tools") as server:
    ...     # Server running
    ...     pass
    
    # Connect to external MCP server
    >>> tools = mcp.connect("http://localhost:8080")
    >>> result = tools.call("calculator", a=1, b=2)
"""

from typing import Callable, Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from functools import wraps
import json
import inspect
import asyncio
from concurrent.futures import ThreadPoolExecutor


@dataclass
class MCPTool:
    """A tool exposed via MCP."""
    
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to MCP tool schema."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": self.parameters,
                "required": list(self.parameters.keys())
            }
        }
    
    def __call__(self, **kwargs) -> Any:
        """Execute the tool."""
        return self.handler(**kwargs)


@dataclass
class MCPServer:
    """An MCP server that exposes tools."""
    
    name: str
    version: str = "1.0.0"
    tools: List[MCPTool] = field(default_factory=list)
    _running: bool = False
    _executor: ThreadPoolExecutor = None
    
    def add_tool(self, tool: MCPTool) -> None:
        """Add a tool to the server."""
        self.tools.append(tool)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        return [t.to_schema() for t in self.tools]
    
    def call_tool(self, tool_name: str, **kwargs) -> Any:
        """Call a tool by name."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool(**kwargs)
        raise ValueError(f"Tool not found: {tool_name}")
    
    def start(self, port: int = 8080, background: bool = True) -> None:
        """Start the MCP server."""
        self._running = True
        print(f"MCP Server '{self.name}' v{self.version} starting on port {port}")
        print(f"Tools available: {[t.name for t in self.tools]}")
        
        if not background:
            # Block and serve (for production use)
            self._serve(port)
    
    def stop(self) -> None:
        """Stop the MCP server."""
        self._running = False
        if self._executor:
            self._executor.shutdown(wait=False)
        print(f"MCP Server '{self.name}' stopped")
    
    def _serve(self, port: int) -> None:
        """Internal serve loop (simplified)."""
        # In production, this would use proper HTTP/WebSocket server
        # For now, this is a simplified in-process implementation
        pass
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def __repr__(self) -> str:
        return f"MCPServer(name='{self.name}', tools={len(self.tools)})"


@dataclass
class MCPClient:
    """Client to connect to MCP servers."""
    
    endpoint: str
    tools: List[Dict[str, Any]] = field(default_factory=list)
    _server: MCPServer = None  # For in-process servers
    
    def __post_init__(self):
        if isinstance(self.endpoint, MCPServer):
            self._server = self.endpoint
            self.tools = self._server.list_tools()
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the server."""
        if self._server:
            return self._server.list_tools()
        # For remote servers, would make HTTP call
        return self.tools
    
    def call(self, tool_name: str, **kwargs) -> Any:
        """Call a tool on the server."""
        if self._server:
            return self._server.call_tool(tool_name, **kwargs)
        # For remote servers, would make HTTP call
        raise NotImplementedError("Remote MCP calls not yet implemented")
    
    def __repr__(self) -> str:
        return f"MCPClient(endpoint='{self.endpoint}', tools={len(self.tools)})"


def tool(
    name: str = None,
    description: str = None
) -> Callable:
    """
    Decorator to create an MCP tool from a function.
    
    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        
    Examples:
        >>> @mcp.tool("greet")
        ... def greet_user(name: str) -> str:
        ...     '''Greet a user by name'''
        ...     return f"Hello, {name}!"
    """
    def decorator(func: Callable) -> MCPTool:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"Tool: {tool_name}"
        
        # Extract parameters from function signature
        sig = inspect.signature(func)
        params = {}
        for param_name, param in sig.parameters.items():
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list:
                    param_type = "array"
                elif param.annotation == dict:
                    param_type = "object"
            
            params[param_name] = {"type": param_type}
        
        return MCPTool(
            name=tool_name,
            description=tool_desc.strip(),
            parameters=params,
            handler=func
        )
    
    # Handle @mcp.tool vs @mcp.tool("name")
    if callable(name):
        func = name
        name = None
        return decorator(func)
    
    return decorator


def server(
    name: str,
    *,
    version: str = "1.0.0",
    tools: List[Union[MCPTool, Callable]] = None
) -> MCPServer:
    """
    Create an MCP server.
    
    Args:
        name: Server name
        version: Server version
        tools: List of tools to expose
        
    Returns:
        MCPServer instance
        
    Examples:
        >>> server = mcp.server("my-tools", tools=[add, subtract])
        >>> server.start()
    """
    mcp_tools = []
    for t in (tools or []):
        if isinstance(t, MCPTool):
            mcp_tools.append(t)
        elif callable(t):
            # Auto-convert function to MCPTool
            mcp_tools.append(tool()(t))
    
    return MCPServer(name=name, version=version, tools=mcp_tools)


def connect(endpoint: Union[str, MCPServer]) -> MCPClient:
    """
    Connect to an MCP server.
    
    Args:
        endpoint: Server URL or MCPServer instance
        
    Returns:
        MCPClient for calling tools
        
    Examples:
        >>> client = mcp.connect("http://localhost:8080")
        >>> result = client.call("calculator", a=1, b=2)
        
        >>> # Or connect to in-process server
        >>> client = mcp.connect(my_server)
    """
    return MCPClient(endpoint=endpoint)


def from_directory(
    path: str,
    *,
    pattern: str = "*.py",
    server_name: str = None
) -> MCPServer:
    """
    Create an MCP server from tool definitions in a directory.
    
    Looks for functions decorated with @mcp.tool in Python files.
    
    Args:
        path: Directory path
        pattern: File pattern to match
        server_name: Server name
        
    Returns:
        MCPServer with discovered tools
        
    Examples:
        >>> server = mcp.from_directory("./tools")
        >>> server.start()
    """
    import glob
    import importlib.util
    from pathlib import Path
    
    tools = []
    dir_path = Path(path)
    
    for file_path in dir_path.glob(pattern):
        # Load the module
        spec = importlib.util.spec_from_file_location(
            file_path.stem, 
            file_path
        )
        module = importlib.util.module_from_spec(spec)
        
        try:
            spec.loader.exec_module(module)
            
            # Find MCPTool instances
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, MCPTool):
                    tools.append(attr)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    
    return MCPServer(
        name=server_name or dir_path.name,
        tools=tools
    )


class MCPModule:
    """MCP module with all functions attached."""
    
    tool = staticmethod(tool)
    server = staticmethod(server)
    connect = staticmethod(connect)
    from_directory = staticmethod(from_directory)
    
    # Classes
    Tool = MCPTool
    Server = MCPServer
    Client = MCPClient


# Create module-level instance
mcp = MCPModule()
