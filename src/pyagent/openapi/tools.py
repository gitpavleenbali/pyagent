# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
OpenAPI Tools

Generate tools from OpenAPI specifications.
"""

from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
from functools import partial

from .parser import OpenAPISpec, OpenAPIOperation, parse_openapi
from .client import OpenAPIClient


def create_tools_from_openapi(
    spec: Union[str, Path, Dict, OpenAPISpec],
    base_url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    include_operations: Optional[List[str]] = None,
    exclude_operations: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Create tools from an OpenAPI specification.
    
    Automatically generates tool definitions and handlers for each
    operation in the OpenAPI spec.
    
    Args:
        spec: OpenAPI spec (path, dict, or parsed spec)
        base_url: Override base URL
        headers: Default headers for requests
        include_operations: Only include these operation IDs
        exclude_operations: Exclude these operation IDs
        
    Returns:
        List of tool definitions with handlers
        
    Example:
        tools = create_tools_from_openapi("petstore.yaml")
        
        # Use with agent
        agent = Agent(tools=tools)
    """
    # Parse spec if needed
    if isinstance(spec, OpenAPISpec):
        parsed_spec = spec
    else:
        parsed_spec = parse_openapi(spec, base_url)
    
    # Determine base URL
    effective_base_url = base_url or parsed_spec.base_url
    
    # Create client
    client = OpenAPIClient(effective_base_url, headers=headers)
    
    # Generate tools
    tools = []
    
    for operation in parsed_spec.operations:
        # Filter operations
        if include_operations and operation.operation_id not in include_operations:
            continue
        if exclude_operations and operation.operation_id in exclude_operations:
            continue
        
        tool = _create_tool_from_operation(operation, client)
        tools.append(tool)
    
    return tools


def _create_tool_from_operation(
    operation: OpenAPIOperation,
    client: OpenAPIClient
) -> Dict[str, Any]:
    """Create a single tool from an operation."""
    
    # Build parameter schema
    properties = {}
    required = []
    
    for param in operation.parameters:
        prop = {
            "type": _map_openapi_type(param.schema_type),
            "description": param.description or f"The {param.name} parameter",
        }
        
        if param.enum:
            prop["enum"] = param.enum
        
        properties[param.name] = prop
        
        if param.required:
            required.append(param.name)
    
    # Handle request body
    if operation.request_body:
        content = operation.request_body.get("content", {})
        json_content = content.get("application/json", {})
        schema = json_content.get("schema", {})
        
        if schema.get("type") == "object":
            for prop_name, prop_schema in schema.get("properties", {}).items():
                properties[prop_name] = {
                    "type": _map_openapi_type(prop_schema.get("type", "string")),
                    "description": prop_schema.get("description", ""),
                }
            
            for req in schema.get("required", []):
                if req not in required:
                    required.append(req)
    
    # Create handler function
    def handler(**kwargs):
        # Separate path, query, and body params
        path_params = {}
        query_params = {}
        body_params = {}
        
        for param in operation.parameters:
            if param.name in kwargs:
                value = kwargs[param.name]
                if param.location == "path":
                    path_params[param.name] = value
                elif param.location == "query":
                    query_params[param.name] = value
        
        # Remaining kwargs go to body
        for key, value in kwargs.items():
            if key not in path_params and key not in query_params:
                body_params[key] = value
        
        return client.call(
            method=operation.method,
            path=operation.path,
            params=query_params if query_params else None,
            body=body_params if body_params else None,
            path_params=path_params if path_params else None,
        )
    
    # Bind operation to handler
    handler.__name__ = operation.function_name
    handler.__doc__ = operation.description or operation.summary
    
    return {
        "type": "function",
        "function": {
            "name": operation.function_name,
            "description": operation.summary or operation.description or f"{operation.method} {operation.path}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
        "handler": handler,
        "operation": operation,
    }


def _map_openapi_type(openapi_type: str) -> str:
    """Map OpenAPI types to JSON Schema types."""
    mapping = {
        "integer": "integer",
        "number": "number",
        "string": "string",
        "boolean": "boolean",
        "array": "array",
        "object": "object",
    }
    return mapping.get(openapi_type, "string")


def openapi_tool(
    spec_path: Union[str, Path],
    operation_id: str,
    base_url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None
) -> Callable:
    """Decorator to create a tool from a specific OpenAPI operation.
    
    Example:
        @openapi_tool("petstore.yaml", "getPetById")
        def get_pet(pet_id: int):
            '''Get a pet by ID.'''
            pass  # Handler is auto-generated
    """
    def decorator(func: Callable) -> Callable:
        spec = parse_openapi(spec_path, base_url)
        operation = spec.get_operation(operation_id)
        
        if operation is None:
            raise ValueError(f"Operation not found: {operation_id}")
        
        client = OpenAPIClient(base_url or spec.base_url, headers=headers)
        
        def wrapper(**kwargs):
            # Separate params by location
            path_params = {}
            query_params = {}
            body_params = {}
            
            for param in operation.parameters:
                if param.name in kwargs:
                    value = kwargs.pop(param.name)
                    if param.location == "path":
                        path_params[param.name] = value
                    elif param.location == "query":
                        query_params[param.name] = value
            
            body_params = kwargs
            
            return client.call(
                method=operation.method,
                path=operation.path,
                params=query_params if query_params else None,
                body=body_params if body_params else None,
                path_params=path_params if path_params else None,
            )
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__ or operation.summary
        wrapper._openapi_operation = operation
        
        return wrapper
    
    return decorator


class OpenAPITools:
    """A collection of tools from an OpenAPI spec.
    
    Provides easy access to auto-generated tools.
    
    Example:
        api = OpenAPITools("petstore.yaml")
        
        # Get tool definitions for agent
        tools = api.tools
        
        # Call directly
        result = api.call("getPetById", pet_id=123)
    """
    
    def __init__(
        self,
        spec: Union[str, Path, Dict, OpenAPISpec],
        base_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        """Initialize from spec.
        
        Args:
            spec: OpenAPI specification
            base_url: Override base URL
            headers: Default headers
        """
        self._tools = create_tools_from_openapi(spec, base_url, headers)
        self._handlers = {
            t["function"]["name"]: t["handler"]
            for t in self._tools
        }
    
    @property
    def tools(self) -> List[Dict[str, Any]]:
        """Get tool definitions (without handlers)."""
        return [
            {
                "type": t["type"],
                "function": t["function"],
            }
            for t in self._tools
        ]
    
    def call(self, operation_name: str, **kwargs) -> Dict[str, Any]:
        """Call an operation by name.
        
        Args:
            operation_name: Operation/function name
            **kwargs: Operation parameters
            
        Returns:
            API response
        """
        if operation_name not in self._handlers:
            raise ValueError(f"Unknown operation: {operation_name}")
        
        return self._handlers[operation_name](**kwargs)
    
    def list_operations(self) -> List[str]:
        """List available operation names."""
        return list(self._handlers.keys())
