# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
OpenAPI Parser

Parse OpenAPI/Swagger specifications.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class OpenAPIParameter:
    """A parameter in an OpenAPI operation."""
    name: str
    location: str  # path, query, header, cookie
    description: str = ""
    required: bool = False
    schema_type: str = "string"
    default: Any = None
    enum: Optional[List[str]] = None


@dataclass
class OpenAPIOperation:
    """An operation (endpoint) in the OpenAPI spec."""
    operation_id: str
    method: str
    path: str
    summary: str = ""
    description: str = ""
    parameters: List[OpenAPIParameter] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    @property
    def function_name(self) -> str:
        """Get a valid Python function name."""
        # Convert to snake_case
        name = self.operation_id or f"{self.method}_{self.path}"
        name = name.replace("-", "_").replace(" ", "_")
        name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
        return name.lower().strip("_")


@dataclass
class OpenAPISpec:
    """Parsed OpenAPI specification.
    
    Attributes:
        title: API title
        version: API version
        description: API description
        base_url: Base URL for API calls
        operations: List of operations
        security: Security definitions
    """
    title: str
    version: str
    description: str = ""
    base_url: str = ""
    operations: List[OpenAPIOperation] = field(default_factory=list)
    security: Dict[str, Any] = field(default_factory=dict)
    servers: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_operation(self, operation_id: str) -> Optional[OpenAPIOperation]:
        """Get operation by ID."""
        for op in self.operations:
            if op.operation_id == operation_id:
                return op
        return None
    
    def list_operations(self) -> List[str]:
        """List all operation IDs."""
        return [op.operation_id for op in self.operations]


def parse_openapi(
    source: Union[str, Path, Dict],
    base_url: Optional[str] = None
) -> OpenAPISpec:
    """Parse an OpenAPI specification.
    
    Supports OpenAPI 3.0+ and Swagger 2.0.
    
    Args:
        source: Path to spec file, URL, or dict
        base_url: Override base URL
        
    Returns:
        Parsed OpenAPISpec
        
    Example:
        spec = parse_openapi("api_spec.yaml")
        for op in spec.operations:
            print(f"{op.method} {op.path}: {op.summary}")
    """
    # Load the spec
    if isinstance(source, dict):
        spec_dict = source
    elif isinstance(source, (str, Path)):
        source = Path(source) if isinstance(source, str) else source
        if source.exists():
            content = source.read_text(encoding="utf-8")
            if source.suffix in (".yaml", ".yml"):
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML required for YAML files")
                spec_dict = yaml.safe_load(content)
            else:
                spec_dict = json.loads(content)
        else:
            # Assume it's raw JSON/YAML content
            try:
                spec_dict = json.loads(str(source))
            except json.JSONDecodeError:
                if YAML_AVAILABLE:
                    spec_dict = yaml.safe_load(str(source))
                else:
                    raise ValueError(f"Could not parse: {source}")
    else:
        raise TypeError(f"Invalid source type: {type(source)}")
    
    # Parse info
    info = spec_dict.get("info", {})
    title = info.get("title", "Untitled API")
    version = info.get("version", "1.0.0")
    description = info.get("description", "")
    
    # Determine base URL
    if base_url:
        resolved_base_url = base_url
    elif "servers" in spec_dict:
        # OpenAPI 3.0+
        servers = spec_dict.get("servers", [])
        resolved_base_url = servers[0]["url"] if servers else ""
    elif "host" in spec_dict:
        # Swagger 2.0
        scheme = spec_dict.get("schemes", ["https"])[0]
        host = spec_dict["host"]
        base_path = spec_dict.get("basePath", "")
        resolved_base_url = f"{scheme}://{host}{base_path}"
    else:
        resolved_base_url = ""
    
    # Parse operations
    operations = []
    paths = spec_dict.get("paths", {})
    
    for path, path_item in paths.items():
        for method in ["get", "post", "put", "patch", "delete", "options", "head"]:
            if method not in path_item:
                continue
            
            op_dict = path_item[method]
            
            # Parse parameters
            params = []
            for param in op_dict.get("parameters", []) + path_item.get("parameters", []):
                param_schema = param.get("schema", {})
                params.append(OpenAPIParameter(
                    name=param.get("name", ""),
                    location=param.get("in", "query"),
                    description=param.get("description", ""),
                    required=param.get("required", False),
                    schema_type=param_schema.get("type", param.get("type", "string")),
                    default=param_schema.get("default", param.get("default")),
                    enum=param_schema.get("enum", param.get("enum")),
                ))
            
            # Create operation
            operation = OpenAPIOperation(
                operation_id=op_dict.get("operationId", f"{method}_{path}"),
                method=method.upper(),
                path=path,
                summary=op_dict.get("summary", ""),
                description=op_dict.get("description", ""),
                parameters=params,
                request_body=op_dict.get("requestBody"),
                responses=op_dict.get("responses", {}),
                tags=op_dict.get("tags", []),
            )
            operations.append(operation)
    
    return OpenAPISpec(
        title=title,
        version=version,
        description=description,
        base_url=resolved_base_url,
        operations=operations,
        security=spec_dict.get("securityDefinitions", spec_dict.get("components", {}).get("securitySchemes", {})),
        servers=spec_dict.get("servers", []),
    )
