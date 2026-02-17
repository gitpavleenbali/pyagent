# OpenAPIParser

The `OpenAPIParser` class parses OpenAPI/Swagger specifications and extracts operation details.

## Import

```python
from pyagent.openapi import OpenAPIParser
```

## Constructor

```python
OpenAPIParser(
    spec: str | dict,           # Spec URL, path, or dict
    validate: bool = True       # Validate spec structure
)
```

## Basic Usage

### Parse Specification

```python
# From URL
parser = OpenAPIParser("https://api.example.com/openapi.json")

# From file
parser = OpenAPIParser("./specs/api.yaml")

# From dict
parser = OpenAPIParser({
    "openapi": "3.0.0",
    "info": {"title": "My API", "version": "1.0"},
    "paths": {...}
})
```

### Get Operations

```python
operations = parser.get_operations()

for op in operations:
    print(f"ID: {op.operation_id}")
    print(f"Method: {op.method}")
    print(f"Path: {op.path}")
    print(f"Summary: {op.summary}")
    print(f"Parameters: {op.parameters}")
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `title` | str | API title |
| `version` | str | API version |
| `description` | str | API description |
| `base_url` | str | Base URL |
| `servers` | list | Server URLs |
| `operations` | list | All operations |

## Methods

### get_operations()

Get all operations:

```python
operations = parser.get_operations()
```

### get_operation()

Get specific operation:

```python
op = parser.get_operation("getUserById")
print(op.parameters)
print(op.request_body)
print(op.responses)
```

### get_schemas()

Get defined schemas:

```python
schemas = parser.get_schemas()
for name, schema in schemas.items():
    print(f"Schema: {name}")
    print(f"Properties: {schema['properties']}")
```

### get_security_schemes()

Get security definitions:

```python
security = parser.get_security_schemes()
# {'api_key': {'type': 'apiKey', ...}, ...}
```

### to_tools()

Convert to agent tools:

```python
tools = parser.to_tools()

# With filtering
tools = parser.to_tools(
    include_operations=["getUser", "createUser"],
    exclude_tags=["admin"]
)
```

## Operation Object

Each operation has:

```python
op.operation_id     # Unique identifier
op.method           # HTTP method (get, post, etc.)
op.path             # URL path
op.summary          # Short description
op.description      # Full description
op.parameters       # List of parameters
op.request_body     # Request body schema
op.responses        # Response schemas
op.tags             # Tags for categorization
op.security         # Security requirements
```

## Parameter Details

```python
for param in op.parameters:
    print(f"Name: {param['name']}")
    print(f"In: {param['in']}")  # path, query, header
    print(f"Required: {param['required']}")
    print(f"Type: {param['schema']['type']}")
```

## Request Body

```python
if op.request_body:
    content_type = list(op.request_body['content'].keys())[0]
    schema = op.request_body['content'][content_type]['schema']
    print(f"Body schema: {schema}")
```

## Validation

```python
from pyagent.openapi import OpenAPIParser, SpecValidationError

try:
    parser = OpenAPIParser("invalid_spec.yaml", validate=True)
except SpecValidationError as e:
    print(f"Invalid spec: {e.errors}")
```

## Supported Versions

| Version | Support |
|---------|---------|
| OpenAPI 3.0 | ✅ Full |
| OpenAPI 3.1 | ✅ Full |
| Swagger 2.0 | ✅ Converted |

## See Also

- [OpenAPI-Module](OpenAPI-Module) - Module overview
- [OpenAPIClient](OpenAPIClient) - API client
