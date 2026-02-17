# OpenAPIClient

The `OpenAPIClient` provides a programmatic interface to call REST APIs defined by OpenAPI specifications.

## Import

```python
from pyai.openapi import OpenAPIClient
```

## Constructor

```python
OpenAPIClient(
    spec_url: str,              # URL or path to OpenAPI spec
    base_url: str = None,       # Override base URL
    auth_token: str = None,     # Bearer token
    api_key: str = None,        # API key
    headers: dict = None,       # Custom headers
    timeout: float = 30.0       # Request timeout
)
```

## Basic Usage

### Initialize Client

```python
# From URL
client = OpenAPIClient("https://api.example.com/openapi.json")

# From file
client = OpenAPIClient("./specs/api.yaml")

# With authentication
client = OpenAPIClient(
    "https://api.example.com/openapi.json",
    auth_token="Bearer eyJhbGciOiJIUzI1NiIs..."
)
```

### Call Operations

```python
# Simple GET
users = client.call("getUsers")

# GET with parameters
user = client.call("getUserById", id=123)

# POST with body
new_user = client.call("createUser", body={
    "name": "John Doe",
    "email": "john@example.com"
})

# With query parameters
results = client.call("searchUsers", q="john", limit=10)
```

## Methods

### call()

Execute an API operation:

```python
def call(
    self,
    operation_id: str,         # Operation ID from spec
    **kwargs                   # Parameters
) -> dict:
```

**Parameter Types:**
- Path parameters: Passed directly
- Query parameters: Passed directly
- Body: Use `body=` keyword
- Headers: Use `headers=` keyword

### list_operations()

Get available operations:

```python
operations = client.list_operations()
for op in operations:
    print(f"{op['method']} {op['path']} - {op['summary']}")
```

### get_operation_schema()

Get details about an operation:

```python
schema = client.get_operation_schema("createUser")
print(f"Parameters: {schema['parameters']}")
print(f"Request body: {schema['requestBody']}")
```

## Authentication

### API Key

```python
client = OpenAPIClient(
    spec_url,
    api_key="your-api-key",
    api_key_header="X-API-Key"  # Header name
)
```

### Bearer Token

```python
client = OpenAPIClient(
    spec_url,
    auth_token="Bearer your-jwt-token"
)
```

### Custom Auth

```python
client = OpenAPIClient(
    spec_url,
    headers={
        "Authorization": "Custom auth-scheme token",
        "X-Custom-Auth": "value"
    }
)
```

## Error Handling

```python
from pyai.openapi import APIError, ValidationError

try:
    result = client.call("getUser", id=999)
except APIError as e:
    print(f"API Error: {e.status_code} - {e.message}")
except ValidationError as e:
    print(f"Invalid parameters: {e.errors}")
```

## Async Usage

```python
async def fetch_users():
    async with OpenAPIClient(spec_url) as client:
        users = await client.call_async("getUsers")
        return users
```

## Response Handling

```python
# Get raw response
response = client.call("getUsers", raw=True)
print(f"Status: {response.status_code}")
print(f"Headers: {response.headers}")
print(f"Body: {response.json()}")
```

## See Also

- [OpenAPI-Module](OpenAPI-Module) - Module overview
- [OpenAPIParser](OpenAPIParser) - Spec parsing
