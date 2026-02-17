# OpenAPI Tools

Auto-generate agent tools from OpenAPI/Swagger specifications.

## Overview

PYAI can automatically create tools from any OpenAPI spec:

```python
from pyai.openapi import OpenAPIClient

# Load from spec
client = OpenAPIClient.from_spec("https://api.example.com/openapi.json")

# Use with agent
agent = Agent(
    name="API Agent",
    tools=client.tools
)
```

## From URL

```python
from pyai.openapi import OpenAPIClient

# Swagger/OpenAPI URL
client = OpenAPIClient.from_spec(
    "https://petstore.swagger.io/v2/swagger.json"
)

# All endpoints become tools
print(client.tools)
# [get_pet_by_id, create_pet, update_pet, delete_pet, ...]
```

## From File

```python
# JSON spec
client = OpenAPIClient.from_file("api_spec.json")

# YAML spec
client = OpenAPIClient.from_file("api_spec.yaml")
```

## From Dictionary

```python
spec = {
    "openapi": "3.0.0",
    "info": {"title": "My API", "version": "1.0"},
    "paths": {
        "/users": {
            "get": {
                "operationId": "get_users",
                "summary": "Get all users"
            }
        }
    }
}

client = OpenAPIClient.from_dict(spec)
```

## With Authentication

```python
# API Key
client = OpenAPIClient.from_spec(
    "https://api.example.com/openapi.json",
    auth={"api_key": "your-key"}
)

# Bearer token
client = OpenAPIClient.from_spec(
    spec_url,
    auth={"bearer": "your-token"}
)

# OAuth
client = OpenAPIClient.from_spec(
    spec_url,
    auth={
        "oauth": {
            "client_id": "...",
            "client_secret": "...",
            "token_url": "..."
        }
    }
)
```

## Filtering Operations

```python
# Only specific operations
client = OpenAPIClient.from_spec(
    spec_url,
    include=["get_users", "create_user"]
)

# Exclude operations
client = OpenAPIClient.from_spec(
    spec_url,
    exclude=["delete_user", "admin_*"]
)

# Filter by tag
client = OpenAPIClient.from_spec(
    spec_url,
    tags=["users", "products"]
)
```

## Full Example

```python
from pyai import Agent, Runner
from pyai.openapi import OpenAPIClient

# Create client from pet store API
petstore = OpenAPIClient.from_spec(
    "https://petstore.swagger.io/v2/swagger.json"
)

# Create agent with API tools
agent = Agent(
    name="Pet Store Assistant",
    instructions="""
    You help manage the pet store.
    Use the available tools to:
    - Look up pets
    - Add new pets
    - Update pet status
    """,
    tools=petstore.tools
)

# Run
result = Runner.run_sync(agent, "Find all available pets")
print(result.output)
```

## Custom Tool Names

```python
client = OpenAPIClient.from_spec(
    spec_url,
    name_transform=lambda op: f"api_{op}"
)
# get_users -> api_get_users
```

## Direct API Calls

```python
# Call API directly
response = await client.call("get_users", limit=10)
response = await client.call("create_user", name="Alice", email="alice@example.com")
```

## Tool Schema

Generated tools include full schema:

```python
tool = client.tools[0]
print(tool.name)        # "get_pet_by_id"
print(tool.description) # "Returns a single pet"
print(tool.parameters)  # {"petId": {"type": "integer", "required": True}}
```

## See Also

- [[OpenAPI-Module]] - Full module docs
- [[OpenAPIClient]] - Client class
- [[OpenAPIParser]] - Parser class
- [[Creating-Tools]] - Custom tools
