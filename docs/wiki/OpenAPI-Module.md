# OpenAPI Module

The OpenAPI module enables automatic tool generation from OpenAPI/Swagger specifications, allowing agents to interact with any REST API.

## Overview

```python
from pyagent.openapi import OpenAPIClient, OpenAPIParser
from pyagent.openapi.tools import create_tools_from_spec
```

## Key Components

| Component | Description |
|-----------|-------------|
| [OpenAPIClient](OpenAPIClient) | HTTP client for API calls |
| [OpenAPIParser](OpenAPIParser) | Parse OpenAPI specs |
| create_tools_from_spec | Generate agent tools |

## Quick Start

### From OpenAPI Spec

```python
from pyagent.openapi import create_tools_from_spec

# Load from URL
tools = create_tools_from_spec("https://api.example.com/openapi.json")

# Load from file
tools = create_tools_from_spec("./api_spec.yaml")

# Use with agent
agent = Agent(
    name="APIAgent",
    tools=tools
)
```

### Direct API Client

```python
from pyagent.openapi import OpenAPIClient

client = OpenAPIClient("https://api.example.com/openapi.json")

# Call an endpoint
response = client.call("getUsers", limit=10)

# With authentication
client = OpenAPIClient(
    "https://api.example.com/openapi.json",
    auth_token="Bearer your-token"
)
```

## Tool Generation

### Basic Generation

```python
tools = create_tools_from_spec(
    "https://petstore.swagger.io/v2/swagger.json"
)

# Generated tools include:
# - getPetById
# - addPet
# - updatePet
# - deletePet
# etc.
```

### Filtered Generation

```python
# Only include specific operations
tools = create_tools_from_spec(
    "api_spec.yaml",
    include_operations=["getUser", "createUser"]
)

# Exclude certain operations
tools = create_tools_from_spec(
    "api_spec.yaml",
    exclude_operations=["deleteUser"]
)

# Filter by tags
tools = create_tools_from_spec(
    "api_spec.yaml",
    include_tags=["users", "products"]
)
```

## Authentication

### API Key

```python
tools = create_tools_from_spec(
    "api_spec.yaml",
    auth={"api_key": "your-key"}
)
```

### Bearer Token

```python
tools = create_tools_from_spec(
    "api_spec.yaml",
    auth={"bearer_token": "your-jwt-token"}
)
```

### OAuth2

```python
tools = create_tools_from_spec(
    "api_spec.yaml",
    auth={
        "oauth2": {
            "client_id": "...",
            "client_secret": "...",
            "token_url": "https://..."
        }
    }
)
```

## Using with Agents

```python
from pyagent import Agent
from pyagent.openapi import create_tools_from_spec

# Create tools from Petstore API
tools = create_tools_from_spec(
    "https://petstore.swagger.io/v2/swagger.json"
)

# Create agent with API tools
agent = Agent(
    name="PetStoreAgent",
    instructions="You help users manage pets in the store.",
    tools=tools
)

# Agent can now call API endpoints
result = agent.run("Find all available dogs")
```

## Customization

### Custom Base URL

```python
tools = create_tools_from_spec(
    "api_spec.yaml",
    base_url="https://staging.api.example.com"
)
```

### Custom Headers

```python
tools = create_tools_from_spec(
    "api_spec.yaml",
    headers={
        "X-Custom-Header": "value",
        "Accept-Language": "en-US"
    }
)
```

## See Also

- [OpenAPIClient](OpenAPIClient) - Direct API client
- [OpenAPIParser](OpenAPIParser) - Spec parsing
