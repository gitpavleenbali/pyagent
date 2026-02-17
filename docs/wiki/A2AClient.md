# A2AClient

The `A2AClient` class connects to remote agents via the A2A protocol.

## Overview

A2AClient enables:
- Connecting to remote agents
- Retrieving agent capabilities (Agent Card)
- Sending messages and tasks
- Handling async task execution

## Installation

```python
from pyai.a2a import A2AClient
```

## Basic Usage

```python
from pyai.a2a import A2AClient

# Create client
client = A2AClient("http://remote-agent:8080")

# Get agent info
card = client.get_card()
print(f"Connected to: {card.name}")

# Send message
response = client.send("Hello, what can you do?")
print(response.content)
```

## Constructor

```python
A2AClient(
    url: str,
    api_key: Optional[str] = None,
    timeout: float = 30.0
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | required | Remote agent URL |
| `api_key` | `str` | `None` | Optional API key for authentication |
| `timeout` | `float` | `30.0` | Request timeout in seconds |

## Methods

### get_card()

Retrieve the remote agent's capabilities card.

```python
def get_card(self, refresh: bool = False) -> AgentCard
```

**Parameters:**
- `refresh`: Force refresh from server (default: `False`)

**Returns:** `AgentCard` object

**Example:**
```python
card = client.get_card()
print(f"Name: {card.name}")
print(f"Description: {card.description}")
print(f"Skills: {card.skills}")
```

### send()

Send a message to the remote agent.

```python
def send(
    self,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None
) -> A2AResponse
```

**Parameters:**
- `message`: Message text to send
- `context`: Optional context dictionary
- `session_id`: Optional session ID for continuity

**Returns:** `A2AResponse` object

**Example:**
```python
response = client.send(
    message="Analyze this data",
    context={"data": [1, 2, 3]},
    session_id="session-123"
)

if response.success:
    print(response.content)
else:
    print(f"Error: {response.error}")
```

### send_task()

Send a task for asynchronous execution.

```python
async def send_task(
    self,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None
) -> A2AResponse
```

**Parameters:**
- `message`: Task description
- `context`: Optional context
- `timeout`: Override default timeout

**Returns:** `A2AResponse` object

**Example:**
```python
import asyncio

async def main():
    response = await client.send_task(
        "Research quantum computing",
        timeout=120.0  # 2 minutes for complex task
    )
    print(response.content)

asyncio.run(main())
```

### get_task_status()

Check status of an async task.

```python
def get_task_status(self, task_id: str) -> TaskStatus
```

**Example:**
```python
status = client.get_task_status("task-abc123")
print(f"Status: {status.state}")  # pending, running, completed, failed
print(f"Progress: {status.progress}%")
```

## RemoteAgent Wrapper

For agent-like interface to remote agents:

```python
from pyai.a2a import RemoteAgent

# Wrap remote agent
agent = RemoteAgent("http://remote-agent:8080")

# Use like local agent
result = agent.run("What can you help me with?")
print(result)
```

## Error Handling

```python
from pyai.a2a import A2AClient, A2AClientError

client = A2AClient("http://agent:8080")

try:
    response = client.send("Hello")
except A2AClientError as e:
    print(f"Error: {e.message}")
    print(f"Status code: {e.status_code}")
```

## Connection Pooling

For high-throughput scenarios:

```python
from pyai.a2a import A2AClient

client = A2AClient(
    url="http://agent:8080",
    timeout=30.0
)

# Reuse client for multiple requests
for query in queries:
    response = client.send(query)
    process(response)
```

## Authentication

### Bearer Token

```python
client = A2AClient(
    url="http://agent:8080",
    api_key="your-api-key"  # Sent as Bearer token
)
```

### Custom Headers

```python
# Not directly supported - extend class if needed
class CustomA2AClient(A2AClient):
    def _request(self, path, method="GET", data=None):
        # Add custom headers
        self.custom_headers = {"X-Custom": "value"}
        return super()._request(path, method, data)
```

## See Also

- [A2A-Module](A2A-Module) - Module overview
- [A2AServer](A2AServer) - Server implementation
- [Workflows](Workflows) - Multi-agent collaboration
