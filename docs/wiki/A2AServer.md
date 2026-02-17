# A2AServer

The `A2AServer` class exposes agents over HTTP using the A2A protocol.

## Overview

A2AServer provides:
- HTTP server for agent access
- Agent Card endpoint (`/.well-known/agent-card`)
- Task submission and execution
- CORS support for web clients

## Installation

```python
from pyagent.a2a import A2AServer
```

## Basic Usage

```python
from pyagent import Agent
from pyagent.a2a import A2AServer

# Create agent
agent = Agent(
    name="my-agent",
    instructions="You are a helpful assistant."
)

# Create server
server = A2AServer(agent=agent, port=8080)

# Start serving
server.start()
```

## Constructor

```python
A2AServer(
    agent: Optional[Agent] = None,
    endpoint: Optional[A2AEndpoint] = None,
    host: str = "0.0.0.0",
    port: int = 8080,
    api_key: Optional[str] = None
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | `Agent` | `None` | PyAgent Agent instance |
| `endpoint` | `A2AEndpoint` | `None` | Custom endpoint configuration |
| `host` | `str` | `"0.0.0.0"` | Host to bind |
| `port` | `int` | `8080` | Port to listen on |
| `api_key` | `str` | `None` | Optional API key for authentication |

## Methods

### start()

Start the server (blocking).

```python
def start(self) -> None
```

**Example:**
```python
server = A2AServer(agent=agent, port=8080)
server.start()  # Blocks until shutdown
```

### start_background()

Start server in background thread.

```python
def start_background(self) -> None
```

**Example:**
```python
server = A2AServer(agent=agent)
server.start_background()

# Continue with other code
print("Server running in background")

# Later: stop
server.stop()
```

### stop()

Stop the server.

```python
def stop(self) -> None
```

### is_running()

Check if server is running.

```python
def is_running(self) -> bool
```

## A2AEndpoint

For custom request handling:

```python
from pyagent.a2a import A2AEndpoint, A2AServer, A2ATask, A2AResponse

def custom_handler(task: A2ATask) -> A2AResponse:
    """Custom task handler."""
    message = task.messages[-1].content
    
    # Process message
    result = f"Processed: {message}"
    
    return A2AResponse.success(
        task_id=task.id,
        content=result
    )

endpoint = A2AEndpoint(
    name="custom-agent",
    handler=custom_handler,
    description="A custom agent",
    skills=["processing", "analysis"]
)

server = A2AServer(endpoint=endpoint, port=8080)
server.start()
```

## HTTP Endpoints

### GET /

Returns the agent card (same as `/.well-known/agent-card`).

### GET /.well-known/agent-card

Returns agent metadata:

```json
{
  "name": "my-agent",
  "description": "A helpful assistant",
  "url": "http://localhost:8080",
  "skills": ["research", "analysis"],
  "version": "1.0.0"
}
```

### POST /tasks

Submit a new task:

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "context": {},
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "task_id": "task-abc123",
  "status": "completed",
  "content": "Hello! How can I help you?",
  "metadata": {}
}
```

### GET /tasks/{task_id}

Get task status for async tasks.

## Configuration Examples

### With Agent

```python
from pyagent import Agent
from pyagent.a2a import A2AServer

agent = Agent(
    name="research-agent",
    instructions="You are a research specialist.",
    model="gpt-4"
)

server = A2AServer(
    agent=agent,
    host="0.0.0.0",
    port=8080
)
```

### With Custom Skills

```python
from pyagent.a2a import A2AEndpoint, A2AServer

endpoint = A2AEndpoint(
    name="specialist-agent",
    agent=agent,
    description="Domain specialist",
    skills=["research", "analysis", "summarization", "translation"]
)

server = A2AServer(endpoint=endpoint)
```

### With API Key

```python
server = A2AServer(
    agent=agent,
    port=8080,
    api_key="your-secret-key"
)

# Clients must provide Authorization header:
# Authorization: Bearer your-secret-key
```

## Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY agent.py .

EXPOSE 8080
CMD ["python", "agent.py"]
```

```python
# agent.py
from pyagent import Agent
from pyagent.a2a import A2AServer

agent = Agent(name="dockerized-agent")
server = A2AServer(agent=agent, port=8080)
server.start()
```

## Health Checks

```python
# GET /health returns {"status": "ok"}
import requests

response = requests.get("http://localhost:8080/health")
if response.json()["status"] == "ok":
    print("Server healthy")
```

## Graceful Shutdown

```python
import signal

server = A2AServer(agent=agent)

def shutdown_handler(signum, frame):
    print("Shutting down...")
    server.stop()

signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)

server.start()
```

## See Also

- [A2A-Module](A2A-Module) - Module overview
- [A2AClient](A2AClient) - Client implementation
- [Agent](Agent) - Agent class reference
