# A2A Protocol Module

The Agent-to-Agent (A2A) module implements Google's A2A protocol for inter-agent communication and collaboration.

## Overview

The A2A protocol enables:
- **Agent Discovery**: Automatically find and connect to remote agents
- **Message Passing**: Send messages between agents
- **Task Delegation**: Delegate tasks to specialized agents
- **Capability Advertisement**: Agents advertise their skills

## Quick Start

```python
from pyagent.a2a import A2AServer, A2AClient, AgentCard

# Server: Expose your agent
server = A2AServer(agent, port=8080)
server.start()

# Client: Connect to remote agent
client = A2AClient("http://remote-agent:8080")
result = await client.send("Research topic X")
```

## Components

### Protocol Types

| Type | Description |
|------|-------------|
| `AgentCard` | Agent metadata and capabilities |
| `A2AMessage` | Message between agents |
| `A2ATask` | Task to be executed |
| `A2AResponse` | Task execution result |
| `TaskStatus` | Status of task execution |

### Server Components

| Component | Description |
|-----------|-------------|
| [A2AServer](A2AServer) | HTTP server exposing agent |
| `A2AEndpoint` | Endpoint configuration |

### Client Components

| Component | Description |
|-----------|-------------|
| [A2AClient](A2AClient) | Connect to remote agents |
| `RemoteAgent` | Wrapper for remote agent |

### Registry

| Function | Description |
|----------|-------------|
| `AgentRegistry` | Central agent registry |
| `register_agent()` | Register agent in registry |
| `discover_agents()` | Find available agents |

## Architecture

```
┌─────────────────┐          ┌─────────────────┐
│   Agent A       │          │   Agent B       │
│  (A2AServer)    │◄────────►│  (A2AClient)    │
│                 │   HTTP    │                 │
│  /.well-known/  │          │                 │
│   agent-card    │          │                 │
└─────────────────┘          └─────────────────┘
         │
         ▼
┌─────────────────┐
│  AgentRegistry  │
│   (Discovery)   │
└─────────────────┘
```

## Usage Patterns

### 1. Expose Agent as Server

```python
from pyagent import Agent
from pyagent.a2a import A2AServer

# Create your agent
agent = Agent(
    name="research-agent",
    instructions="You are a research specialist."
)

# Expose via A2A
server = A2AServer(
    agent=agent,
    port=8080,
    host="0.0.0.0"
)

# Start serving
server.start()  # Blocking
# or
server.start_background()  # Non-blocking
```

### 2. Connect to Remote Agent

```python
from pyagent.a2a import A2AClient

# Connect to remote agent
client = A2AClient(
    url="http://research-agent:8080",
    api_key="optional-api-key"
)

# Get agent capabilities
card = client.get_card()
print(f"Agent: {card.name}")
print(f"Skills: {card.skills}")

# Send task
response = client.send(
    message="Research quantum computing advances",
    context={"depth": "detailed"}
)
print(response.content)
```

### 3. Multi-Agent Collaboration

```python
from pyagent.a2a import A2AClient, discover_agents

# Discover available agents
agents = discover_agents("http://registry:8000")

# Find specialist
researcher = next(
    a for a in agents 
    if "research" in a.skills
)

# Delegate task
client = A2AClient(researcher.url)
result = await client.send_task(
    "Analyze market trends",
    timeout=60.0
)
```

## Agent Card

The Agent Card follows the A2A specification:

```python
from pyagent.a2a import AgentCard

card = AgentCard(
    name="my-agent",
    description="A helpful assistant",
    url="http://localhost:8080",
    skills=["research", "analysis", "summarization"],
    version="1.0.0",
    provider="MyOrg"
)

# Serialize
data = card.to_dict()

# Endpoint: /.well-known/agent-card
```

## Security

### API Key Authentication

```python
# Server with API key
server = A2AServer(
    agent=agent,
    api_key="secret-key"
)

# Client with API key
client = A2AClient(
    url="http://agent:8080",
    api_key="secret-key"
)
```

### CORS Configuration

```python
server = A2AServer(
    agent=agent,
    cors_origins=["https://myapp.com"]
)
```

## See Also

- [A2AClient](A2AClient) - Client implementation details
- [A2AServer](A2AServer) - Server implementation details
- [Workflows](Workflows) - Multi-agent workflows
