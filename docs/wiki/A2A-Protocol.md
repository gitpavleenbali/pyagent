# A2A Protocol

Agent-to-Agent communication protocol for distributed agents.

> **See [[A2A-Module]] for full documentation.**

## Quick Start

### Server

```python
from pyai.a2a import A2AServer

server = A2AServer(
    name="Weather Agent",
    description="Provides weather information",
    agent=weather_agent
)

server.run(port=8080)
```

### Client

```python
from pyai.a2a import A2AClient

client = A2AClient("http://localhost:8080")

# Get agent info
card = await client.get_card()

# Send message
response = await client.send("What's the weather?")
```

## Features

- HTTP-based communication
- Agent discovery
- Task management
- Async messaging
- Agent cards

## Related Pages

- [[A2A-Module]] - Full module documentation
- [[A2AClient]] - Client class
- [[A2AServer]] - Server class
