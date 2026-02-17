# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
pyai A2A (Agent-to-Agent) Protocol

Implementation of Google's Agent-to-Agent protocol for
inter-agent communication and collaboration.

Features:
- Agent discovery and registration
- Message passing between agents
- Task delegation and results
- Agent capabilities advertisement

Example:
    from pyai.a2a import A2AServer, A2AClient, AgentCard

    # Server: Expose agent
    server = A2AServer(agent, port=8080)
    server.start()

    # Client: Connect to remote agent
    client = A2AClient("http://remote-agent:8080")
    result = await client.send("Research topic X")
"""

from .client import (
    A2AClient,
    RemoteAgent,
)
from .protocol import (
    A2AMessage,
    A2AResponse,
    A2ATask,
    AgentCard,
    TaskStatus,
)
from .registry import (
    AgentRegistry,
    discover_agents,
    register_agent,
)
from .server import (
    A2AEndpoint,
    A2AServer,
)

__all__ = [
    # Protocol types
    "AgentCard",
    "A2AMessage",
    "A2ATask",
    "A2AResponse",
    "TaskStatus",
    # Server
    "A2AServer",
    "A2AEndpoint",
    # Client
    "A2AClient",
    "RemoteAgent",
    # Registry
    "AgentRegistry",
    "register_agent",
    "discover_agents",
]
