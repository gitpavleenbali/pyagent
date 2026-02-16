# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
A2A Server Implementation

Expose agents over HTTP for remote access.
"""

import json
import threading
from dataclasses import dataclass
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse, parse_qs

from .protocol import AgentCard, A2ATask, A2AResponse, A2AMessage, TaskStatus


class A2AEndpoint:
    """A2A endpoint configuration.
    
    Wraps an agent or callable with A2A protocol handling.
    
    Example:
        def my_handler(task: A2ATask) -> A2AResponse:
            # Process task
            return A2AResponse.success(task.id, "Done!")
        
        endpoint = A2AEndpoint(
            name="my-agent",
            handler=my_handler
        )
    """
    
    def __init__(
        self,
        name: str,
        handler: Optional[Callable[[A2ATask], A2AResponse]] = None,
        agent: Optional[Any] = None,
        description: str = "",
        skills: Optional[list] = None,
    ):
        """Initialize endpoint.
        
        Args:
            name: Endpoint name
            handler: Custom handler function
            agent: PyAgent Agent instance
            description: Endpoint description
            skills: List of skills
        """
        self.name = name
        self.description = description
        self.skills = skills or []
        self._handler = handler
        self._agent = agent
    
    def handle(self, task: A2ATask) -> A2AResponse:
        """Handle an incoming task."""
        try:
            if self._handler:
                return self._handler(task)
            
            if self._agent:
                return self._handle_with_agent(task)
            
            return A2AResponse.failure(
                task.id,
                "No handler configured"
            )
        except Exception as e:
            return A2AResponse.failure(task.id, str(e))
    
    def _handle_with_agent(self, task: A2ATask) -> A2AResponse:
        """Handle task with PyAgent agent."""
        # Extract text from messages
        user_messages = [
            m.content for m in task.messages
            if m.role == "user"
        ]
        input_text = "\n".join(user_messages) if user_messages else ""
        
        # Run agent
        result = self._agent.run(input_text)
        
        # Convert to response
        if hasattr(result, "output"):
            content = result.output
        elif isinstance(result, str):
            content = result
        else:
            content = str(result)
        
        return A2AResponse.success(task.id, content, result=result)
    
    def get_card(self, url: str = "") -> AgentCard:
        """Get agent card for this endpoint."""
        return AgentCard(
            name=self.name,
            description=self.description,
            url=url,
            skills=self.skills,
        )


class A2ARequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for A2A protocol."""
    
    server: "A2AServer"
    
    def log_message(self, format: str, *args):
        """Suppress default logging."""
        pass
    
    def _send_json(self, data: Dict[str, Any], status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))
    
    def _send_error(self, message: str, status: int = 400):
        """Send error response."""
        self._send_json({"error": message}, status)
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        
        if path == "" or path == "/":
            # Root: return agent card
            card = self.server.endpoint.get_card(
                f"http://{self.server.server_address[0]}:{self.server.server_address[1]}"
            )
            self._send_json(card.to_dict())
        
        elif path == "/.well-known/agent-card":
            # Standard agent card endpoint
            card = self.server.endpoint.get_card(
                f"http://{self.server.server_address[0]}:{self.server.server_address[1]}"
            )
            self._send_json(card.to_dict())
        
        elif path == "/health":
            # Health check
            self._send_json({"status": "healthy"})
        
        else:
            self._send_error("Not found", 404)
    
    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        
        # Read body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_error("Invalid JSON")
            return
        
        if path == "/task" or path == "/tasks":
            # Create and execute task
            task = A2ATask.from_dict(data)
            response = self.server.endpoint.handle(task)
            self._send_json(response.to_dict())
        
        elif path == "/message" or path == "/send":
            # Simple message endpoint
            content = data.get("content") or data.get("message") or ""
            task = A2ATask.from_text(content)
            response = self.server.endpoint.handle(task)
            self._send_json(response.to_dict())
        
        else:
            self._send_error("Not found", 404)


class A2AServer(HTTPServer):
    """HTTP server for A2A protocol.
    
    Exposes an agent or handler over HTTP.
    
    Example:
        # With agent
        agent = Agent(model="gpt-4")
        server = A2AServer(agent=agent, port=8080)
        server.start()
        
        # With custom handler
        def handler(task):
            return A2AResponse.success(task.id, "Hello!")
        
        server = A2AServer(handler=handler, port=8080)
        server.start()
    """
    
    endpoint: A2AEndpoint
    
    def __init__(
        self,
        agent: Optional[Any] = None,
        handler: Optional[Callable[[A2ATask], A2AResponse]] = None,
        name: str = "pyagent",
        host: str = "0.0.0.0",
        port: int = 8080,
        description: str = "",
        skills: Optional[list] = None,
    ):
        """Initialize A2A server.
        
        Args:
            agent: PyAgent Agent instance
            handler: Custom handler function
            name: Agent name
            host: Host to bind to
            port: Port to listen on
            description: Agent description
            skills: Agent skills
        """
        self.endpoint = A2AEndpoint(
            name=name,
            handler=handler,
            agent=agent,
            description=description,
            skills=skills or [],
        )
        
        super().__init__((host, port), A2ARequestHandler)
        
        self._thread: Optional[threading.Thread] = None
        self._running = False
    
    def start(self, background: bool = True):
        """Start the server.
        
        Args:
            background: Run in background thread
        """
        self._running = True
        
        if background:
            self._thread = threading.Thread(target=self._serve)
            self._thread.daemon = True
            self._thread.start()
        else:
            self._serve()
    
    def stop(self):
        """Stop the server."""
        self._running = False
        self.shutdown()
    
    def _serve(self):
        """Serve requests."""
        while self._running:
            self.handle_request()
    
    @property
    def url(self) -> str:
        """Get server URL."""
        host, port = self.server_address
        return f"http://{host}:{port}"
