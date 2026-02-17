# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
A2A Client Implementation

Connect to remote agents via A2A protocol.
"""

import json
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .protocol import A2AMessage, A2AResponse, A2ATask, AgentCard


class A2AClient:
    """Client for connecting to A2A agents.

    Example:
        client = A2AClient("http://remote-agent:8080")

        # Get agent card
        card = client.get_card()
        print(f"Connected to: {card.name}")

        # Send a task
        response = client.send("Hello, what can you do?")
        print(response.content)
    """

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize client.

        Args:
            url: Remote agent URL
            api_key: Optional API key
            timeout: Request timeout in seconds
        """
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._card: Optional[AgentCard] = None

    def _request(
        self,
        path: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request.

        Args:
            path: URL path
            method: HTTP method
            data: Request body data

        Returns:
            Response data
        """
        url = f"{self.url}{path}"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = None
        if data:
            body = json.dumps(data).encode("utf-8")

        request = Request(url, data=body, headers=headers, method=method)

        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            error_body = e.read().decode("utf-8")
            try:
                error_data = json.loads(error_body)
                raise A2AClientError(
                    error_data.get("error", str(e)),
                    status_code=e.code,
                )
            except json.JSONDecodeError:
                raise A2AClientError(str(e), status_code=e.code)
        except URLError as e:
            raise A2AClientError(f"Connection failed: {e.reason}")

    def get_card(self, refresh: bool = False) -> AgentCard:
        """Get remote agent's card.

        Args:
            refresh: Force refresh from server

        Returns:
            Agent card
        """
        if self._card and not refresh:
            return self._card

        data = self._request("/.well-known/agent-card")
        self._card = AgentCard.from_dict(data)
        return self._card

    def send(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> A2AResponse:
        """Send a message to the remote agent.

        Args:
            message: Message text
            context: Additional context
            session_id: Session ID for continuity

        Returns:
            Agent response
        """
        task = A2ATask.from_text(
            message,
            context=context or {},
            session_id=session_id,
        )
        return self.submit_task(task)

    def submit_task(self, task: A2ATask) -> A2AResponse:
        """Submit a task to the remote agent.

        Args:
            task: Task to submit

        Returns:
            Agent response
        """
        data = self._request("/task", method="POST", data=task.to_dict())
        return A2AResponse.from_dict(data)

    def health_check(self) -> bool:
        """Check if remote agent is healthy.

        Returns:
            True if healthy
        """
        try:
            data = self._request("/health")
            return data.get("status") == "healthy"
        except A2AClientError:
            return False

    @property
    def name(self) -> str:
        """Get remote agent name."""
        return self.get_card().name

    @property
    def skills(self) -> List[str]:
        """Get remote agent skills."""
        return self.get_card().skills


class A2AClientError(Exception):
    """Error from A2A client."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class RemoteAgent:
    """Wrapper for a remote A2A agent.

    Provides a local agent-like interface for remote agents.

    Example:
        remote = RemoteAgent("http://research-agent:8080")

        # Use like a local agent
        result = remote.run("Research topic X")
        print(result.output)
    """

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """Initialize remote agent.

        Args:
            url: Remote agent URL
            api_key: Optional API key
            timeout: Request timeout
        """
        self.client = A2AClient(url, api_key=api_key, timeout=timeout)
        self._session_id: Optional[str] = None

    def run(
        self,
        input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> "RemoteAgentResult":
        """Run the remote agent.

        Args:
            input: Input text
            context: Additional context

        Returns:
            Agent result
        """
        response = self.client.send(
            input,
            context=context,
            session_id=self._session_id,
        )

        return RemoteAgentResult(
            output=response.content,
            messages=response.messages,
            result=response.result,
            is_success=response.is_success,
            error=response.error,
        )

    async def arun(
        self,
        input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> "RemoteAgentResult":
        """Run the remote agent asynchronously.

        Args:
            input: Input text
            context: Additional context

        Returns:
            Agent result
        """
        # For now, use sync client
        # TODO: Add async HTTP client (aiohttp)
        return self.run(input, context)

    def start_session(self) -> str:
        """Start a new session.

        Returns:
            Session ID
        """
        import uuid

        self._session_id = str(uuid.uuid4())
        return self._session_id

    def end_session(self):
        """End the current session."""
        self._session_id = None

    @property
    def card(self) -> AgentCard:
        """Get the agent card."""
        return self.client.get_card()

    @property
    def name(self) -> str:
        """Get agent name."""
        return self.card.name

    @property
    def skills(self) -> List[str]:
        """Get agent skills."""
        return self.card.skills


class RemoteAgentResult:
    """Result from a remote agent execution."""

    def __init__(
        self,
        output: str,
        messages: List[A2AMessage],
        result: Any = None,
        is_success: bool = True,
        error: Optional[str] = None,
    ):
        self.output = output
        self.messages = messages
        self.result = result
        self.is_success = is_success
        self.error = error

    def __str__(self) -> str:
        return self.output
