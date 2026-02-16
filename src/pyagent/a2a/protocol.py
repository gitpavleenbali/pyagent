# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
A2A Protocol Definitions

Based on Google's Agent-to-Agent protocol specification.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class TaskStatus(Enum):
    """Status of an A2A task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentCard:
    """Agent capability advertisement.
    
    Describes what an agent can do and how to interact with it.
    
    Attributes:
        name: Human-readable agent name
        description: Agent description
        url: Agent endpoint URL
        version: Agent version
        skills: List of skill names
        protocols: Supported protocols
        authentication: Auth requirements
        metadata: Additional metadata
    """
    name: str
    description: str = ""
    url: str = ""
    version: str = "1.0.0"
    skills: List[str] = field(default_factory=list)
    protocols: List[str] = field(default_factory=lambda: ["a2a/1.0"])
    authentication: Optional[Dict[str, Any]] = None
    input_modes: List[str] = field(default_factory=lambda: ["text"])
    output_modes: List[str] = field(default_factory=lambda: ["text"])
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "skills": self.skills,
            "protocols": self.protocols,
            "authentication": self.authentication,
            "input_modes": self.input_modes,
            "output_modes": self.output_modes,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCard":
        """Create from dictionary."""
        return cls(
            name=data.get("name", "Unknown"),
            description=data.get("description", ""),
            url=data.get("url", ""),
            version=data.get("version", "1.0.0"),
            skills=data.get("skills", []),
            protocols=data.get("protocols", ["a2a/1.0"]),
            authentication=data.get("authentication"),
            input_modes=data.get("input_modes", ["text"]),
            output_modes=data.get("output_modes", ["text"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class A2AMessage:
    """A message in the A2A protocol.
    
    Attributes:
        id: Unique message ID
        role: Message role (user, agent)
        content: Message content
        content_type: MIME type of content
        timestamp: When message was created
        metadata: Additional metadata
    """
    content: str
    role: str = "user"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content_type: str = "text/plain"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "content_type": self.content_type,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AMessage":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            role=data.get("role", "user"),
            content=data.get("content", ""),
            content_type=data.get("content_type", "text/plain"),
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
        )


@dataclass
class A2ATask:
    """An A2A task request.
    
    Represents a task delegated to a remote agent.
    
    Attributes:
        id: Unique task ID
        messages: Input messages
        session_id: Optional session for context
        context: Task context
        metadata: Additional metadata
    """
    messages: List[A2AMessage]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "messages": [m.to_dict() for m in self.messages],
            "session_id": self.session_id,
            "context": self.context,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2ATask":
        """Create from dictionary."""
        messages = [
            A2AMessage.from_dict(m)
            for m in data.get("messages", [])
        ]
        
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.utcnow()
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            messages=messages,
            session_id=data.get("session_id"),
            context=data.get("context", {}),
            metadata=data.get("metadata", {}),
            created_at=created_at,
        )
    
    @classmethod
    def from_text(cls, text: str, **kwargs) -> "A2ATask":
        """Create a simple task from text."""
        return cls(
            messages=[A2AMessage(content=text)],
            **kwargs,
        )


@dataclass
class A2AResponse:
    """Response from an A2A task.
    
    Attributes:
        task_id: ID of the task this responds to
        status: Task status
        messages: Response messages
        result: Task result data
        error: Error if failed
        metadata: Additional metadata
    """
    task_id: str
    status: TaskStatus = TaskStatus.COMPLETED
    messages: List[A2AMessage] = field(default_factory=list)
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "messages": [m.to_dict() for m in self.messages],
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
            "completed_at": self.completed_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AResponse":
        """Create from dictionary."""
        messages = [
            A2AMessage.from_dict(m)
            for m in data.get("messages", [])
        ]
        
        status = data.get("status", "completed")
        if isinstance(status, str):
            status = TaskStatus(status)
        
        completed_at = data.get("completed_at")
        if isinstance(completed_at, str):
            completed_at = datetime.fromisoformat(completed_at)
        elif completed_at is None:
            completed_at = datetime.utcnow()
        
        return cls(
            task_id=data.get("task_id", ""),
            status=status,
            messages=messages,
            result=data.get("result"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
            completed_at=completed_at,
        )
    
    @classmethod
    def success(
        cls,
        task_id: str,
        content: str,
        result: Optional[Any] = None,
        **kwargs
    ) -> "A2AResponse":
        """Create a successful response."""
        return cls(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            messages=[A2AMessage(content=content, role="agent")],
            result=result,
            **kwargs,
        )
    
    @classmethod
    def failure(cls, task_id: str, error: str, **kwargs) -> "A2AResponse":
        """Create a failure response."""
        return cls(
            task_id=task_id,
            status=TaskStatus.FAILED,
            error=error,
            **kwargs,
        )
    
    @property
    def is_success(self) -> bool:
        """Check if task succeeded."""
        return self.status == TaskStatus.COMPLETED
    
    @property
    def content(self) -> str:
        """Get response content."""
        if self.messages:
            return self.messages[-1].content
        return ""
