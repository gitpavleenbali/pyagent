# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Base Session Classes

Core abstractions for session management.
"""

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


class SessionState(Enum):
    """Session lifecycle states."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    EXPIRED = "expired"


@dataclass
class SessionMessage:
    """A message within a session.

    Attributes:
        id: Unique message identifier
        role: Message role (system, user, assistant, tool)
        content: Message content
        timestamp: When the message was created
        metadata: Additional message metadata
        tool_calls: Tool calls if any
        tool_call_id: Tool call ID this message responds to
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionMessage":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            role=data["role"],
            content=data["content"],
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
        )

    def to_api_format(self) -> Dict[str, Any]:
        """Convert to LLM API message format."""
        result = {"role": self.role, "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


@dataclass
class SessionCheckpoint:
    """A session checkpoint for rewind capability.

    Attributes:
        id: Checkpoint identifier
        name: Human-readable name
        message_index: Index in messages list
        context_snapshot: Copy of context at checkpoint
        created_at: When checkpoint was created
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    message_index: int = 0
    context_snapshot: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "message_index": self.message_index,
            "context_snapshot": self.context_snapshot,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionCheckpoint":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.utcnow()

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            message_index=data.get("message_index", 0),
            context_snapshot=data.get("context_snapshot", {}),
            created_at=created_at,
        )


@dataclass
class Session:
    """A conversation session.

    Attributes:
        id: Unique session identifier
        user_id: Associated user ID
        agent_id: Associated agent ID
        messages: List of messages in the session
        state: Current session state
        metadata: Session metadata
        created_at: When the session was created
        updated_at: When the session was last updated
        context: Arbitrary context data
        checkpoints: Saved checkpoints for rewind
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    messages: List[SessionMessage] = field(default_factory=list)
    state: SessionState = SessionState.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)
    checkpoints: List[SessionCheckpoint] = field(default_factory=list)

    def add_message(self, role: str, content: str, **kwargs) -> SessionMessage:
        """Add a message to the session.

        Args:
            role: Message role
            content: Message content
            **kwargs: Additional message fields

        Returns:
            The created message
        """
        message = SessionMessage(role=role, content=content, **kwargs)
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        return message

    def add_user_message(self, content: str, **kwargs) -> SessionMessage:
        """Add a user message."""
        return self.add_message("user", content, **kwargs)

    def add_assistant_message(self, content: str, **kwargs) -> SessionMessage:
        """Add an assistant message."""
        return self.add_message("assistant", content, **kwargs)

    def add_system_message(self, content: str, **kwargs) -> SessionMessage:
        """Add a system message."""
        return self.add_message("system", content, **kwargs)

    def get_messages(
        self, include_system: bool = True, last_n: Optional[int] = None
    ) -> List[SessionMessage]:
        """Get messages from the session.

        Args:
            include_system: Whether to include system messages
            last_n: Only return the last N messages

        Returns:
            List of messages
        """
        messages = self.messages
        if not include_system:
            messages = [m for m in messages if m.role != "system"]
        if last_n:
            messages = messages[-last_n:]
        return messages

    def get_api_messages(
        self, include_system: bool = True, last_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get messages in LLM API format."""
        return [m.to_api_format() for m in self.get_messages(include_system, last_n)]

    def clear(self) -> None:
        """Clear all messages except system messages."""
        self.messages = [m for m in self.messages if m.role == "system"]
        self.updated_at = datetime.utcnow()

    def reset(self) -> None:
        """Reset the session completely."""
        self.messages = []
        self.context = {}
        self.state = SessionState.ACTIVE
        self.updated_at = datetime.utcnow()

    def rewind(self, message_id: str) -> bool:
        """Rewind session to before a specific message.

        Like Google ADK's session rewind feature.

        Args:
            message_id: ID of message to rewind to (exclusive)

        Returns:
            True if rewound successfully
        """
        for i, msg in enumerate(self.messages):
            if msg.id == message_id:
                self.messages = self.messages[:i]
                self.updated_at = datetime.utcnow()
                return True
        return False

    def checkpoint(self, name: str = "") -> SessionCheckpoint:
        """Create a checkpoint for later rewind.

        Like Google ADK's session checkpoint feature.

        Args:
            name: Optional human-readable name

        Returns:
            The created checkpoint

        Example:
            session.add_user_message("Start")
            cp = session.checkpoint("before_research")
            session.add_assistant_message("Researching...")
            # Later, rewind to checkpoint
            session.rewind_to_checkpoint(cp.id)
        """
        import copy

        checkpoint = SessionCheckpoint(
            name=name,
            message_index=len(self.messages),
            context_snapshot=copy.deepcopy(self.context),
        )
        self.checkpoints.append(checkpoint)
        self.updated_at = datetime.utcnow()
        return checkpoint

    def rewind_to_checkpoint(self, checkpoint_id: str, restore_context: bool = True) -> bool:
        """Rewind session to a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to rewind to
            restore_context: Also restore context data

        Returns:
            True if rewound successfully
        """
        for checkpoint in self.checkpoints:
            if checkpoint.id == checkpoint_id:
                self.messages = self.messages[: checkpoint.message_index]
                if restore_context:
                    import copy

                    self.context = copy.deepcopy(checkpoint.context_snapshot)
                self.updated_at = datetime.utcnow()
                return True
        return False

    def rewind_to_checkpoint_by_name(self, name: str, restore_context: bool = True) -> bool:
        """Rewind to a checkpoint by name.

        Args:
            name: Checkpoint name
            restore_context: Also restore context data

        Returns:
            True if rewound successfully
        """
        for checkpoint in reversed(self.checkpoints):
            if checkpoint.name == name:
                return self.rewind_to_checkpoint(checkpoint.id, restore_context)
        return False

    def get_checkpoints(self) -> List[SessionCheckpoint]:
        """Get all checkpoints."""
        return list(self.checkpoints)

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        for i, cp in enumerate(self.checkpoints):
            if cp.id == checkpoint_id:
                self.checkpoints.pop(i)
                return True
        return False

    def rewind_n_messages(self, n: int) -> int:
        """Rewind the last N messages.

        Args:
            n: Number of messages to remove

        Returns:
            Number of messages actually removed
        """
        original_count = len(self.messages)
        self.messages = self.messages[:-n] if n > 0 else self.messages
        removed = original_count - len(self.messages)
        if removed > 0:
            self.updated_at = datetime.utcnow()
        return removed

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "messages": [m.to_dict() for m in self.messages],
            "state": self.state.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "context": self.context,
            "checkpoints": [cp.to_dict() for cp in self.checkpoints],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Create session from dictionary."""
        messages = [SessionMessage.from_dict(m) for m in data.get("messages", [])]
        checkpoints = [SessionCheckpoint.from_dict(cp) for cp in data.get("checkpoints", [])]

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.utcnow()

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = datetime.utcnow()

        state = data.get("state", "active")
        if isinstance(state, str):
            state = SessionState(state)

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            user_id=data.get("user_id"),
            agent_id=data.get("agent_id"),
            messages=messages,
            state=state,
            metadata=data.get("metadata", {}),
            created_at=created_at,
            updated_at=updated_at,
            context=data.get("context", {}),
            checkpoints=checkpoints,
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "Session":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @property
    def message_count(self) -> int:
        """Get total message count."""
        return len(self.messages)

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.state == SessionState.ACTIVE

    def __len__(self) -> int:
        return len(self.messages)


class BaseSessionStore(ABC):
    """Abstract base class for session stores.

    Subclasses must implement save, load, delete, and list_sessions.
    """

    @abstractmethod
    def save(self, session: Session) -> None:
        """Save a session."""
        pass

    @abstractmethod
    def load(self, session_id: str) -> Optional[Session]:
        """Load a session by ID."""
        pass

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        pass

    @abstractmethod
    def list_sessions(
        self, user_id: Optional[str] = None, agent_id: Optional[str] = None, limit: int = 100
    ) -> List[Session]:
        """List sessions with optional filtering."""
        pass

    def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return self.load(session_id) is not None

    def get_or_create(
        self, session_id: str, user_id: Optional[str] = None, agent_id: Optional[str] = None
    ) -> Session:
        """Get existing session or create new one."""
        session = self.load(session_id)
        if session is None:
            session = Session(id=session_id, user_id=user_id, agent_id=agent_id)
            self.save(session)
        return session
