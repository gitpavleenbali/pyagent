# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
In-Memory Session Store

Simple in-memory session storage for development and testing.
"""

from typing import Dict, List, Optional

from .base import BaseSessionStore, Session


class MemorySessionStore(BaseSessionStore):
    """In-memory session store.

    Good for:
    - Development and testing
    - Short-lived applications
    - Single-process deployments

    Note: Sessions are lost when the process exits.

    Example:
        store = MemorySessionStore()
        session = Session(id="my-session")
        store.save(session)
        loaded = store.load("my-session")
    """

    def __init__(self):
        """Initialize the memory store."""
        self._sessions: Dict[str, Session] = {}

    def save(self, session: Session) -> None:
        """Save a session to memory."""
        self._sessions[session.id] = session

    def load(self, session_id: str) -> Optional[Session]:
        """Load a session from memory."""
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        """Delete a session from memory."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(
        self, user_id: Optional[str] = None, agent_id: Optional[str] = None, limit: int = 100
    ) -> List[Session]:
        """List sessions with optional filtering."""
        sessions = list(self._sessions.values())

        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        if agent_id:
            sessions = [s for s in sessions if s.agent_id == agent_id]

        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.updated_at, reverse=True)

        return sessions[:limit]

    def clear(self) -> None:
        """Clear all sessions."""
        self._sessions.clear()

    @property
    def count(self) -> int:
        """Get number of stored sessions."""
        return len(self._sessions)
