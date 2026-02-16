# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
PyAgent Sessions Module - Persistent Conversation State

Inspired by Google ADK's sessions/ module, this provides:

- In-memory sessions (default)
- SQLite persistence (local deployment)
- Redis persistence (distributed deployment)
- Database persistence (enterprise)
- Custom session stores

Example:
    from pyagent.sessions import Session, SQLiteSessionStore, get_session
    
    # Get or create a session
    session = get_session("user-123")
    
    # Add messages
    session.add_message(role="user", content="Hello!")
    
    # Persist with SQLite
    store = SQLiteSessionStore("sessions.db")
    store.save(session)
"""

from .base import Session, SessionMessage, SessionState, SessionCheckpoint
from .memory import MemorySessionStore
from .sqlite import SQLiteSessionStore
from .redis import RedisSessionStore
from .manager import SessionManager, get_session, create_session

__all__ = [
    # Core
    "Session",
    "SessionMessage", 
    "SessionState",
    "SessionCheckpoint",
    # Stores
    "MemorySessionStore",
    "SQLiteSessionStore",
    "RedisSessionStore",
    # Manager
    "SessionManager",
    "get_session",
    "create_session",
]
