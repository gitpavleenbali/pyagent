# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Session Manager

High-level session management with auto-detection of storage backend.
"""

import os
from typing import Optional, Type

from .base import BaseSessionStore, Session
from .memory import MemorySessionStore
from .sqlite import SQLiteSessionStore
from .redis import RedisSessionStore


class SessionManager:
    """Central session management.
    
    Provides a unified interface for session operations with
    automatic backend detection.
    
    Example:
        # Auto-detect backend
        manager = SessionManager()
        
        # Get or create session
        session = manager.get_or_create("user-123")
        session.add_user_message("Hello!")
        manager.save(session)
        
        # Use specific backend
        manager = SessionManager(backend="sqlite", db_path="sessions.db")
    """
    
    _default_instance: "SessionManager" = None
    
    def __init__(
        self,
        backend: Optional[str] = None,
        store: Optional[BaseSessionStore] = None,
        **kwargs
    ):
        """Initialize session manager.
        
        Args:
            backend: Backend type ("memory", "sqlite", "redis")
            store: Pre-configured session store
            **kwargs: Backend-specific arguments
        """
        if store:
            self._store = store
        else:
            backend = backend or self._detect_backend()
            self._store = self._create_store(backend, **kwargs)
    
    def _detect_backend(self) -> str:
        """Auto-detect the best available backend."""
        # Check environment variable
        explicit = os.environ.get("PYAGENT_SESSION_BACKEND")
        if explicit:
            return explicit.lower()
        
        # Check for Redis
        redis_host = os.environ.get("REDIS_HOST") or os.environ.get("PYAGENT_REDIS_HOST")
        if redis_host:
            return "redis"
        
        # Check for SQLite path
        sqlite_path = os.environ.get("PYAGENT_SESSION_DB")
        if sqlite_path:
            return "sqlite"
        
        # Default to memory
        return "memory"
    
    def _create_store(self, backend: str, **kwargs) -> BaseSessionStore:
        """Create a session store for the given backend."""
        if backend == "memory":
            return MemorySessionStore()
        
        elif backend == "sqlite":
            db_path = kwargs.get("db_path") or os.environ.get(
                "PYAGENT_SESSION_DB", "pyagent_sessions.db"
            )
            return SQLiteSessionStore(db_path=db_path)
        
        elif backend == "redis":
            return RedisSessionStore(
                host=kwargs.get("host") or os.environ.get("REDIS_HOST", "localhost"),
                port=int(kwargs.get("port") or os.environ.get("REDIS_PORT", 6379)),
                password=kwargs.get("password") or os.environ.get("REDIS_PASSWORD"),
                **{k: v for k, v in kwargs.items() if k not in ["host", "port", "password"]}
            )
        
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    @property
    def store(self) -> BaseSessionStore:
        """Get the underlying session store."""
        return self._store
    
    def get(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self._store.load(session_id)
    
    def get_or_create(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> Session:
        """Get existing session or create new one."""
        return self._store.get_or_create(session_id, user_id, agent_id)
    
    def create(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        **kwargs
    ) -> Session:
        """Create a new session."""
        session = Session(
            id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            **kwargs
        )
        self._store.save(session)
        return session
    
    def save(self, session: Session) -> None:
        """Save a session."""
        self._store.save(session)
    
    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        return self._store.delete(session_id)
    
    def list(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 100
    ) -> list:
        """List sessions."""
        return self._store.list_sessions(user_id, agent_id, limit)
    
    def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return self._store.exists(session_id)
    
    @classmethod
    def get_default(cls) -> "SessionManager":
        """Get the default session manager instance."""
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance
    
    @classmethod
    def set_default(cls, manager: "SessionManager") -> None:
        """Set the default session manager instance."""
        cls._default_instance = manager


# Convenience functions
def get_session(
    session_id: str,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None
) -> Session:
    """Get or create a session (convenience function).
    
    Example:
        session = get_session("user-123")
        session.add_user_message("Hello!")
        # Auto-saved on next get_session call
    """
    manager = SessionManager.get_default()
    return manager.get_or_create(session_id, user_id, agent_id)


def create_session(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    **kwargs
) -> Session:
    """Create a new session (convenience function).
    
    Example:
        session = create_session(user_id="user-123")
    """
    manager = SessionManager.get_default()
    return manager.create(session_id, user_id, agent_id, **kwargs)
