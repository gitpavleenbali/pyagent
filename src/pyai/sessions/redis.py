# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Redis Session Store

Distributed session storage using Redis.
Like OpenAI Agents SDK's RedisSession.
"""

from typing import List, Optional

from .base import BaseSessionStore, Session


class RedisSessionStore(BaseSessionStore):
    """Redis-backed session store.

    Good for:
    - Multi-server deployments
    - Distributed applications
    - High availability requirements
    - Session sharing across services

    Example:
        store = RedisSessionStore(host="redis.example.com")
        session = Session(id="my-session")
        store.save(session)

        # From another server...
        loaded = store.load("my-session")
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "pyai:session:",
        ttl: Optional[int] = None,
        **kwargs,
    ):
        """Initialize Redis session store.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            prefix: Key prefix for sessions
            ttl: Session TTL in seconds (None = no expiry)
            **kwargs: Additional Redis connection arguments
        """
        self.prefix = prefix
        self.ttl = ttl
        self._client = None

        self._connection_kwargs = {
            "host": host,
            "port": port,
            "db": db,
            "password": password,
            "decode_responses": True,
            **kwargs,
        }

    def _get_client(self):
        """Get or create Redis client."""
        if self._client is None:
            try:
                import redis
            except ImportError:
                raise ImportError("redis package required. Install with: pip install redis")

            self._client = redis.Redis(**self._connection_kwargs)
        return self._client

    def _key(self, session_id: str) -> str:
        """Get Redis key for a session."""
        return f"{self.prefix}{session_id}"

    def _index_key(self, index_type: str, value: str) -> str:
        """Get Redis key for an index."""
        return f"{self.prefix}index:{index_type}:{value}"

    def save(self, session: Session) -> None:
        """Save a session to Redis."""
        client = self._get_client()

        key = self._key(session.id)
        data = session.to_json()

        if self.ttl:
            client.setex(key, self.ttl, data)
        else:
            client.set(key, data)

        # Update indexes
        if session.user_id:
            client.sadd(self._index_key("user", session.user_id), session.id)
        if session.agent_id:
            client.sadd(self._index_key("agent", session.agent_id), session.id)

    def load(self, session_id: str) -> Optional[Session]:
        """Load a session from Redis."""
        client = self._get_client()

        data = client.get(self._key(session_id))
        if data is None:
            return None

        return Session.from_json(data)

    def delete(self, session_id: str) -> bool:
        """Delete a session from Redis."""
        client = self._get_client()

        # Load session to get user_id/agent_id for index cleanup
        session = self.load(session_id)
        if session:
            if session.user_id:
                client.srem(self._index_key("user", session.user_id), session_id)
            if session.agent_id:
                client.srem(self._index_key("agent", session.agent_id), session_id)

        return client.delete(self._key(session_id)) > 0

    def list_sessions(
        self, user_id: Optional[str] = None, agent_id: Optional[str] = None, limit: int = 100
    ) -> List[Session]:
        """List sessions with optional filtering."""
        client = self._get_client()

        session_ids = set()

        if user_id:
            session_ids = client.smembers(self._index_key("user", user_id))
        elif agent_id:
            session_ids = client.smembers(self._index_key("agent", agent_id))
        else:
            # Scan all session keys
            cursor = 0
            pattern = f"{self.prefix}*"
            while True:
                cursor, keys = client.scan(cursor, match=pattern, count=100)
                for key in keys:
                    if ":index:" not in key:
                        session_id = key.replace(self.prefix, "")
                        session_ids.add(session_id)
                if cursor == 0:
                    break

        # Load sessions
        sessions = []
        for session_id in list(session_ids)[:limit]:
            session = self.load(session_id)
            if session:
                if agent_id and user_id:
                    if session.user_id == user_id and session.agent_id == agent_id:
                        sessions.append(session)
                elif agent_id:
                    if session.agent_id == agent_id:
                        sessions.append(session)
                else:
                    sessions.append(session)

        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.updated_at, reverse=True)

        return sessions[:limit]

    def clear(self) -> None:
        """Clear all sessions."""
        client = self._get_client()

        cursor = 0
        pattern = f"{self.prefix}*"
        while True:
            cursor, keys = client.scan(cursor, match=pattern, count=100)
            if keys:
                client.delete(*keys)
            if cursor == 0:
                break

    def ping(self) -> bool:
        """Check if Redis is available."""
        try:
            return self._get_client().ping()
        except Exception:
            return False
