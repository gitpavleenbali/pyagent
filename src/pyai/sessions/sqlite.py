# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
SQLite Session Store

Persistent session storage using SQLite.
Like OpenAI Agents SDK's SQLiteSession.
"""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .base import BaseSessionStore, Session


class SQLiteSessionStore(BaseSessionStore):
    """SQLite-backed session store.

    Good for:
    - Local development
    - Single-server deployments
    - Persistent sessions across restarts

    Example:
        store = SQLiteSessionStore("sessions.db")
        session = Session(id="my-session", user_id="user-123")
        session.add_user_message("Hello!")
        store.save(session)

        # Later...
        loaded = store.load("my-session")
        print(loaded.messages)
    """

    def __init__(self, db_path: str = "pyai_sessions.db"):
        """Initialize SQLite session store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                agent_id TEXT,
                state TEXT DEFAULT 'active',
                metadata TEXT DEFAULT '{}',
                context TEXT DEFAULT '{}',
                created_at TEXT,
                updated_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TEXT,
                metadata TEXT DEFAULT '{}',
                tool_calls TEXT,
                tool_call_id TEXT,
                position INTEGER,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session_user ON sessions(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session_agent ON sessions(agent_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_message_session ON messages(session_id)")
        conn.commit()

    def save(self, session: Session) -> None:
        """Save a session to SQLite."""
        conn = self._get_connection()

        # Upsert session
        conn.execute(
            """
            INSERT OR REPLACE INTO sessions
            (id, user_id, agent_id, state, metadata, context, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                session.id,
                session.user_id,
                session.agent_id,
                session.state.value,
                json.dumps(session.metadata),
                json.dumps(session.context),
                session.created_at.isoformat(),
                datetime.utcnow().isoformat(),
            ),
        )

        # Delete existing messages
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session.id,))

        # Insert messages
        for i, msg in enumerate(session.messages):
            conn.execute(
                """
                INSERT INTO messages
                (id, session_id, role, content, timestamp, metadata, tool_calls, tool_call_id, position)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    msg.id,
                    session.id,
                    msg.role,
                    msg.content,
                    msg.timestamp.isoformat(),
                    json.dumps(msg.metadata),
                    json.dumps(msg.tool_calls) if msg.tool_calls else None,
                    msg.tool_call_id,
                    i,
                ),
            )

        conn.commit()

    def load(self, session_id: str) -> Optional[Session]:
        """Load a session from SQLite."""
        conn = self._get_connection()

        # Load session
        cursor = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        # Load messages
        cursor = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY position", (session_id,)
        )
        messages = []
        for msg_row in cursor.fetchall():
            from .base import SessionMessage

            tool_calls = msg_row["tool_calls"]
            if tool_calls:
                tool_calls = json.loads(tool_calls)

            messages.append(
                SessionMessage(
                    id=msg_row["id"],
                    role=msg_row["role"],
                    content=msg_row["content"],
                    timestamp=datetime.fromisoformat(msg_row["timestamp"]),
                    metadata=json.loads(msg_row["metadata"] or "{}"),
                    tool_calls=tool_calls,
                    tool_call_id=msg_row["tool_call_id"],
                )
            )

        from .base import SessionState

        return Session(
            id=row["id"],
            user_id=row["user_id"],
            agent_id=row["agent_id"],
            messages=messages,
            state=SessionState(row["state"]),
            metadata=json.loads(row["metadata"] or "{}"),
            context=json.loads(row["context"] or "{}"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def delete(self, session_id: str) -> bool:
        """Delete a session from SQLite."""
        conn = self._get_connection()
        cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        return cursor.rowcount > 0

    def list_sessions(
        self, user_id: Optional[str] = None, agent_id: Optional[str] = None, limit: int = 100
    ) -> List[Session]:
        """List sessions with optional filtering."""
        conn = self._get_connection()

        query = "SELECT id FROM sessions WHERE 1=1"
        params = []

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)

        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, params)

        sessions = []
        for row in cursor.fetchall():
            session = self.load(row["id"])
            if session:
                sessions.append(session)

        return sessions

    def clear(self) -> None:
        """Clear all sessions."""
        conn = self._get_connection()
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM sessions")
        conn.commit()

    @property
    def count(self) -> int:
        """Get number of stored sessions."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM sessions")
        return cursor.fetchone()[0]
