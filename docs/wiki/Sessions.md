# Sessions

Sessions provide persistent conversation storage for agents.

## Overview

PYAI supports multiple session backends:

| Backend | Description | Use Case |
|---------|-------------|----------|
| `SQLiteSession` | Local SQLite | Development, single-server |
| `RedisSession` | Redis | Production, multi-server |
| `InMemorySession` | In-memory | Testing |
| `PostgresSession` | PostgreSQL | Enterprise |

## SQLite Session

```python
from pyai.sessions import SQLiteSession

session = SQLiteSession("conversations.db")

# Use with agent
agent = Agent(
    name="Assistant",
    session=session
)
```

### Basic Usage

```python
from pyai import Agent, Runner
from pyai.sessions import SQLiteSession

# Create session
session = SQLiteSession(
    db_path="sessions.db",
    session_id="user-123"
)

agent = Agent(
    name="Assistant",
    instructions="Remember user preferences",
    session=session
)

# Conversation persists
result1 = Runner.run_sync(agent, "My name is Alice")
result2 = Runner.run_sync(agent, "What's my name?")
# "Your name is Alice"

# Later session...
agent2 = Agent(name="Assistant", session=session)
result = Runner.run_sync(agent2, "Do you remember me?")
# "Yes, you're Alice!"
```

## Redis Session

```python
from pyai.sessions import RedisSession

session = RedisSession(
    host="localhost",
    port=6379,
    session_id="user-123"
)

agent = Agent(
    name="Assistant",
    session=session
)
```

### With Authentication

```python
session = RedisSession(
    host="redis.example.com",
    port=6379,
    password="your-password",
    ssl=True,
    session_id="user-123"
)
```

### TTL (Time-to-Live)

```python
session = RedisSession(
    host="localhost",
    session_id="user-123",
    ttl=3600  # Expire after 1 hour
)
```

## PostgreSQL Session

```python
from pyai.sessions import PostgresSession

session = PostgresSession(
    connection_string="postgresql://user:pass@host:5432/db",
    session_id="user-123"
)
```

## Session Operations

### Get/Set Messages

```python
# Get conversation history
messages = session.get_messages()

# Add message
session.add_message("user", "Hello")
session.add_message("assistant", "Hi there!")

# Clear session
session.clear()
```

### Get/Set Metadata

```python
# Store metadata
session.set("user_name", "Alice")
session.set("preferences", {"theme": "dark"})

# Retrieve metadata
name = session.get("user_name")
prefs = session.get("preferences")
```

### Multiple Sessions

```python
from pyai.sessions import SQLiteSession

# Different session per user
def get_session(user_id: str):
    return SQLiteSession(
        db_path="sessions.db",
        session_id=f"user-{user_id}"
    )

# Usage
session = get_session("alice")
agent = Agent(name="Assistant", session=session)
```

## Session Context Manager

```python
from pyai.sessions import SQLiteSession

with SQLiteSession("db.sqlite", "user-123") as session:
    agent = Agent(name="Assistant", session=session)
    result = Runner.run_sync(agent, "Hello")
# Session automatically saved
```

## Export/Import

```python
# Export session
data = session.export()

# Import session
new_session = SQLiteSession("new.db", "user-123")
new_session.import_data(data)
```

## Session Events

```python
from pyai.sessions import SessionEvent

session.on(SessionEvent.MESSAGE_ADDED, lambda msg: print(f"New: {msg}"))
session.on(SessionEvent.SESSION_CLEARED, lambda: print("Cleared"))
```

## Examples

### Multi-user Chat

```python
from pyai import Agent, Runner
from pyai.sessions import RedisSession

def create_agent_for_user(user_id: str) -> Agent:
    session = RedisSession(
        host="redis",
        session_id=f"chat:{user_id}",
        ttl=86400  # 24 hours
    )
    return Agent(
        name="ChatBot",
        instructions="Friendly assistant",
        session=session
    )

# Per-user agents
alice_agent = create_agent_for_user("alice")
bob_agent = create_agent_for_user("bob")
```

### Session Migration

```python
from pyai.sessions import SQLiteSession, RedisSession

# Migrate from SQLite to Redis
sqlite = SQLiteSession("old.db", "user-123")
redis = RedisSession(host="localhost", session_id="user-123")

# Export and import
data = sqlite.export()
redis.import_data(data)
```

## See Also

- [[Memory]] - Memory systems
- [[Kernel-Registry]] - Service management
- [[Agent]] - Agent class
