# Memory

The `Memory` system provides conversation history and vector memory for agents.

## Import

```python
from pyai.core import Memory, VectorMemory
```

## Memory Types

| Type | Description |
|------|-------------|
| `Memory` | Conversation history (in-memory) |
| `VectorMemory` | Semantic search memory |
| `PersistentMemory` | SQLite/Redis backed |

## Basic Usage

```python
from pyai import Agent
from pyai.core import Memory

# Agent with memory
agent = Agent(
    name="Assistant",
    instructions="Remember user preferences",
    memory=Memory(max_messages=100)
)
```

## Conversation Memory

### Basic Memory

```python
from pyai.core import Memory

memory = Memory(max_messages=50)

# Add messages
memory.add("user", "My favorite color is blue")
memory.add("assistant", "I'll remember that!")

# Get history
history = memory.get_messages()
```

### With Agent

```python
agent = Agent(
    name="Personal Assistant",
    instructions="Remember user information",
    memory=Memory()
)

# Memory persists across runs
result1 = Runner.run_sync(agent, "My name is Alice")
result2 = Runner.run_sync(agent, "What's my name?")
# Remembers: "Your name is Alice"
```

## Vector Memory

```python
from pyai.core import VectorMemory

# Create vector memory
vmem = VectorMemory(
    embedding_model="text-embedding-3-small",
    max_results=5
)

# Store information
vmem.store("PYAI is a Python SDK for AI agents")
vmem.store("It supports multiple LLM providers")
vmem.store("Enterprise features include Azure AD auth")

# Search
results = vmem.search("What is PYAI?")
for result in results:
    print(f"{result.score}: {result.content}")
```

### With Agent

```python
from pyai.core import VectorMemory

agent = Agent(
    name="Knowledge Agent",
    instructions="Use your knowledge to answer",
    memory=VectorMemory()
)

# Agent can retrieve relevant memories
```

## Persistent Memory

### SQLite

```python
from pyai.sessions import SQLiteSession

session = SQLiteSession("memory.db")
agent = Agent(
    name="Assistant",
    session=session
)

# Memory persists to disk
```

### Redis

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

## Memory Methods

| Method | Description |
|--------|-------------|
| `add(role, content)` | Add message |
| `get_messages()` | Get all messages |
| `search(query)` | Semantic search (VectorMemory) |
| `clear()` | Clear all messages |
| `save(path)` | Save to file |
| `load(path)` | Load from file |

## Examples

### Summarization Memory

```python
from pyai.core import Memory

class SummarizingMemory(Memory):
    """Memory that summarizes old conversations"""
    
    def add(self, role, content):
        super().add(role, content)
        if len(self.messages) > 20:
            self._summarize_old_messages()
```

### Hybrid Memory

```python
from pyai.core import Memory, VectorMemory

class HybridMemory:
    """Combines recent + semantic memory"""
    
    def __init__(self):
        self.recent = Memory(max_messages=10)
        self.long_term = VectorMemory()
    
    def add(self, role, content):
        self.recent.add(role, content)
        if role == "user":
            self.long_term.store(content)
```

## Configuration

```python
memory = Memory(
    max_messages=100,      # Message limit
    max_tokens=4000,       # Token limit
    summarize=True,        # Auto-summarize
    system_prompt=False    # Exclude system
)
```

## See Also

- [[Agent]] - Agent class
- [[Runner]] - Execution engine
- [[VectorDB-Module]] - Vector databases
