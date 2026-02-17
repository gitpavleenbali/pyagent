# chat

The `chat` function provides interactive conversation sessions.

## Import

```python
from pyai import chat
```

## Basic Usage

```python
# Start chat session
session = chat()

# Send messages
response = session.send("Hello!")
response = session.send("What's your name?")
```

## Quick Start

```python
from pyai import chat

# Create session
session = chat()

# Conversation
print(session.send("Hi, I need help with Python"))
print(session.send("How do I read a file?"))
print(session.send("Can you show an example?"))

# Session maintains context
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `system` | str | None | System prompt |
| `model` | str | None | Model to use |
| `temperature` | float | 0.7 | Response creativity |
| `memory` | bool | True | Remember conversation |

## Examples

### Basic Chat

```python
from pyai import chat

session = chat()

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = session.send(user_input)
    print(f"AI: {response}")
```

### With System Prompt

```python
session = chat(
    system="You are a helpful Python tutor. Explain concepts simply."
)

response = session.send("What is a decorator?")
```

### With Custom Model

```python
session = chat(
    model="gpt-4",
    temperature=0.5
)

response = session.send("Help me design an API")
```

### Context Manager

```python
from pyai import chat

with chat() as session:
    print(session.send("Hello"))
    print(session.send("Tell me a joke"))
# Session automatically cleaned up
```

### Conversation History

```python
session = chat()
session.send("My name is Alice")
session.send("I work at Acme Corp")

# Later in conversation
response = session.send("What's my name?")
# "Your name is Alice"

# Access history
for msg in session.history:
    print(f"{msg['role']}: {msg['content']}")
```

### Clear and Reset

```python
session = chat()
session.send("Remember: secret code is 1234")

# Clear history
session.clear()

session.send("What's the secret code?")
# Won't remember
```

### Streaming Responses

```python
session = chat()

# Stream response
for chunk in session.stream("Write a long story"):
    print(chunk, end="", flush=True)
print()
```

### Async Usage

```python
import asyncio
from pyai import chat

async def main():
    session = chat()
    response = await session.send_async("Hello!")
    print(response)

asyncio.run(main())
```

## Session Methods

| Method | Description |
|--------|-------------|
| `send(message)` | Send message and get response |
| `stream(message)` | Stream response |
| `clear()` | Clear conversation history |
| `save(path)` | Save session to file |
| `load(path)` | Load session from file |

## Persistence

```python
session = chat()
session.send("Important context here")

# Save session
session.save("session.json")

# Later, restore session
restored = chat.load("session.json")
restored.send("Do you remember?")
```

## See Also

- [[ask]] - Single question answering
- [[Agent]] - Full agent framework
- [[Voice-Module]] - Voice chat
