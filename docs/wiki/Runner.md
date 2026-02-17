# Runner

The `Runner` class executes agents and manages their lifecycle.

## Import

```python
from pyai import Runner
```

## Basic Usage

```python
from pyai import Agent, Runner

agent = Agent(name="Assistant", instructions="You are helpful")

# Sync execution
result = Runner.run_sync(agent, "Hello!")

# Async execution
result = await Runner.run(agent, "Hello!")
```

## Methods

### run_sync

```python
result = Runner.run_sync(
    agent,
    messages,
    context=None,
    max_turns=10
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | Agent | required | Agent to run |
| `messages` | str/list | required | Input messages |
| `context` | dict | None | Additional context |
| `max_turns` | int | 10 | Maximum conversation turns |

### run (async)

```python
result = await Runner.run(
    agent,
    messages,
    context=None,
    max_turns=10
)
```

### run_stream

```python
async for event in Runner.run_stream(agent, messages):
    print(event)
```

## Examples

### Simple Execution

```python
from pyai import Agent, Runner

agent = Agent(
    name="Math Helper",
    instructions="You help with math problems"
)

# Run synchronously
result = Runner.run_sync(agent, "What is 25 * 4?")
print(result.final_output)  # "100"
```

### With Context

```python
result = Runner.run_sync(
    agent,
    "Summarize the document",
    context={"document": "Long text here..."}
)
```

### Streaming

```python
import asyncio
from pyai import Agent, Runner

async def main():
    agent = Agent(name="Writer", instructions="You write stories")
    
    async for event in Runner.run_stream(agent, "Write a short story"):
        if hasattr(event, 'content'):
            print(event.content, end="", flush=True)

asyncio.run(main())
```

### With Tools

```python
from pyai import Agent, Runner, tool

@tool
def calculate(expression: str) -> str:
    """Calculate a math expression"""
    return str(eval(expression))

agent = Agent(
    name="Calculator",
    instructions="Use the calculate tool for math",
    tools=[calculate]
)

result = Runner.run_sync(agent, "What is 123 * 456?")
```

### Multi-turn Conversation

```python
# Conversation with history
messages = [
    {"role": "user", "content": "My name is Alice"},
    {"role": "assistant", "content": "Hi Alice!"},
    {"role": "user", "content": "What's my name?"}
]

result = Runner.run_sync(agent, messages)
# Will remember context
```

## Result Object

```python
result = Runner.run_sync(agent, "Hello")

print(result.final_output)    # Final response
print(result.messages)        # All messages
print(result.tool_calls)      # Tool calls made
print(result.tokens_used)     # Token consumption
print(result.cost)            # Estimated cost
```

## Error Handling

```python
from pyai import Runner, RunnerError

try:
    result = Runner.run_sync(agent, "Hello")
except RunnerError as e:
    print(f"Runner error: {e}")
```

## See Also

- [[Agent]] - Agent class
- [[Memory]] - Conversation memory
- [[Workflows]] - Multi-agent workflows
