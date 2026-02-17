# ask

The `ask` function is the simplest way to get answers from an AI model.

## Import

```python
from pyai import ask
```

## Basic Usage

```python
# Simple question
answer = ask("What is the capital of France?")
print(answer)  # "Paris"

# With context
answer = ask("Summarize this", context="Long text here...")

# Async version
answer = await ask.async_("What is Python?")
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | str | required | The question to ask |
| `context` | str | None | Additional context to include |
| `model` | str | None | Override default model |
| `temperature` | float | 0.7 | Response creativity (0-1) |

## Return Value

Returns a string containing the AI's response.

## Examples

### Simple Questions

```python
from pyai import ask

# Factual questions
answer = ask("What is machine learning?")

# With specific model
answer = ask("Explain quantum computing", model="gpt-4")

# With custom temperature
answer = ask("Write a creative story", temperature=0.9)
```

### With Context

```python
document = """
PYAI is a Python SDK for building AI agents.
It supports multiple providers and enterprise features.
"""

answer = ask("What does PYAI support?", context=document)
# "PYAI supports multiple providers and enterprise features."
```

### Async Usage

```python
import asyncio
from pyai import ask

async def main():
    answer = await ask.async_("What is PYAI?")
    print(answer)

asyncio.run(main())
```

## Configuration

Configure default behavior using environment variables:

```bash
OPENAI_API_KEY=your-key
# or
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

## See Also

- [[research]] - Deep topic research
- [[summarize]] - Text summarization
- [[chat]] - Interactive conversation
