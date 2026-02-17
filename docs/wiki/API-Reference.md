# API Reference

Complete API reference for pyai.

## Quick Links

| Module | Description |
|--------|-------------|
| [Configuration](#configuration) | Global configuration |
| [One-Liner Functions](#one-liner-functions) | Simple AI operations |
| [Agent](#agent-class) | Agent class |
| [Modules](#modules) | Module reference |

---

## Configuration

### configure()

Configure pyai globally:

```python
import pyai

pyai.configure(
    api_key="sk-...",           # API key
    provider="openai",          # "openai", "anthropic", "azure"
    model="gpt-4o-mini",        # Default model
    azure_endpoint="...",       # Azure OpenAI endpoint
    temperature=0.7,            # Default temperature
    max_tokens=2048,            # Default max tokens
)
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint |
| `AZURE_OPENAI_DEPLOYMENT` | Azure deployment name |

---

## One-Liner Functions

### ask()

Ask any question:

```python
from pyai import ask

answer = ask("What is the capital of France?")

# With options
answer = ask(
    "Explain quantum computing",
    detailed=True,
    format="bullet",
    model="gpt-4"
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | `str` | required | Question to ask |
| `detailed` | `bool` | `False` | Comprehensive answer |
| `concise` | `bool` | `False` | Brief answer |
| `format` | `str` | `None` | "bullet", "numbered", "markdown" |
| `model` | `str` | `None` | Override model |

### research()

Deep research on any topic:

```python
from pyai import research

result = research("quantum computing")
print(result.summary)
print(result.key_points)
print(result.insights)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `topic` | `str` | required | Research topic |
| `quick` | `bool` | `False` | Quick summary only |
| `depth` | `str` | `"medium"` | "shallow", "medium", "deep" |
| `focus` | `str` | `None` | Focus area |

### summarize()

Summarize text or documents:

```python
from pyai import summarize

summary = summarize("./document.pdf")
summary = summarize(long_text, max_words=100)
```

### extract()

Extract structured data:

```python
from pyai import extract

data = extract(text, fields=["name", "date", "amount"])
```

### translate()

Translate text:

```python
from pyai import translate

result = translate("Hello world", to="spanish")
```

### generate()

Generate content:

```python
from pyai import generate

content = generate("blog post about AI", style="professional")
```

### chat()

Interactive chat:

```python
from pyai import chat

session = chat(system="You are a helpful assistant.")
response = session.send("Hello!")
```

---

## Agent Class

Create intelligent agents:

```python
from pyai import Agent

agent = Agent(
    name="my-agent",
    instructions="You are a helpful assistant.",
    model="gpt-4",
    skills=[my_skill],
    memory=True
)

result = agent.run("What can you help me with?")
```

See [Agent](Agent) for full documentation.

---

## Modules

### Easy Module

| Function | Description |
|----------|-------------|
| `ask()` | Ask questions |
| `research()` | Research topics |
| `summarize()` | Summarize content |
| `extract()` | Extract data |
| `translate()` | Translate text |
| `generate()` | Generate content |
| `chat()` | Interactive chat |
| `agent()` | Create agents |

### Core Module

| Class | Description |
|-------|-------------|
| `Agent` | Agent class |
| `LLMProvider` | LLM interface |
| `Memory` | Memory management |

### Blueprint Module

| Class | Description |
|-------|-------------|
| `Workflow` | Define workflows |
| `Pipeline` | Pipeline processing |
| `Blueprint` | Complex patterns |

### Integrations

| Module | Description |
|--------|-------------|
| `vectordb` | Vector databases |
| `sessions` | Session storage |
| `plugins` | Plugin system |

---

## Return Types

### ResearchResult

```python
@dataclass
class ResearchResult:
    summary: str
    key_points: List[str]
    insights: List[str]
    sources: List[str]
    confidence: float
```

### ExtractResult

```python
@dataclass
class ExtractResult:
    fields: Dict[str, Any]
    confidence: float
```

---

## Error Handling

```python
from pyai.errors import (
    pyaiError,
    ConfigurationError,
    APIError,
    RateLimitError,
    ValidationError
)

try:
    result = ask("...")
except RateLimitError as e:
    print(f"Rate limit: retry after {e.retry_after}s")
except APIError as e:
    print(f"API error: {e.message}")
```

---

## Full Documentation

For complete API documentation, see:
- [Quick-Start](Quick-Start) - Getting started guide
- [Configuration](Configuration) - Detailed configuration
- [Agent](Agent) - Agent class reference
- [Workflows](Workflows) - Workflow patterns
