# Agent

The `Agent` class is the heart of PYAI's agent framework.

---

## Basic Usage

```python
from pyai import Agent, Runner

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant."
)

result = Runner.run_sync(agent, "Hello, how are you?")
print(result.final_output)
```

---

## Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Agent identifier |
| `instructions` | `str` | System prompt / persona |
| `tools` | `list` | Functions the agent can call |
| `handoffs` | `list[Agent]` | Agents to transfer to |
| `llm` | `LLMProvider` | LLM provider instance |
| `memory` | `Memory` | Memory system |
| `memory_type` | `str` | `"conversation"` or `"vector"` |
| `config` | `AgentConfig` | Advanced configuration |

---

## With Tools

```python
from pyai import Agent, Runner
from pyai.skills import tool

@tool(description="Search the web")
async def search(query: str) -> str:
    return f"Results for: {query}"

@tool(description="Get weather")
async def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny, 72°F"

agent = Agent(
    name="ResearchBot",
    instructions="Help users find information.",
    tools=[search, get_weather]
)

result = Runner.run_sync(agent, "What's the weather in NYC?")
```

---

## With Handoffs

```python
from pyai import Agent, Runner

# Specialist agents
billing = Agent(name="Billing", instructions="Handle billing questions")
technical = Agent(name="Technical", instructions="Handle technical issues")
sales = Agent(name="Sales", instructions="Handle sales inquiries")

# Router agent
router = Agent(
    name="Router",
    instructions="Route to appropriate specialist",
    handoffs=[billing, technical, sales]
)

result = Runner.run_sync(router, "I have a question about my invoice")
# Automatically routes to billing agent
```

---

## With LLM Provider

### OpenAI

```python
from pyai import Agent
from pyai.core import OpenAIProvider, LLMConfig

provider = OpenAIProvider(LLMConfig(
    api_key="sk-...",
    model="gpt-4o-mini"
))

agent = Agent(name="Bot", instructions="...", llm=provider)
```

### Azure OpenAI

```python
from pyai.core import AzureOpenAIProvider, LLMConfig

# With API Key
provider = AzureOpenAIProvider(LLMConfig(
    api_key="your-key",
    api_base="https://your-resource.openai.azure.com/",
    model="gpt-4o-mini"
))

# With Azure AD (no API key needed!)
provider = AzureOpenAIProvider(LLMConfig(
    api_base="https://your-resource.openai.azure.com/",
    model="gpt-4o-mini"
    # Uses DefaultAzureCredential automatically
))

agent = Agent(name="Bot", instructions="...", llm=provider)
```

### Anthropic

```python
from pyai.core import AnthropicProvider, LLMConfig

provider = AnthropicProvider(LLMConfig(
    api_key="sk-ant-...",
    model="claude-3-sonnet"
))
```

### Ollama (Local)

```python
from pyai.core import OllamaProvider, LLMConfig

provider = OllamaProvider(LLMConfig(
    api_base="http://localhost:11434",
    model="llama2"
))
```

---

## With Memory

### Conversation Memory

```python
from pyai import Agent
from pyai.core import ConversationMemory

memory = ConversationMemory(max_messages=50)

agent = Agent(
    name="Bot",
    instructions="...",
    memory=memory
)

# Or use shorthand
agent = Agent(
    name="Bot",
    instructions="...",
    memory_type="conversation"
)
```

### Vector Memory

```python
from pyai.core import VectorMemory

memory = VectorMemory(provider="chromadb")

agent = Agent(
    name="Bot",
    instructions="...",
    memory=memory
)
```

---

## Advanced Configuration

```python
from pyai import Agent
from pyai.core import AgentConfig

config = AgentConfig(
    max_iterations=10,
    timeout=60,
    retry_count=3,
    retry_delay=1.0,
    temperature=0.7,
    max_tokens=4096,
)

agent = Agent(
    name="Bot",
    instructions="...",
    config=config
)
```

---

## Agent Response

The `Runner` returns a `RunResult`:

```python
result = Runner.run_sync(agent, "Hello")

result.status           # RunStatus.COMPLETED
result.final_output     # The final text response
result.messages         # Full conversation history
result.tool_calls       # Tools that were called
result.metadata         # Additional info (tokens, timing)
```

---

## Running Agents

### Synchronous

```python
result = Runner.run_sync(agent, "Hello")
```

### Asynchronous

```python
result = await Runner.run_async(agent, "Hello")
```

### Streaming

```python
from pyai.runner import StreamingRunner

async for event in StreamingRunner.stream(agent, "Hello"):
    if event.type == "token":
        print(event.data, end="", flush=True)
```

---

## Next Steps

- [[Runner]] - Execution patterns in detail
- [[Creating Tools]] - Build custom tools
- [[Handoffs]] - Agent-to-agent transfers
- [[Memory]] - Memory systems
