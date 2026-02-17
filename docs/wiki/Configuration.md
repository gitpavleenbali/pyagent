# Configuration

Set up PYAI for your environment.

---

## Environment Variables

### OpenAI

```bash
export OPENAI_API_KEY=sk-your-api-key
```

### Azure OpenAI

```bash
# With API Key
export AZURE_OPENAI_API_KEY=your-api-key
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini

# With Azure AD (recommended - no API key needed!)
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
# Uses your az login credentials automatically
```

### Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-your-api-key
```

### Ollama (Local)

```bash
# No configuration needed for default
# Or specify custom endpoint
export OLLAMA_HOST=http://localhost:11434
```

---

## Programmatic Configuration

```python
import pyai

pyai.configure(
    api_key="sk-...",
    model="gpt-4o",
    temperature=0.7
)
```

---

## Provider-Specific Configuration

### OpenAI

```python
from pyai.core import OpenAIProvider, LLMConfig

provider = OpenAIProvider(LLMConfig(
    api_key="sk-...",
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=4096,
))
```

### Azure OpenAI

```python
from pyai.core import AzureOpenAIProvider, LLMConfig

# API Key auth
provider = AzureOpenAIProvider(LLMConfig(
    api_key="your-key",
    api_base="https://your-resource.openai.azure.com/",
    model="gpt-4o-mini",
    api_version="2024-02-15-preview",
))

# Azure AD auth (recommended for enterprise)
provider = AzureOpenAIProvider(LLMConfig(
    api_base="https://your-resource.openai.azure.com/",
    model="gpt-4o-mini",
    # No api_key = uses DefaultAzureCredential
))
```

### Anthropic

```python
from pyai.core import AnthropicProvider, LLMConfig

provider = AnthropicProvider(LLMConfig(
    api_key="sk-ant-...",
    model="claude-3-sonnet-20240229",
    max_tokens=4096,
))
```

---

## Model Selection

PYAI automatically selects the appropriate model based on environment:

1. If `AZURE_OPENAI_ENDPOINT` is set → Azure OpenAI
2. If `OPENAI_API_KEY` is set → OpenAI
3. If `ANTHROPIC_API_KEY` is set → Anthropic
4. If Ollama is running → Ollama

Override with explicit provider:

```python
from pyai import Agent
from pyai.core import AzureOpenAIProvider, LLMConfig

agent = Agent(
    name="Bot",
    instructions="...",
    llm=AzureOpenAIProvider(LLMConfig(...))
)
```

---

## YAML Configuration

Define agents in YAML files:

```yaml
# agents/research_assistant.yaml
name: ResearchAssistant
instructions: |
  You are a research assistant that helps users find information.
  Be thorough and cite your sources.
model: gpt-4o-mini
temperature: 0.7
tools:
  - web_search
  - summarize
memory:
  type: conversation
  max_messages: 50
```

Load and use:

```python
from pyai.config import load_agent, AgentBuilder

config = load_agent("agents/research_assistant.yaml")
agent = AgentBuilder.from_config(config).build()
```

---

## Default Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `model` | `gpt-4o-mini` | Default model |
| `temperature` | `0.7` | Creativity level |
| `max_tokens` | `4096` | Max output tokens |
| `timeout` | `60` | Request timeout (seconds) |
| `retry_count` | `3` | Retry attempts |
| `retry_delay` | `1.0` | Delay between retries |

---

## Next Steps

- [[Quick Start]] - Run your first program
- [[Azure AD Auth]] - Enterprise authentication
- [[Agent]] - Create agents
