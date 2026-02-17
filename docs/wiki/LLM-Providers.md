# LLM Providers

PYAI supports multiple LLM providers for maximum flexibility.

## Supported Providers

| Provider | Models | Auth |
|----------|--------|------|
| **OpenAI** | GPT-4, GPT-3.5 | API Key |
| **Azure OpenAI** | GPT-4, GPT-3.5 | API Key or Azure AD |
| **Anthropic** | Claude 3.5, Claude 3 | API Key |
| **Ollama** | Llama, Mistral, etc. | Local |

## Configuration

### OpenAI

```python
# Environment variables
OPENAI_API_KEY=sk-...

# Or in code
from pyai import configure
configure(provider="openai", api_key="sk-...")
```

### Azure OpenAI

```python
# Environment variables
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini

# Or with Azure AD (no API key needed)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
# Uses DefaultAzureCredential
```

### Anthropic

```python
# Environment variables
ANTHROPIC_API_KEY=sk-ant-...

# Or in code
from pyai import configure
configure(provider="anthropic", api_key="sk-ant-...")
```

### Ollama (Local)

```python
# Start Ollama server
# ollama serve

# Environment variables
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2

# Or in code
from pyai import configure
configure(provider="ollama", model="llama2")
```

## Usage Examples

### Switch Providers

```python
from pyai import Agent

# OpenAI agent
openai_agent = Agent(
    name="OpenAI Agent",
    model="gpt-4",
    provider="openai"
)

# Azure agent
azure_agent = Agent(
    name="Azure Agent",
    model="gpt-4o-mini",
    provider="azure"
)

# Anthropic agent
claude_agent = Agent(
    name="Claude Agent",
    model="claude-3-sonnet-20240229",
    provider="anthropic"
)

# Local Ollama
local_agent = Agent(
    name="Local Agent",
    model="llama2",
    provider="ollama"
)
```

### Using Easy API

```python
from pyai import ask, configure

# Use OpenAI (default)
answer = ask("What is Python?")

# Switch to Claude
configure(provider="anthropic")
answer = ask("What is Python?")

# Switch to local
configure(provider="ollama", model="llama2")
answer = ask("What is Python?")
```

### Custom Provider

```python
from pyai.core import LLMProvider

class CustomProvider(LLMProvider):
    """Custom LLM provider implementation"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def generate(self, messages, **kwargs):
        # Implement API call
        response = await self._call_api(messages)
        return response
    
    async def stream(self, messages, **kwargs):
        # Implement streaming
        async for chunk in self._stream_api(messages):
            yield chunk
```

## Provider Features

### OpenAI

```python
from pyai.core import OpenAIProvider

provider = OpenAIProvider(
    api_key="sk-...",
    organization="org-..."
)

# Function calling
result = await provider.generate(
    messages,
    tools=[my_tool],
    tool_choice="auto"
)

# Vision
result = await provider.generate(
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "..."}}
        ]
    }]
)
```

### Azure OpenAI

```python
from pyai.core import AzureOpenAIProvider

# With API key
provider = AzureOpenAIProvider(
    endpoint="https://your-resource.openai.azure.com/",
    api_key="your-key",
    deployment="gpt-4o-mini"
)

# With Azure AD (enterprise)
provider = AzureOpenAIProvider(
    endpoint="https://your-resource.openai.azure.com/",
    deployment="gpt-4o-mini",
    use_azure_ad=True  # Uses DefaultAzureCredential
)
```

### Anthropic

```python
from pyai.core import AnthropicProvider

provider = AnthropicProvider(api_key="sk-ant-...")

# With extended context
result = await provider.generate(
    messages,
    model="claude-3-opus-20240229",
    max_tokens=4096
)
```

### Ollama

```python
from pyai.core import OllamaProvider

provider = OllamaProvider(
    host="http://localhost:11434",
    model="llama2"
)

# List available models
models = await provider.list_models()
```

## Provider Selection Priority

1. Explicit provider parameter
2. `pyai_PROVIDER` env var
3. Available credentials (in order: OpenAI, Azure, Anthropic, Ollama)

## See Also

- [[Agent]] - Agent class
- [[Configuration]] - Configuration options
- [[Azure-AD-Auth]] - Azure AD authentication
