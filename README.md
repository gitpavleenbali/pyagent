# PyAgent

**Build AI agents in Python with elegant simplicity.**

[![PyPI version](https://img.shields.io/badge/pypi-v0.4.0-blue)](https://pypi.org/project/pyagent/)
[![Python](https://img.shields.io/badge/python-3.10+-green)](https://python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-671%20passing-brightgreen)]()

PyAgent is a lightweight, model-agnostic SDK for building AI agents and multi-agent systems. From simple Q&A to complex autonomous workflows, PyAgent scales with your needs while keeping your code clean and maintainable.

> **The pandas of AI development** - What pandas did for data, PyAgent does for intelligence.

---

## Quick Start

### Installation

```bash
pip install pyagent
```

For Azure OpenAI with Azure AD authentication:
```bash
pip install pyagent[azure]
```

### Hello World

```python
from pyagent import Agent, Runner

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant."
)

result = Runner.run_sync(agent, "Write a haiku about Python.")
print(result.final_output)
```

Or use the simple one-liner API:

```python
from pyagent import ask

answer = ask("What is the capital of France?")
print(answer)  # Paris
```

---

## Features at a Glance

### Python-Based Tools

Create tools using simple decorators:

```python
from pyagent import Agent, Runner
from pyagent.skills import tool

@tool(description="Get weather for a city")
async def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny, 72F"

agent = Agent(
    name="WeatherBot",
    instructions="Help users with weather information.",
    tools=[get_weather]
)

result = Runner.run_sync(agent, "What's the weather in Tokyo?")
print(result.final_output)
# The weather in Tokyo is sunny, 72F
```

### Multiple Model Providers

Support for OpenAI, Azure OpenAI, Anthropic, and more:

```python
from pyagent import Agent
from pyagent.core import AzureOpenAIProvider, OpenAIProvider, LLMConfig

# OpenAI
openai_provider = OpenAIProvider(LLMConfig(
    api_key="sk-...",
    model="gpt-4o-mini"
))

# Azure OpenAI (with Azure AD - no API key needed!)
azure_provider = AzureOpenAIProvider(LLMConfig(
    api_base="https://your-resource.openai.azure.com/",
    model="gpt-4o-mini"
    # Uses DefaultAzureCredential automatically
))

agent = Agent(name="Assistant", llm=azure_provider)
```

### Multi-Agent Systems

Build systems where specialized agents collaborate:

```python
from pyagent import Agent
from pyagent.blueprint import Workflow, Step

# Create specialized agents
researcher = Agent(name="Researcher", instructions="Find information on topics.")
writer = Agent(name="Writer", instructions="Write clear, engaging content.")
editor = Agent(name="Editor", instructions="Review and improve writing.")

# Chain them in a workflow
workflow = (Workflow("ContentPipeline")
    .add_step(Step("research", researcher))
    .add_step(Step("write", writer))
    .add_step(Step("edit", editor))
    .build())
```

### Agent Handoffs

Transfer control between agents seamlessly:

```python
from pyagent import Agent, Runner

spanish_agent = Agent(
    name="SpanishAgent",
    instructions="You only speak Spanish."
)

english_agent = Agent(
    name="EnglishAgent",
    instructions="You only speak English."
)

triage_agent = Agent(
    name="TriageAgent",
    instructions="Route to the appropriate language agent.",
    handoffs=[spanish_agent, english_agent]
)

result = Runner.run_sync(triage_agent, "Hola, como estas?")
print(result.final_output)
# Hola! Estoy bien, gracias. Y tu?
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Model Agnostic** | OpenAI, Azure OpenAI, Anthropic, Ollama, and custom providers |
| **Agent Framework** | Build modular agents with tools, memory, and planning |
| **Multi-Agent Systems** | Orchestrate complex workflows with collaborating agents |
| **Plugin Ecosystem** | Extend with native functions, OpenAPI specs, or MCP tools |
| **Sessions & Memory** | SQLite and Redis session support for conversation history |
| **Streaming** | Real-time streaming responses |
| **Voice Support** | Voice input/output with OpenAI Realtime API |
| **RAG Built-in** | Document Q&A with vector database connectors |
| **Enterprise Ready** | Azure AD authentication, tracing, evaluation tools |

---

## Installation Options

```bash
# Core package
pip install pyagent

# With Azure support (Azure AD authentication)
pip install pyagent[azure]

# With voice support
pip install pyagent[voice]

# With Redis sessions
pip install pyagent[redis]

# Full installation
pip install pyagent[all]
```

---

## Model Providers

PyAgent supports multiple AI model providers:

| Provider | Status | Auth Methods |
|----------|--------|--------------|
| **OpenAI** | Supported | API Key |
| **Azure OpenAI** | Supported | API Key, Azure AD |
| **Anthropic** | Supported | API Key |
| **Ollama** | Supported | Local |
| **LiteLLM** | Supported | Various |
| **Custom** | Supported | Configurable |

### Azure OpenAI with Azure AD

For enterprise scenarios, use Azure AD authentication (no API key required):

```python
import os
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource.openai.azure.com/"
os.environ["AZURE_OPENAI_DEPLOYMENT"] = "gpt-4o-mini"

from pyagent import ask

# Automatically uses your Azure login (az login / VS Code)
answer = ask("Hello!")
```

---

## Documentation

- [Getting Started Guide](docs/QUICKSTART.md)
- [API Reference](docs/API_REFERENCE.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Azure Setup Guide](docs/AZURE_SETUP.md)
- [Examples](examples/)

---

## Examples

Explore the [examples/](examples/) directory:

| Example | Description |
|---------|-------------|
| `basic_agent.py` | Simple agent with tools |
| `multi_agent_workflow.py` | Multi-agent collaboration |
| `weather_app.py` | Real-world weather assistant |
| `smart_research_assistant.py` | RAG + research capabilities |
| `custom_skills.py` | Creating custom tools/skills |

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for:

- Bug reports and feature requests
- Development setup
- Pull request guidelines
- Code style guide

---

## Acknowledgements

PyAgent builds on the excellent work of the open-source community:

- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) - Runner pattern inspiration
- [Microsoft Semantic Kernel](https://github.com/microsoft/semantic-kernel) - Plugin architecture
- [Google ADK](https://github.com/google/adk-python) - Agent configuration patterns
- [Strands Agents](https://github.com/strands-agents/sdk-python) - Tool auto-discovery
- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) - Token counting utilities

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>PyAgent</strong> - The Intelligence Engine for Modern Applications
</p>
