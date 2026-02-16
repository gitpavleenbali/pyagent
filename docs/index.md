# PyAgent

**Build AI agents in Python with elegant simplicity.**

PyAgent is a lightweight, model-agnostic SDK for building AI agents and multi-agent systems. From simple Q&A to complex autonomous workflows, PyAgent scales with your needs while keeping your code clean and maintainable.

## Quick Start

```python
from pyagent import ask

answer = ask("What is the capital of France?")
print(answer)  # Paris
```

Or build a full agent:

```python
from pyagent import Agent, Runner

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant."
)

result = Runner.run_sync(agent, "Write a haiku about Python.")
print(result.final_output)
```

## Installation

```bash
pip install pyagent
```

For Azure OpenAI with Azure AD:
```bash
pip install pyagent[azure]
```

## Key Features

- **Model Agnostic** - OpenAI, Azure OpenAI, Anthropic, Ollama
- **Simple APIs** - One-liners like `ask()`, `summarize()`, `research()`
- **Full Agent Framework** - Agent, Runner, Workflow, Skills
- **Multi-Agent Systems** - Chain, Supervisor, and custom patterns
- **Enterprise Ready** - Azure AD auth, sessions, evaluation

## Next Steps

- [Quick Start Guide](QUICKSTART.md)
- [Azure Setup](AZURE_SETUP.md)
- [API Reference](API_REFERENCE.md)
- [Examples](examples.md)
