# AGENTS.md - pyai Project Context

This file provides context for AI coding assistants (GitHub Copilot, Cursor, Claude, etc.) working on this codebase.

## Project Overview

**pyai** is a Python SDK for building AI agents and multi-agent systems. It provides:
- Simple one-liner APIs (`ask`, `summarize`, `research`)
- Full agent framework (`Agent`, `Runner`, `Workflow`)
- Multi-provider support (OpenAI, Azure OpenAI, Anthropic, Ollama)
- Enterprise features (Azure AD auth, sessions, evaluation)

## Repository Structure

```
pyai/
├── src/pyai/          # Main package
│   ├── core/             # Core agent, LLM providers, memory
│   ├── easy/             # Simple one-liner APIs
│   ├── runner/           # Agent execution (Runner pattern)
│   ├── blueprint/        # Workflows, patterns (Chain, Supervisor)
│   ├── skills/           # Tools and skills system
│   ├── plugins/          # Plugin architecture
│   ├── kernel/           # Service registry (SK pattern)
│   ├── config/           # YAML/JSON agent configuration
│   ├── sessions/         # SQLite/Redis session storage
│   ├── tokens/           # Token counting utilities
│   ├── evaluation/       # Agent evaluation framework
│   ├── openapi/          # OpenAPI tool generation
│   ├── multimodal/       # Vision, audio support
│   ├── voice/            # Voice streaming
│   ├── vectordb/         # Vector database connectors
│   ├── a2a/              # Agent-to-Agent protocol
│   └── devui/            # Development UI
├── tests/                # Test suite (671 tests)
├── examples/             # Usage examples
└── docs/                 # Documentation
```

## Key Patterns

### 1. Simple API (easy module)
```python
from pyai import ask, summarize, research
answer = ask("What is Python?")
```

### 2. Agent API
```python
from pyai import Agent, Runner
agent = Agent(name="Assistant", instructions="You are helpful.")
result = Runner.run_sync(agent, "Hello")
```

### 3. Kernel Pattern (service registry)
```python
from pyai.kernel import Kernel
kernel = Kernel()
kernel.add_service(provider, service_id="default")
```

## Development Commands

```bash
# Run tests
pytest tests/ -v

# Run specific test file
pytest tests/test_kernel.py -v

# Type checking
mypy src/pyai

# Linting
ruff check src/

# Format
ruff format src/
```

## Azure OpenAI Authentication

The SDK supports Azure AD authentication via DefaultAzureCredential:
```python
# Set environment variables
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
# No API key needed - uses az login credentials
```

## Testing Guidelines

- All new features need tests
- Tests use pytest with async support
- Mock external API calls
- Target 671+ tests passing

## Code Style

- Python 3.10+
- Type hints required
- Docstrings for public APIs
- Follow existing patterns in codebase
