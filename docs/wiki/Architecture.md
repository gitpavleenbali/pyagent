# Architecture

PyAgent's architecture is designed around the concept of being the "Pandas of AI" - making complex AI operations simple.

## Vision

Just as pandas revolutionized data analysis with one-liners, PyAgent revolutionizes AI development.

## The 3-Dimensional Library

Traditional libraries are 2-dimensional (Input → Output). PyAgent is 3-dimensional:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PYAGENT 3D ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    USER LAYER (Dimension 1 - Surface)                           │
│    ═══════════════════════════════════                          │
│    • ask("question") → answer                                   │
│    • research("topic") → insights                               │
│    • agent("persona") → intelligent assistant                   │
│                                                                  │
│    ─────────────────────────────────────────────────────────    │
│                                                                  │
│    INTELLIGENCE LAYER (Dimension 2 - Depth)                     │
│    ═══════════════════════════════════════                       │
│    • Auto-configuration (zero-config)                           │
│    • Smart defaults (model selection)                           │
│    • Memory management (conversation context)                   │
│    • RAG indexing (document understanding)                      │
│                                                                  │
│    ─────────────────────────────────────────────────────────    │
│                                                                  │
│    FOUNDATION LAYER (Dimension 3 - Infrastructure)             │
│    ═══════════════════════════════════════════════              │
│    • Multi-provider LLM support                                 │
│    • Skill system (extensible capabilities)                     │
│    • Blueprint patterns (complex workflows)                     │
│    • Memory stores (conversation, vector, hybrid)              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Principles

### 1. Zero Friction

```python
# Other frameworks: 10+ lines
from langchain.llms import OpenAI
from langchain.chains import LLMChain
# ... setup code ...
result = chain.run("What is AI?")

# PyAgent: 1 line
from pyagent import ask
answer = ask("What is AI?")
```

### 2. Sensible Defaults

- Auto-detects API keys from environment
- Uses optimal model for each task type
- Manages memory automatically
- Handles errors gracefully

### 3. Progressive Complexity

```python
# Level 1: One-liner
answer = ask("What is AI?")

# Level 2: Customized
answer = ask("What is AI?", detailed=True, model="gpt-4")

# Level 3: Agent
agent = Agent(name="researcher", skills=[web_search])

# Level 4: Workflow
workflow = Workflow().chain(analyze).then(summarize)
```

## Module Structure

```
pyagent/
├── easy/           # One-liner functions
│   ├── ask.py
│   ├── research.py
│   ├── summarize.py
│   └── ...
├── core/           # Core components
│   ├── agent.py
│   ├── llm.py
│   └── memory.py
├── blueprint/      # Workflow patterns
│   ├── workflow.py
│   ├── pipeline.py
│   └── patterns.py
├── skills/         # Extensible skills
│   ├── skill.py
│   └── registry.py
├── integrations/   # External integrations
│   ├── vector_db.py
│   └── langchain_adapter.py
└── ...
```

## Component Overview

### Easy Module

User-facing one-liner functions:

| Function | Purpose |
|----------|---------|
| `ask()` | Q&A |
| `research()` | Deep research |
| `summarize()` | Text summarization |
| `extract()` | Data extraction |
| `translate()` | Translation |
| `generate()` | Content generation |
| `chat()` | Conversational AI |
| `agent()` | Agent factory |

### Core Module

Foundation components:

| Component | Purpose |
|-----------|---------|
| `Agent` | Agent abstraction |
| `LLMProvider` | LLM interface |
| `Memory` | Context management |
| `Skills` | Agent capabilities |

### Blueprint Module

Workflow orchestration:

| Pattern | Purpose |
|---------|---------|
| `Workflow` | Define workflows |
| `Pipeline` | Data pipelines |
| `Chain` | Sequential processing |
| `Supervisor` | Agent supervision |

### Integration Layer

External services:

| Integration | Services |
|-------------|----------|
| `vectordb` | ChromaDB, Pinecone, Weaviate, Qdrant |
| `sessions` | SQLite, Redis, Memory |
| `openapi` | OpenAPI tool generation |
| `a2a` | Agent-to-Agent protocol |

## Data Flow

```
User Request
     │
     ▼
┌─────────────┐
│ Easy Layer  │  ask(), research(), etc.
└─────────────┘
     │
     ▼
┌─────────────┐
│ Agent Layer │  Agent, Skills, Memory
└─────────────┘
     │
     ▼
┌─────────────┐
│ LLM Layer   │  OpenAI, Anthropic, Azure
└─────────────┘
     │
     ▼
Response
```

## Memory Architecture

```
┌─────────────────────────────────────┐
│           Memory Manager            │
├─────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐          │
│  │ Short   │  │ Long    │          │
│  │ Term    │  │ Term    │          │
│  │ (Chat)  │  │ (Vector)│          │
│  └─────────┘  └─────────┘          │
└─────────────────────────────────────┘
```

## Security Architecture

```
┌─────────────────────────────────────┐
│           Guardrails                │
├─────────────────────────────────────┤
│  Input Validators                   │
│  • PII Detection                    │
│  • Harmful Content                  │
│  • Injection Prevention             │
├─────────────────────────────────────┤
│  Output Filters                     │
│  • Redaction                        │
│  • Sanitization                     │
│  • Length Limits                    │
└─────────────────────────────────────┘
```

## Extensibility

### Custom Skills

```python
from pyagent.skills import skill

@skill
def my_tool(query: str) -> str:
    """Custom tool."""
    return process(query)
```

### Custom Providers

```python
from pyagent.core import LLMProvider

class MyProvider(LLMProvider):
    def complete(self, messages):
        # Custom implementation
        pass
```

## See Also

- [Design-Philosophy](Design-Philosophy) - Design principles
- [Three-Dimensions](Three-Dimensions) - 3D architecture details
- [Quick-Start](Quick-Start) - Getting started
