# Design Philosophy

PYAI is built on four fundamental principles that guide every design decision.

---

## 1. Intelligence as Infrastructure

> *AI shouldn't be bolted on — it should be woven in.*

PYAI treats intelligence as a first-class architectural component, not an afterthought.

### What This Means

| Traditional Approach | PYAI Approach |
|---------------------|---------------|
| AI as a feature | AI as infrastructure |
| Call AI when needed | AI embedded throughout |
| Separate AI service | Unified intelligence layer |
| Manual orchestration | Automatic coordination |

### In Practice

```python
# Not this (AI as add-on)
response = openai.chat.completions.create(...)
parsed = json.loads(response.choices[0].message.content)
validated = validate(parsed)
# ... more manual wiring

# This (AI as infrastructure)
from pyai import extract
data = extract(text, ["name", "email", "phone"])
# Intelligence is the infrastructure
```

---

## 2. Progressive Complexity

> *Start with one line. Scale to software factories. Same API, same patterns, infinite scale.*

PYAI grows with your needs. Simple tasks stay simple. Complex tasks become possible.

### The Progression

```python
# 🌱 Level 1: One Line
answer = ask("What is Python?")

# 🌿 Level 2: Configured Agent
agent = Agent(
    name="Explainer",
    instructions="Explain concepts clearly",
    tools=[search_tool]
)
result = Runner.run_sync(agent, "Explain Python")

# 🌳 Level 3: Multi-Agent Workflow
workflow = (Workflow("ExplainPipeline")
    .add_step(Step("research", researcher))
    .add_step(Step("explain", explainer))
    .add_step(Step("review", reviewer))
    .build())
result = await workflow.run("Explain Python comprehensively")

# 🏔️ Level 4: Full Orchestration
kernel = (KernelBuilder()
    .add_llm(azure_client, name="primary")
    .add_memory(redis_store)
    .add_plugins([ResearchPlugin(), WritingPlugin()])
    .build())
```

### Same Patterns Throughout

Whether you use one-liners or full orchestration, the patterns remain consistent:
- Input → Processing → Output
- Tools extend capabilities
- Memory persists context
- Evaluation ensures quality

---

## 3. Zero Friction

> *No boilerplate. No ceremony. If it takes more than 3 lines for a common task, we failed.*

Every API is designed for minimal cognitive load.

### The Three-Line Rule

Common tasks should never require more than 3 lines:

```python
# RAG: 2 lines
docs = rag.index("./documents")
answer = docs.ask("What's the conclusion?")

# Custom agent: 3 lines
agent = Agent(name="Helper", instructions="Be helpful")
result = Runner.run_sync(agent, "Hello")
print(result.final_output)

# Multi-agent: 3 lines
workflow = Workflow("Pipeline").add_step(...).build()
result = await workflow.run("Task")
print(result)
```

### Compare to Others

| Task | LangChain | CrewAI | PYAI |
|------|-----------|--------|------|
| Simple question | 10+ lines | N/A | 1 line |
| RAG setup | 15+ lines | N/A | 2 lines |
| Agent with tools | 20+ lines | 25+ lines | 5 lines |
| Multi-agent | 40+ lines | 50+ lines | 10 lines |

---

## 4. Production Ready

> *Type hints. Error handling. Retry logic. Rate limiting. Caching. Built in, not bolted on.*

PYAI is designed for production from day one.

### Built-In Features

| Feature | Description |
|---------|-------------|
| **Type Hints** | Full typing throughout for IDE support |
| **Error Handling** | Structured exceptions with context |
| **Retry Logic** | Automatic retries with backoff |
| **Rate Limiting** | Built-in rate limit handling |
| **Caching** | Response caching for efficiency |
| **Tracing** | Full observability |
| **Azure AD** | Enterprise authentication |
| **Sessions** | Persistent conversation state |
| **Evaluation** | Testing framework included |

### Example: Production Agent

```python
from pyai import Agent, Runner
from pyai.runner import RunConfig
from pyai.sessions import SessionManager, SQLiteSessionStore

# Production-grade configuration
agent = Agent(
    name="ProductionBot",
    instructions="You are a helpful assistant.",
    tools=[verified_tool],
)

config = RunConfig(
    max_turns=10,
    timeout=60,
    trace_enabled=True,
)

sessions = SessionManager(store=SQLiteSessionStore("prod.db"))

# Run with full production features
session = sessions.get_or_create("user-123")
result = Runner.run_sync(agent, user_input, config=config, session=session)
sessions.save(session)
```

---

## Guiding Questions

When designing PYAI features, we ask:

1. **Can a new user do this in under 3 lines?**
2. **Does this work out of the box with sensible defaults?**
3. **Can power users still access full control?**
4. **Is this production-ready without additional setup?**
5. **Does this follow the patterns users already know?**

---

## The Result

PYAI achieves the rare combination of:

```
┌─────────────────────────────────────────┐
│           SIMPLICITY                     │
│   ┌─────────────────────────────────┐   │
│   │         POWER                    │   │
│   │   ┌─────────────────────────┐   │   │
│   │   │     PRODUCTION          │   │   │
│   │   │   ┌─────────────────┐   │   │   │
│   │   │   │  ENTERPRISE     │   │   │   │
│   │   │   └─────────────────┘   │   │   │
│   │   └─────────────────────────┘   │   │
│   └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

All four, no tradeoffs.

---

## Next Steps

- [[Quick Start]] - See the philosophy in action
- [[Three Dimensions]] - The architecture
- [[Agent]] - Build your first agent
