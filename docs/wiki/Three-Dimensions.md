# The Three Dimensions

PYAI operates across **three dimensions of intelligence**, each building upon the last.

```
                    ┌─────────────────────────────────┐
                    │     DIMENSION 3: CREATION       │
                    │   Software Factory Intelligence  │
                    │ ┌─────────────────────────────┐ │
                    │ │ • Self-generating systems   │ │
                    │ │ • Code synthesis engines    │ │
                    │ │ • Autonomous development    │ │
                    │ └─────────────────────────────┘ │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │    DIMENSION 2: ORCHESTRATION   │
                    │     Multi-Agent Intelligence     │
                    │ ┌─────────────────────────────┐ │
                    │ │ • Agent coordination        │ │
                    │ │ • Workflow automation       │ │
                    │ │ • Knowledge synthesis       │ │
                    │ └─────────────────────────────┘ │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │     DIMENSION 1: COGNITION      │
                    │      Core AI Operations          │
                    │ ┌─────────────────────────────┐ │
                    │ │ • ask() • research()        │ │
                    │ │ • summarize() • analyze()   │ │
                    │ │ • extract() • generate()    │ │
                    │ └─────────────────────────────┘ │
                    └─────────────────────────────────┘
```

---

## Dimension 1️⃣ — Cognition

The foundation. Single-purpose AI operations that **just work**.

### Key Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `ask()` | Question answering | `ask("What is Python?")` |
| `research()` | Deep research | `research("AI trends 2024")` |
| `summarize()` | Summarization | `summarize("./report.pdf")` |
| `extract()` | Data extraction | `extract(text, ["name", "email"])` |
| `generate()` | Content creation | `generate("blog post", type="article")` |
| `translate()` | Translation | `translate("Hello", to="spanish")` |
| `code` | Code operations | `code.write("REST API")` |
| `analyze` | Analysis | `analyze.sentiment("I love it!")` |

### Example

```python
from pyai import ask, summarize, extract

# Instant intelligence
answer = ask("Explain quantum entanglement")
summary = summarize(long_document)
entities = extract(text, fields=["names", "dates", "amounts"])
```

---

## Dimension 2️⃣ — Orchestration

Coordinated intelligence. Multiple agents working in harmony.

### Key Patterns

| Pattern | Purpose | Use Case |
|---------|---------|----------|
| **Chain** | Sequential processing | Draft → Edit → Publish |
| **Router** | Route to specialists | Code questions → Coder agent |
| **MapReduce** | Parallel with aggregation | Multi-angle research |
| **Supervisor** | Hierarchical management | Manager + workers |
| **Consensus** | Voting-based decisions | Multi-expert approval |
| **Debate** | Adversarial reasoning | Pro/con analysis |

### Example

```python
from pyai import Agent
from pyai.blueprint import Workflow, Step

researcher = Agent(name="Researcher", instructions="Find information.")
analyst = Agent(name="Analyst", instructions="Analyze data.")
writer = Agent(name="Writer", instructions="Write content.")

workflow = (Workflow("ResearchPipeline")
    .add_step(Step("research", researcher))
    .add_step(Step("analyze", analyst))
    .add_step(Step("write", writer))
    .build())

result = await workflow.run("Create report on AI trends")
```

---

## Dimension 3️⃣ — Creation

Self-generating systems. **The Software Factory.**

### Vision

Software Factories don't just *use* AI — they *are* AI. They generate code, tests, documentation, and entire systems from high-level descriptions.

### Key Capabilities

| Feature | Description |
|---------|-------------|
| **Code Generation** | Generate complete modules from descriptions |
| **Code Review** | Automated quality analysis |
| **Refactoring** | Transform architecture intelligently |
| **Debugging** | Fix errors with explanations |
| **Test Generation** | Create comprehensive test suites |
| **Documentation** | Auto-generate docs from code |

### Example

```python
from pyai import code

# Generate code
api_code = code.write("REST API for user management with JWT auth")

# Review existing code
review = code.review(existing_code)
print(review.issues, review.suggestions, review.score)

# Debug errors
fix = code.debug("TypeError: cannot unpack non-iterable NoneType")

# Refactor architecture
improved = code.refactor(old_code, goal="async architecture")
```

---

## The Intelligence Stack

```
┌──────────────────────────────────────────────────────────┐
│                    YOUR APPLICATION                       │
├──────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │
│  │ pyai │  │ PyFlow  │  │PyVision │  │ PyVoice │     │
│  │ Agents  │  │Workflow │  │ Vision  │  │  Audio  │     │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘     │
│       │            │            │            │           │
│  ┌────▼────────────▼────────────▼────────────▼────┐     │
│  │              PYAI INTELLIGENCE ENGINE           │     │
│  │  • Unified Memory  • Context Management         │     │
│  │  • Model Routing   • Intelligent Caching        │     │
│  └────────────────────────────────────────────────┘     │
├──────────────────────────────────────────────────────────┤
│     Azure OpenAI  |  OpenAI  |  Anthropic  |  Ollama    │
└──────────────────────────────────────────────────────────┘
```

---

## Progressive Complexity

Start simple, scale infinitely:

```python
# Level 1: One line
answer = ask("Translate to French: Hello")

# Level 2: Agent with tools
translator = Agent(name="Translator", instructions="...", tools=[...])
result = Runner.run_sync(translator, "Translate to all languages")

# Level 3: Multi-agent orchestration
workflow = Workflow("TranslationService").add_step(...).build()
```

---

## Next Steps

- [[Software Factories]] - The vision in detail
- [[Design Philosophy]] - Our guiding principles
- [[Agent Framework]] - Build agents
