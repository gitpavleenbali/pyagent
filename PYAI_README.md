<p align="center">
  <img src="https://img.shields.io/badge/PYAI-Intelligence%20Engine-blueviolet?style=for-the-badge&logo=python&logoColor=white" alt="PYAI"/>
</p>

<h1 align="center">🧠 PYAI</h1>
<h3 align="center">Three-Dimensional Intelligence Engine</h3>

<p align="center">
  <strong>The Intelligence Engine for Software Factories</strong><br/>
  <em>Build, Orchestrate, and Scale AI-Native Applications</em>
</p>

<p align="center">
  <a href="#-the-three-dimensions">Three Dimensions</a> •
  <a href="#-software-factories">Software Factories</a> •
  <a href="#-the-ecosystem">Ecosystem</a> •
  <a href="#-get-started">Get Started</a>
</p>

---

## 🎯 What is PYAI?

**PYAI is not just another AI library. It's an Intelligence Engine.**

While other frameworks help you *call* AI models, PYAI embeds intelligence *into* your software architecture. It's the foundation for building **Software Factories** — systems that don't just use AI, but think, adapt, and create.

> *"What SAS did for statistics, what pandas did for data, PYAI does for intelligence."*

---

## 🔺 The Three Dimensions

PYAI operates across **three dimensions of intelligence**, each building upon the last:

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

### Dimension 1️⃣ — Cognition
The foundation. Single-purpose AI operations that **just work**.

```python
from pyagent import ask, summarize, extract

# Instant intelligence
answer = ask("Explain quantum entanglement")
summary = summarize(long_document)
entities = extract(text, fields=["names", "dates", "amounts"])
```

### Dimension 2️⃣ — Orchestration
Coordinated intelligence. Multiple agents working in harmony.

```python
from pyagent import agent, workflow

# Specialized agents
researcher = agent(persona="researcher")
analyst = agent(persona="analyst")
writer = agent(persona="writer")

# Orchestrated workflow
report = workflow([
    researcher >> "Find latest AI trends",
    analyst >> "Analyze market impact",
    writer >> "Write executive summary"
])
```

### Dimension 3️⃣ — Creation
Self-generating systems. **The Software Factory.**

```python
from pyagent import factory

# The factory builds software
factory.create("Build a REST API for user management")
# → Generates models, routes, tests, documentation

factory.extend("Add authentication with JWT")
# → Intelligently extends existing codebase

factory.refactor("Convert to async architecture")
# → Transforms architecture while preserving logic
```

---

## 🏭 Software Factories

A **Software Factory** is a system that generates software, not just code snippets. PYAI provides the intelligence engine to build them.

### Traditional Development vs Software Factory

| Traditional | Software Factory |
|-------------|------------------|
| Write code manually | Describe what you need |
| Debug line by line | Self-healing systems |
| Copy-paste patterns | Intelligent pattern synthesis |
| Manual testing | Auto-generated test suites |
| Static architecture | Evolving, adaptive systems |

### The Intelligence Stack

```
┌──────────────────────────────────────────────────────────┐
│                    YOUR APPLICATION                       │
├──────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │
│  │ PyAgent │  │ PyFlow  │  │PyVision │  │ PyVoice │     │
│  │ Agents  │  │Workflow │  │ Vision  │  │  Audio  │     │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘     │
│       │            │            │            │           │
│  ┌────▼────────────▼────────────▼────────────▼────┐     │
│  │              PYAI INTELLIGENCE ENGINE           │     │
│  │  • Unified Memory  • Context Management         │     │
│  │  • Model Routing   • Intelligent Caching        │     │
│  └────────────────────────────────────────────────┘     │
├──────────────────────────────────────────────────────────┤
│         Azure OpenAI  |  OpenAI  |  Anthropic            │
└──────────────────────────────────────────────────────────┘
```

---

## 📦 The Ecosystem

### 🐼 PyAgent — *Available Now*
**The Pandas of AI Agents**

Build AI-powered applications in 3 lines or less. The most accessible AI agent framework ever created.

```python
from pyagent import ask, agent, rag

# One-liner AI
answer = ask("What is the meaning of life?")

# Expert agents
coder = agent(persona="coder")
solution = coder("Optimize this algorithm for O(log n)")

# RAG in 2 lines
knowledge = rag.index(["research/*.pdf"])
insight = knowledge.ask("What are the key findings?")
```

[📚 PyAgent Documentation](./docs/QUICKSTART.md) | [🚀 API Reference](./docs/API_REFERENCE.md)

---

### 🔮 Coming Soon

| Library | Purpose | Dimension |
|---------|---------|-----------|
| **PyFlow** | Visual AI workflow orchestration | 2 |
| **PyVision** | Computer vision made simple | 1 |
| **PyVoice** | Speech & audio intelligence | 1 |
| **PyFactory** | Software generation engine | 3 |
| **PyMind** | Autonomous reasoning systems | 3 |

---

## 🚀 Get Started

### Installation

```bash
pip install pyagent

# With Azure support
pip install pyagent[azure]
```

### Configuration

```bash
# OpenAI
export OPENAI_API_KEY=sk-your-key

# Azure OpenAI (with Azure AD - recommended)
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

### Hello, Intelligence

```python
from pyagent import ask

# Your first intelligent operation
answer = ask("What makes PYAI revolutionary?")
print(answer)
# → "PYAI is revolutionary because it embeds intelligence into 
#    software architecture across three dimensions, enabling 
#    the creation of self-generating software factories..."
```

---

## 🧬 Design Philosophy

### 1. **Intelligence as Infrastructure**
AI shouldn't be bolted on — it should be woven in. PYAI treats intelligence as a first-class architectural component.

### 2. **Progressive Complexity**
Start with one line. Scale to software factories. Same API, same patterns, infinite scale.

```python
# Level 1: One line
answer = ask("Translate to French: Hello")

# Level 2: Agent
translator = agent(persona="translator", languages=["fr", "de", "es"])
result = translator("Translate to all languages: Hello")

# Level 3: Factory
factory.create("Build a multi-language translation service with API")
```

### 3. **Zero Friction**
No boilerplate. No ceremony. If it takes more than 3 lines for a common task, we failed.

### 4. **Production Ready**
Type hints. Error handling. Retry logic. Rate limiting. Caching. Built in, not bolted on.

---

## 🔥 Why PYAI?

| Other Frameworks | PYAI |
|-----------------|------|
| 50 lines for RAG | 2 lines |
| Agent = configuration hell | `agent(persona="coder")` |
| Memory = complex setup | Built-in, automatic |
| Workflows = YAML nightmares | Python functions |
| "Hello World" = 30 minutes | "Hello World" = 30 seconds |

---

## 🌍 The Vision

We're building the **operating system for intelligent software**.

```
2024: PyAgent launches → Simple AI operations
2025: PyFlow launches  → Orchestrated intelligence  
2026: PyFactory       → Software Factories emerge
2027: PyMind          → Autonomous development
2030: ???             → Software that writes itself
```

**This is not hype. This is the roadmap.**

---

## 👥 Community

- 📖 [Documentation](./docs/)
- 🐛 [Report Issues](https://github.com/gitpavleenbali/PYAI/issues)
- 💡 [Feature Requests](https://github.com/gitpavleenbali/PYAI/discussions)
- 🤝 [Contributing Guide](./docs/CONTRIBUTING.md)

---

## 📜 License

MIT License — Build freely, build boldly.

---

<p align="center">
  <strong>PYAI</strong><br/>
  <em>Intelligence, Embedded.</em>
</p>

<p align="center">
  <sub>Built with 🧠 by the PYAI team</sub>
</p>
