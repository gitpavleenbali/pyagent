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
  <a href="https://pypi.org/project/pyagent/"><img src="https://img.shields.io/badge/pypi-v0.4.0-blue" alt="PyPI"/></a>
  <a href="https://python.org/"><img src="https://img.shields.io/badge/python-3.10+-green" alt="Python"/></a>
  <a href="#"><img src="https://img.shields.io/badge/tests-671%20passing-brightgreen" alt="Tests"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-yellow" alt="License"/></a>
</p>

<p align="center">
  <a href="#-the-three-dimensions">Three Dimensions</a> •
  <a href="#-software-factories">Software Factories</a> •
  <a href="#-the-ecosystem">Ecosystem</a> •
  <a href="#-complete-feature-guide">Features</a> •
  <a href="#-get-started">Get Started</a>
</p>

---

## 🎯 What is PYAI?

**PYAI is not just another AI library. It's an Intelligence Engine.**

While other frameworks help you *call* AI models, PYAI embeds intelligence *into* your software architecture. It's the foundation for building **Software Factories** — systems that don't just use AI, but think, adapt, and create.

> *"The best code is the code you never had to write. The best software is the software that writes itself."*

Built on **PyAgent**, our core SDK, PYAI provides **25+ modules** with **150+ classes** covering every AI use case — from one-liner operations to enterprise-grade multi-agent orchestration.

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
from pyagent import Agent
from pyagent.blueprint import Workflow, Step

# Specialized agents
researcher = Agent(name="Researcher", instructions="Find information.")
analyst = Agent(name="Analyst", instructions="Analyze data deeply.")
writer = Agent(name="Writer", instructions="Write compelling content.")

# Orchestrated workflow
workflow = (Workflow("ResearchPipeline")
    .add_step(Step("research", researcher))
    .add_step(Step("analyze", analyst))
    .add_step(Step("write", writer))
    .build())
```

### Dimension 3️⃣ — Creation
Self-generating systems. **The Software Factory.**

```python
from pyagent import code, generate

# The factory builds software
api_code = code.write("REST API for user management with JWT auth")
# → Generates models, routes, middleware, tests

# Intelligent code operations
review = code.review(existing_code)
improved = code.refactor(old_code, goal="async architecture")
fixed = code.debug("TypeError: cannot unpack non-iterable NoneType")
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
│     Azure OpenAI  |  OpenAI  |  Anthropic  |  Ollama    │
└──────────────────────────────────────────────────────────┘
```

---

## 📦 The Ecosystem

### 🤖 PyAgent — *Available Now*
**The Core Intelligence SDK**

Build AI-powered applications with elegant simplicity. 25+ modules, 150+ classes, 671 tests.

```python
from pyagent import ask, Agent, Runner, rag

# One-liner AI
answer = ask("What is the meaning of life?")

# Powerful agents
agent = Agent(name="Coder", instructions="Expert Python developer")
result = Runner.run_sync(agent, "Optimize this for O(log n)")

# RAG in 2 lines
knowledge = rag.index("./research/")
insight = knowledge.ask("What are the key findings?")
```

[📚 Documentation](./docs/QUICKSTART.md) | [🚀 API Reference](./docs/API_REFERENCE.md) | [💡 Examples](./examples/)

---

### 🔮 Coming Soon — The PYAI Product Suite

| Library | Purpose | Dimension |
|---------|---------|-----------|
| **PyFlow** | Visual AI workflow orchestration | Orchestration |
| **PyVision** | Computer vision made simple | Cognition |
| **PyVoice** | Speech & audio intelligence | Cognition |
| **PyFactory** | Software generation engine | Creation |
| **PyMind** | Autonomous reasoning systems | Creation |

---

## 🧬 Design Philosophy

### 1. **Intelligence as Infrastructure**
AI shouldn't be bolted on — it should be woven in. PYAI treats intelligence as a first-class architectural component.

### 2. **Progressive Complexity**
Start with one line. Scale to software factories. Same API, same patterns, infinite scale.

```python
# Level 1: One line
answer = ask("Translate to French: Hello")

# Level 2: Agent with tools
translator = Agent(name="Translator", instructions="Expert linguist", tools=[...])
result = Runner.run_sync(translator, "Translate to all languages: Hello")

# Level 3: Multi-agent orchestration
workflow = Workflow("TranslationService").add_step(...).add_step(...).build()
```

### 3. **Zero Friction**
No boilerplate. No ceremony. If it takes more than 3 lines for a common task, we failed.

### 4. **Production Ready**
Type hints. Error handling. Retry logic. Rate limiting. Caching. Azure AD auth. Built in, not bolted on.

---

## ✨ Why PYAI?

| Other Frameworks | PYAI |
|-----------------|------|
| 50 lines for RAG | 2 lines |
| Agent = configuration hell | `Agent(name="Bot", instructions="...")` |
| Memory = complex setup | Built-in, automatic |
| Workflows = YAML nightmares | Python functions |
| "Hello World" = 30 minutes | "Hello World" = 30 seconds |

### Framework Comparison

| Framework | Lines for RAG | Lines for Agent | Lines for Research | Multi-Agent |
|-----------|--------------|-----------------|-------------------|-------------|
| LangChain | 15+ | 20+ | 25+ | 40+ |
| LlamaIndex | 10+ | 15+ | 20+ | 30+ |
| CrewAI | 30+ | 25+ | 35+ | 50+ |
| **PYAI** | **2** | **5** | **1** | **10** |

### Feature Comparison

| Feature | PYAI | OpenAI Agents | Google ADK | Semantic Kernel | LangChain |
|---------|------|---------------|------------|-----------------|-----------|
| One-liner APIs | ✅ | ❌ | ❌ | ❌ | ❌ |
| Multi-provider LLM | ✅ | ❌ | ✅ | ✅ | ✅ |
| Azure AD Auth | ✅ | ❌ | ❌ | ✅ | ❌ |
| Session Management | ✅ | ✅ | ✅ | ❌ | ✅ |
| Evaluation Framework | ✅ | ❌ | ✅ | ❌ | ❌ |
| Voice Streaming | ✅ | ✅ | ❌ | ❌ | ❌ |
| MCP Protocol | ✅ | ❌ | ❌ | ❌ | ❌ |
| A2A Protocol | ✅ | ❌ | ✅ | ❌ | ❌ |
| Guardrails | ✅ | ✅ | ❌ | ❌ | ✅ |
| Workflow Patterns | ✅ | ❌ | ❌ | ✅ | ✅ |
| Plugin System | ✅ | ❌ | ❌ | ✅ | ❌ |
| YAML Config | ✅ | ❌ | ✅ | ❌ | ❌ |

---

## 🚀 Get Started

### Installation

```bash
# Basic install
pip install pyagent

# With providers
pip install pyagent[openai]      # OpenAI models
pip install pyagent[anthropic]   # Anthropic Claude
pip install pyagent[azure]       # Azure OpenAI + Azure AD

# With integrations
pip install pyagent[langchain]   # LangChain integration
pip install pyagent[vector]      # Vector databases

# Full installation
pip install pyagent[all]         # Everything
```

### Configuration

```bash
# OpenAI
export OPENAI_API_KEY=sk-your-key

# Azure OpenAI (with Azure AD - no API key needed!)
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

### Hello, Intelligence

```python
from pyagent import ask

# Your first intelligent operation
answer = ask("What makes PYAI revolutionary?")
print(answer)
```

---

# 📚 Complete Feature Guide

## 🎯 One-Liner APIs (easy/ module)

### ask() - Universal Question Answering

```python
from pyagent import ask

answer = ask("What is Python?")
answer = ask("Explain quantum computing", detailed=True)
answer = ask("List 5 tips", format="bullet")
```

### research() - Deep Topic Research

```python
from pyagent import research

result = research("AI trends in enterprise")
print(result.summary)          # Executive summary
print(result.key_points)       # Main takeaways
print(result.insights)         # Analysis
print(result.sources)          # References
```

### summarize() - Text/File/URL Summarization

```python
from pyagent import summarize

summary = summarize("Long article text here...")
summary = summarize("./report.pdf")
summary = summarize("https://example.com/article")
summary = summarize(text, length="short", bullet_points=True)
```

### rag - 2-Line RAG System

```python
from pyagent import rag

docs = rag.index("./documents")
answer = docs.ask("What is the main conclusion?")

# Or one-liner
answer = rag.ask("./research_paper.pdf", "What methodology was used?")
```

### generate() - Content Generation

```python
from pyagent import generate

code = generate("fibonacci function", type="code")
email = generate("welcome email", type="email")
article = generate("blog about AI", type="article")
```

### translate() - Language Translation

```python
from pyagent import translate

spanish = translate("Hello, how are you?", to="spanish")
japanese = translate("Welcome", to="japanese", formal=True)
```

### extract() - Structured Data Extraction

```python
from pyagent import extract

text = "John is 30 years old and lives in New York"
data = extract(text, ["name", "age", "city"])
# {"name": "John", "age": 30, "city": "New York"}
```

### fetch - Real-Time Data

```python
from pyagent import fetch

weather = fetch.weather("New York")
news = fetch.news("artificial intelligence")
stock = fetch.stock("AAPL")
crypto = fetch.crypto("BTC")
```

### analyze - Data Analysis

```python
from pyagent import analyze

insights = analyze.data(sales_data)
sentiment = analyze.sentiment("I love this product!")
# {"sentiment": "positive", "confidence": 0.95}
```

### code - Code Operations

```python
from pyagent import code

python_code = code.write("REST API for todo app")
review = code.review(my_code)
solution = code.debug("TypeError: cannot unpack...")
explanation = code.explain(complex_function)
improved = code.refactor(old_code, goal="readability")
```

### chat() - Interactive Sessions

```python
from pyagent import chat

session = chat(persona="teacher")
session("Explain machine learning")
session("What about deep learning?")  # Continues conversation
```

---

## 🤖 Agent Framework (core/ module)

### Agent Class

```python
from pyagent import Agent
from pyagent.core import LLMConfig, AzureOpenAIProvider

# Simple agent
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant."
)

# Configured agent with Azure AD
agent = Agent(
    name="Coder",
    instructions="You are an expert Python developer.",
    llm=AzureOpenAIProvider(LLMConfig(
        api_base="https://your-resource.openai.azure.com/",
        model="gpt-4o-mini"
    )),
    tools=[search_tool, code_tool],
    memory_type="conversation"
)
```

### Runner Pattern

```python
from pyagent import Agent, Runner
from pyagent.runner import RunConfig, StreamingRunner

agent = Agent(name="Bot", instructions="Be helpful")

# Synchronous
result = Runner.run_sync(agent, "Hello")
print(result.final_output)

# Asynchronous
result = await Runner.run_async(agent, "Hello")

# With configuration
config = RunConfig(max_turns=10, timeout=60)
result = Runner.run_sync(agent, "Complex task", config=config)

# Streaming
async for event in StreamingRunner.stream(agent, "Hello"):
    print(event.data, end="", flush=True)
```

### Memory Systems

```python
from pyagent.core import ConversationMemory, VectorMemory

# Conversation memory (sliding window)
memory = ConversationMemory(max_messages=50)

# Vector memory (semantic search)
memory = VectorMemory(provider="chromadb")

agent = Agent(name="Bot", instructions="...", memory=memory)
```

---

## 🔗 Multi-Agent Systems (blueprint/ module)

### Workflows

```python
from pyagent import Agent
from pyagent.blueprint import Workflow, Step

researcher = Agent(name="Researcher", instructions="Find information.")
writer = Agent(name="Writer", instructions="Write engaging content.")
editor = Agent(name="Editor", instructions="Review and improve.")

workflow = (Workflow("ContentPipeline")
    .add_step(Step("research", researcher))
    .add_step(Step("write", writer))
    .add_step(Step("edit", editor))
    .build())

result = await workflow.run("Create article about AI")
```

### Agent Handoffs

```python
from pyagent import Agent, Runner

spanish_agent = Agent(name="SpanishAgent", instructions="You only speak Spanish.")
english_agent = Agent(name="EnglishAgent", instructions="You only speak English.")

triage_agent = Agent(
    name="TriageAgent",
    instructions="Route to the appropriate language agent.",
    handoffs=[spanish_agent, english_agent]
)

result = Runner.run_sync(triage_agent, "Hola, como estas?")
```

### Orchestration Patterns

```python
from pyagent.blueprint import ChainPattern, RouterPattern, MapReducePattern, SupervisorPattern

# Chain: Sequential agent processing
chain = ChainPattern()
chain.add("draft", writer)
chain.add("edit", editor)
result = await chain.run("Write about AI")

# Router: Route to specialist agents
router = RouterPattern()
router.add_route("code", coder, keywords=["code", "python", "function"])
router.add_route("math", calculator, keywords=["calculate", "compute"])
result = await router.run("Write a fibonacci function")

# MapReduce: Parallel processing with aggregation
mapreduce = MapReducePattern(
    map_agents=[researcher1, researcher2, researcher3],
    reduce_agent=synthesizer
)
result = await mapreduce.run("Research AI from different angles")

# Supervisor: Hierarchical agent management
supervisor = SupervisorPattern(
    manager=manager_agent,
    workers=[worker1, worker2, worker3]
)
result = await supervisor.run("Complex project requiring coordination")

# Consensus: Voting-based decision making
from pyagent.orchestrator import AgentPatterns

decision = AgentPatterns.consensus(
    task="Should we approve this feature?",
    agents=[security_expert, ux_expert, perf_expert],
    threshold=0.66
)

# Debate: Adversarial reasoning
verdict = AgentPatterns.debate(
    topic="AI open-source vs proprietary",
    pro_agent=advocate,
    con_agent=skeptic,
    judge=arbiter
)
```

---

## 🛠️ Skills & Tools (skills/ module)

### Creating Tools

```python
from pyagent.skills import tool, action

@tool(description="Search the web for information")
async def web_search(query: str, limit: int = 10) -> list:
    results = await search_api(query)
    return results[:limit]

@tool(description="Send an email")
async def send_email(to: str, subject: str, body: str) -> str:
    return f"Email sent to {to}"

agent = Agent(
    name="Assistant",
    instructions="Help users with tasks.",
    tools=[web_search, send_email]
)
```

### Built-in Skills

```python
from pyagent.skills.builtin import SearchSkill, CodeSkill, FileSkill, WebSkill, MathSkill

agent = Agent(
    name="PowerUser",
    instructions="You can search, code, and analyze.",
    tools=[SearchSkill(), CodeSkill(), FileSkill(), MathSkill()]
)
```

### OpenAPI Tools (Auto-generate from specs)

```python
from pyagent.openapi import create_tools_from_openapi

tools = create_tools_from_openapi("petstore.yaml")
agent = Agent(name="PetStoreBot", instructions="Help manage pets.", tools=tools)
```

---

## 🔌 Plugin System (plugins/ module)

```python
from pyagent.plugins import plugin, function, Plugin

@plugin(name="weather", description="Weather information")
class WeatherPlugin(Plugin):
    
    @function(description="Get current weather")
    def get_weather(self, city: str) -> str:
        return f"Weather in {city}: Sunny, 72°F"
    
    @function(description="Get forecast")
    def get_forecast(self, city: str, days: int = 5) -> str:
        return f"5-day forecast for {city}..."

agent = Agent(
    name="WeatherBot",
    instructions="Help with weather.",
    plugins=[WeatherPlugin()]
)
```

---

## 🧠 Kernel Registry (kernel/ module)

MS Semantic Kernel-style service management:

```python
from pyagent.kernel import Kernel, KernelBuilder

kernel = (KernelBuilder()
    .add_llm(openai_client, name="gpt4", is_default=True)
    .add_llm(azure_client, name="azure-gpt4")
    .add_memory(redis_memory)
    .add_plugin(WeatherPlugin())
    .build())

result = await kernel.invoke("weather", "get_weather", city="NYC")
kernel.set_default_service("llm", "azure-gpt4")
```

---

## 💾 Session Management (sessions/ module)

```python
from pyagent.sessions import SessionManager, SQLiteSessionStore, RedisSessionStore

# SQLite (local)
manager = SessionManager(store=SQLiteSessionStore("sessions.db"))

# Redis (distributed)
manager = SessionManager(store=RedisSessionStore(host="localhost", port=6379))

# Use sessions
session = manager.get_or_create("user-123")
session.add_user_message("Hello!")
session.add_assistant_message("Hi there!")
manager.save(session)

# Resume later
session = manager.get("user-123")
```

---

## 📊 Evaluation Framework (evaluation/ module)

```python
from pyagent.evaluation import Evaluator, EvalSet, TestCase, ExactMatch, LLMJudge

eval_set = EvalSet([
    TestCase(input="What is 2+2?", expected="4", criteria=ExactMatch()),
    TestCase(input="Write a haiku", criteria=LLMJudge(prompt="Is this valid?")),
])

evaluator = Evaluator(agent)
results = await evaluator.run(eval_set)
print(f"Pass Rate: {results.metrics.pass_rate}%")
```

---

## 🎤 Voice & Audio (voice/ module)

```python
from pyagent.voice import VoiceSession

async with VoiceSession(agent) as session:
    session.send_audio(audio_chunk)
    text = await session.get_transcription()
    response_audio = await session.get_audio_response()
```

---

## 🖼️ Multimodal (multimodal/ module)

```python
from pyagent.multimodal import Image

img = Image.from_file("photo.png")
result = Runner.run_sync(agent, "Describe this image", images=[img])

img = Image.from_url("https://example.com/photo.jpg")
```

---

## 🔄 Agent-to-Agent Protocol (a2a/ module)

```python
from pyagent.a2a import A2AServer, A2AClient, AgentCard

# Server: Expose agent
server = A2AServer(agent)
server.set_card(AgentCard(name="WeatherAgent", capabilities=["weather"]))
await server.start(port=8080)

# Client: Connect to remote agents
client = A2AClient()
remote = await client.connect("http://weather-agent:8080")
result = await remote.run("Weather in NYC?")
```

---

## 🗄️ Vector Database Connectors (vectordb/ module)

```python
from pyagent.vectordb import ChromaStore, PineconeStore, QdrantStore, Document

store = ChromaStore(collection="my_docs")  # Or PineconeStore, QdrantStore

docs = [Document(content="AI is transforming...", metadata={"topic": "ai"})]
await store.add(docs)

results = await store.search("What is AI?", limit=5)
```

---

## 🔐 Azure AD Authentication

Enterprise-grade authentication without API keys:

```python
import os
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource.openai.azure.com/"
os.environ["AZURE_OPENAI_DEPLOYMENT"] = "gpt-4o-mini"

from pyagent import ask
# Uses az login / VS Code / Managed Identity automatically
answer = ask("Hello!")
```

---

## 🛡️ Safety & Guardrails

```python
from pyagent.easy import guardrails

safe_ask = guardrails.wrap(ask, block_pii=True, block_harmful=True, max_tokens=1000)
result = safe_ask("Tell me about...")

if result.violations:
    print(f"Blocked: {result.violations}")
```

---

## 📍 Tracing & Observability

```python
from pyagent.easy import trace

trace.enable()
ask("What is AI?")
trace.show()
trace.export("trace.json")
```

---

## 🎮 Industry Use Cases (usecases/ module)

Pre-built agents for common business scenarios:

```python
from pyagent.usecases import get_agent
from pyagent.usecases.industry import telecom, healthcare, finance

# Customer Service
support = get_agent("support_agent")
tech_support = get_agent("technical_agent")
billing = get_agent("billing_agent")

# Development
code_reviewer = get_agent("code_reviewer")
debugger = get_agent("debugger")
doc_writer = get_agent("documentation_writer")

# Industry-Specific
plan_advisor = telecom.plan_advisor(carrier_name="MobileNet")
scheduler = healthcare.appointment_scheduler(facility="City Hospital")
banker = finance.banking_assistant(bank_name="First Bank")

# Gaming
npc = get_agent("npc_agent")
game_master = get_agent("game_master")
```

---

## 🧮 Token Counting & Cost (tokens/ module)

```python
from pyagent.tokens import count_tokens, calculate_cost

tokens = count_tokens("Hello, how are you?", model="gpt-4o-mini")
cost = calculate_cost(input_tokens=1000, output_tokens=500, model="gpt-4o-mini")
print(f"Tokens: {tokens}, Cost: ${cost:.4f}")
```

---

## 📁 Architecture

```
pyagent/
├── easy/           # 🚀 One-liner APIs (ask, research, summarize...)
├── core/           # 🧠 Agent, LLM providers, Memory
├── runner/         # ⚡ Execution engine (Runner, StreamingRunner)
├── blueprint/      # 🔗 Workflows, orchestration, patterns
├── skills/         # 🛠️ Tools and capabilities
├── kernel/         # 🔌 Service registry (Semantic Kernel style)
├── sessions/       # 💾 SQLite/Redis session persistence
├── evaluation/     # 📊 Testing and evaluation
├── voice/          # 🎤 Voice streaming
├── multimodal/     # 🖼️ Image, audio, video support
├── a2a/            # 🔄 Agent-to-Agent protocol
├── vectordb/       # 🗄️ Vector database connectors
├── openapi/        # 📜 OpenAPI tool generation
├── plugins/        # 🔌 Plugin architecture
├── config/         # ⚙️ YAML/JSON configuration
├── tokens/         # 🧮 Token counting & cost
├── models/         # 🤖 Multi-provider models
├── instructions/   # 📝 Persona and guidelines
├── code_executor/  # 💻 Safe code execution
├── integrations/   # 🔗 LangChain, SK adapters
├── usecases/       # 🎯 Pre-built industry agents
├── devui/          # 🖥️ Development UI
├── cli/            # ⌨️ Command line interface
└── errors/         # ❌ Error hierarchy
```

---

## 🔌 Integrations

Connect PYAI to your existing ecosystem:

```python
from pyagent.integrations import langchain, semantic_kernel, vector_db

# Import LangChain tools
search = langchain.import_tool("serpapi")

# Create Semantic Kernel
kernel = semantic_kernel.create_kernel(provider="azure", deployment="gpt-4o")

# Connect to vector stores
store = vector_db.connect("azure_ai_search", endpoint="...", index="docs")
```

| Integration | Features |
|-------------|----------|
| **LangChain** | Import tools, chains, retrievers; Export agents |
| **Semantic Kernel** | Create kernels, import plugins, execute plans |
| **Azure AI Search** | Enterprise search with hybrid retrieval |
| **Pinecone** | Scalable cloud vector database |
| **ChromaDB** | Open-source embedding database |
| **FAISS** | Fast in-memory similarity search |
| **Qdrant** | High-performance vector search |

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
  <strong>🧠 PYAI</strong><br/>
  <em>Intelligence, Embedded.</em>
</p>

<p align="center">
  <strong>25+ Modules • 150+ Classes • 671 Tests • Infinite Possibilities</strong>
</p>

<p align="center">
  <sub>Built with 🧠 by the PYAI team</sub>
</p>
