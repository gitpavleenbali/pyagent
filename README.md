<p align="center">
  <img src="https://img.shields.io/badge/🐼-PYAI-orange?style=for-the-badge&labelColor=black" alt="PYAI"/>
</p>

<h1 align="center">🐼🤖 PYAI - Three Dimensional Intelligence Engine</h1>

<p align="center">
  <strong>The Pandas of AI Development</strong><br/>
  <em>What pandas did for data, PYAI does for intelligence.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/pyagent/"><img src="https://img.shields.io/badge/pypi-v0.4.0-blue" alt="PyPI"/></a>
  <a href="https://python.org/"><img src="https://img.shields.io/badge/python-3.10+-green" alt="Python"/></a>
  <a href="#"><img src="https://img.shields.io/badge/tests-671%20passing-brightgreen" alt="Tests"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-yellow" alt="License"/></a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-one-liner-apis">One-Liners</a> •
  <a href="#-agent-framework">Agents</a> •
  <a href="#-multi-agent-systems">Multi-Agent</a> •
  <a href="#-enterprise-features">Enterprise</a>
</p>

---

## 🌟 What is PYAI?

**PYAI** is a revolutionary Python framework that transforms AI development. Built on **PyAgent**, our core SDK, PYAI provides:

- **🚀 One-liner APIs** - Common AI tasks in a single line of code
- **🤖 Full Agent Framework** - Build sophisticated autonomous agents
- **🔗 Multi-Agent Systems** - Orchestrate teams of specialized agents
- **🏢 Enterprise Ready** - Azure AD auth, sessions, evaluation, tracing
- **🎯 25+ Modules** - 150+ classes covering every AI use case

```
┌─────────────────────────────────────────────────────────────────┐
│                          🐼 PYAI                                │
│            Three Dimensional Intelligence Engine                 │
├─────────────────────────────────────────────────────────────────┤
│  DIMENSION 1: SIMPLICITY     │  ask() summarize() research()   │
│  DIMENSION 2: POWER          │  Agent, Runner, Workflow        │
│  DIMENSION 3: ENTERPRISE     │  Azure AD, Sessions, Evaluation │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Why PYAI?

| Framework | Lines for RAG | Lines for Agent | Lines for Research | Multi-Agent |
|-----------|--------------|-----------------|-------------------|-------------|
| LangChain | 15+ | 20+ | 25+ | 40+ |
| LlamaIndex | 10+ | 15+ | 20+ | 30+ |
| CrewAI | 30+ | 25+ | 35+ | 50+ |
| **PYAI** | **2** | **5** | **1** | **10** |

---

## 📦 Installation

```bash
pip install pyagent

# With providers
pip install pyagent[openai]      # OpenAI models
pip install pyagent[anthropic]   # Claude models
pip install pyagent[azure]       # Azure OpenAI + Azure AD
pip install pyagent[all]         # Everything
```

---

## 🚀 Quick Start

### Hello World

```python
from pyagent import ask

answer = ask("What is the capital of France?")
print(answer)  # Paris
```

### Agent with Tools

```python
from pyagent import Agent, Runner
from pyagent.skills import tool

@tool(description="Get weather for a city")
async def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny, 72°F"

agent = Agent(
    name="WeatherBot",
    instructions="Help users with weather information.",
    tools=[get_weather]
)

result = Runner.run_sync(agent, "What's the weather in Tokyo?")
print(result.final_output)
```

---

# 📚 Complete Feature Guide

## 🎯 One-Liner APIs

The **easy/** module provides pandas-like simplicity for AI tasks.

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

result = research("AI trends 2024")
print(result.summary)          # Executive summary
print(result.key_points)       # Main takeaways
print(result.insights)         # Analysis
print(result.sources)          # References
```

### summarize() - Text/File/URL Summarization

```python
from pyagent import summarize

# From text
summary = summarize("Long article text here...")

# From file
summary = summarize("./report.pdf")

# From URL
summary = summarize("https://example.com/article")

# With options
summary = summarize(text, length="short", bullet_points=True)
```

### rag - 2-Line RAG System

```python
from pyagent import rag

# Index and query
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

emails = extract(document, "all email addresses")
# ["john@email.com", "jane@company.com"]
```

### fetch - Real-Time Data

```python
from pyagent import fetch

weather = fetch.weather("New York")
print(f"{weather.temperature}°C, {weather.conditions}")

news = fetch.news("artificial intelligence")
for article in news:
    print(article.title)

stock = fetch.stock("AAPL")
print(f"${stock.price} ({stock.change_percent}%)")

crypto = fetch.crypto("BTC")
print(f"Bitcoin: ${crypto.price}")
```

### analyze - Data Analysis

```python
from pyagent import analyze

insights = analyze.data(sales_data)
print(insights.summary)
print(insights.recommendations)

sentiment = analyze.sentiment("I love this product!")
# {"sentiment": "positive", "confidence": 0.95}
```

### code - Code Operations

```python
from pyagent import code

# Write code
python_code = code.write("REST API for todo app")

# Review code
review = code.review(my_code)
print(review.issues)
print(review.suggestions)
print(review.score)

# Debug errors
solution = code.debug("TypeError: cannot unpack non-iterable NoneType")

# Explain code
explanation = code.explain(complex_function)

# Refactor
improved = code.refactor(old_code, goal="readability")
```

### chat() - Interactive Sessions

```python
from pyagent import chat

session = chat(persona="teacher")
session("Explain machine learning")
session("What about deep learning?")  # Continues conversation
session("Give me an example")          # Still has context
```

### agent() - Custom Agents

```python
from pyagent import agent

# Custom agent with memory
coder = agent("You are an expert Python developer")
result = coder("Write a REST API")

# Prebuilt personas
researcher = agent(persona="researcher")
findings = researcher("Research quantum computing")
```

---

## 🤖 Agent Framework

The **core/** module provides the full Agent infrastructure.

### Agent Class

```python
from pyagent import Agent
from pyagent.core import LLMConfig, AzureOpenAIProvider

# Simple agent
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant."
)

# Configured agent
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
from pyagent.runner import RunConfig

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
from pyagent.runner import StreamingRunner
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

agent = Agent(
    name="Bot",
    instructions="...",
    memory=memory
)
```

---

## 🔗 Multi-Agent Systems

The **blueprint/** module enables sophisticated multi-agent orchestration.

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
# Routes to SpanishAgent automatically
```

### Orchestration Patterns

```python
from pyagent.blueprint import ChainPattern, RouterPattern, MapReducePattern, SupervisorPattern

# Chain: Sequential processing
chain = ChainPattern()
chain.add("draft", writer)
chain.add("edit", editor)
result = await chain.run("Write about AI")

# Router: Route to specialists
router = RouterPattern()
router.add_route("code", coder, keywords=["code", "python", "function"])
router.add_route("math", calculator, keywords=["calculate", "compute"])
result = await router.run("Write a fibonacci function")

# MapReduce: Parallel processing
mapreduce = MapReducePattern(
    map_agents=[researcher1, researcher2, researcher3],
    reduce_agent=synthesizer
)
result = await mapreduce.run("Research AI from different angles")

# Supervisor: Hierarchical management
supervisor = SupervisorPattern(
    manager=manager_agent,
    workers=[worker1, worker2, worker3]
)
result = await supervisor.run("Complex project")
```

---

## 🛠️ Skills & Tools

The **skills/** module provides composable agent capabilities.

### Creating Tools

```python
from pyagent.skills import tool, action

@tool(description="Search the web for information")
async def web_search(query: str, limit: int = 10) -> list:
    results = await search_api(query)
    return results[:limit]

@tool(description="Send an email")
async def send_email(to: str, subject: str, body: str) -> str:
    # Send email logic
    return f"Email sent to {to}"

@action(name="approve", description="Approve a request")
async def approve_action(request_id: str) -> dict:
    return {"status": "approved", "request_id": request_id}

agent = Agent(
    name="Assistant",
    instructions="Help users with tasks.",
    tools=[web_search, send_email, approve_action]
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

### OpenAPI Tools

```python
from pyagent.openapi import create_tools_from_openapi

# Auto-generate tools from OpenAPI spec
tools = create_tools_from_openapi("petstore.yaml")

agent = Agent(
    name="PetStoreBot",
    instructions="Help users manage pets.",
    tools=tools
)
```

---

## 🔌 Plugin System

The **plugins/** module enables reusable, shareable capabilities.

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

# Use with agent
from pyagent.plugins import PluginRegistry

registry = PluginRegistry()
registry.register(WeatherPlugin())

agent = Agent(
    name="WeatherBot",
    instructions="Help with weather.",
    plugins=[WeatherPlugin()]
)
```

---

## 🧠 Kernel Registry

The **kernel/** module provides MS Semantic Kernel-style service management.

```python
from pyagent.kernel import Kernel, KernelBuilder, Service, ServiceType

# Build kernel
kernel = (KernelBuilder()
    .add_llm(openai_client, name="gpt4", is_default=True)
    .add_llm(azure_client, name="azure-gpt4")
    .add_memory(redis_memory)
    .add_plugin(WeatherPlugin())
    .build())

# Use kernel
result = await kernel.invoke("weather", "get_weather", city="NYC")

# Switch services
kernel.set_default_service("llm", "azure-gpt4")
```

---

## 💾 Session Management

The **sessions/** module provides persistent conversation state.

```python
from pyagent.sessions import SessionManager, SQLiteSessionStore, RedisSessionStore

# SQLite (local)
manager = SessionManager(
    store=SQLiteSessionStore("sessions.db")
)

# Redis (distributed)
manager = SessionManager(
    store=RedisSessionStore(host="localhost", port=6379)
)

# Use sessions
session = manager.get_or_create("user-123")
session.add_user_message("Hello!")
session.add_assistant_message("Hi there!")
manager.save(session)

# Resume later
session = manager.get("user-123")
print(session.messages)  # Previous conversation
```

---

## 📊 Evaluation Framework

The **evaluation/** module enables systematic agent testing.

```python
from pyagent.evaluation import (
    Evaluator, EvalSet, TestCase,
    ExactMatch, ContainsMatch, SemanticSimilarity, LLMJudge
)

# Create test cases
eval_set = EvalSet([
    TestCase(
        input="What is 2+2?",
        expected="4",
        criteria=ExactMatch()
    ),
    TestCase(
        input="Explain Python",
        expected_contains=["programming", "language"],
        criteria=ContainsMatch()
    ),
    TestCase(
        input="Write a haiku about AI",
        criteria=LLMJudge(prompt="Is this a valid haiku?")
    ),
])

# Run evaluation
evaluator = Evaluator(agent)
results = await evaluator.run(eval_set)

print(f"Pass Rate: {results.metrics.pass_rate}%")
print(f"Average Score: {results.metrics.avg_score}")
results.to_csv("eval_results.csv")
```

---

## 🎤 Voice & Audio

The **voice/** module enables real-time voice interactions.

```python
from pyagent.voice import VoiceSession, AudioFormat

async with VoiceSession(agent) as session:
    # Stream audio in
    session.send_audio(audio_chunk)
    
    # Get transcription
    text = await session.get_transcription()
    
    # Get audio response
    response_audio = await session.get_audio_response()
```

---

## 🖼️ Multimodal

The **multimodal/** module supports images, audio, and video.

```python
from pyagent.multimodal import Image, Audio, MultimodalContent

# Image analysis
img = Image.from_file("photo.png")
result = Runner.run_sync(agent, "Describe this image", images=[img])

# From URL
img = Image.from_url("https://example.com/photo.jpg")

# Base64
img = Image.from_base64(base64_string)
```

---

## 🔄 Agent-to-Agent Protocol

The **a2a/** module implements Google's A2A protocol.

```python
from pyagent.a2a import A2AServer, A2AClient, AgentCard

# Server: Expose agent
server = A2AServer(agent)
server.set_card(AgentCard(
    name="WeatherAgent",
    capabilities=["weather_lookup", "forecast"]
))
await server.start(port=8080)

# Client: Connect to remote agents
client = A2AClient()
remote_agent = await client.connect("http://weather-agent:8080")
result = await remote_agent.run("Weather in NYC?")
```

---

## 🗄️ Vector Database Connectors

The **vectordb/** module provides unified vector storage.

```python
from pyagent.vectordb import ChromaStore, PineconeStore, QdrantStore, Document

# ChromaDB
store = ChromaStore(collection="my_docs")

# Pinecone
store = PineconeStore(index="my-index", api_key="...")

# Qdrant
store = QdrantStore(url="http://localhost:6333")

# Index documents
docs = [
    Document(content="AI is transforming...", metadata={"topic": "ai"}),
    Document(content="Machine learning...", metadata={"topic": "ml"}),
]
await store.add(docs)

# Search
results = await store.search("What is AI?", limit=5)
for result in results:
    print(f"{result.score}: {result.document.content}")
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Azure OpenAI (with Azure AD - no API key!)
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

### YAML Configuration

```yaml
# agents/research_assistant.yaml
name: ResearchAssistant
instructions: |
  You are a research assistant that helps users find information.
model: gpt-4o-mini
temperature: 0.7
tools:
  - web_search
  - summarize
memory:
  type: conversation
  max_messages: 50
```

```python
from pyagent.config import load_agent, AgentBuilder

config = load_agent("agents/research_assistant.yaml")
agent = AgentBuilder.from_config(config).build()
```

---

## 🔐 Azure AD Authentication

Enterprise-grade authentication without API keys.

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

# Wrap any function with safety rails
safe_ask = guardrails.wrap(
    ask,
    block_pii=True,
    block_harmful=True,
    max_tokens=1000
)

result = safe_ask("Tell me about...")

# Check for violations
if result.violations:
    print(f"Blocked: {result.violations}")
```

---

## 📍 Tracing & Observability

```python
from pyagent.easy import trace

# Enable tracing
trace.enable()

# Run operations
ask("What is AI?")
research("ML trends")

# View trace
trace.show()

# Export
trace.export("trace.json")
```

---

## 🎮 Pre-built Use Case Agents

```python
from pyagent.usecases import get_agent

# Customer service
support = get_agent("support_agent")
tech_support = get_agent("technical_agent")

# Development
code_reviewer = get_agent("code_reviewer")
debugger = get_agent("debugger")

# Research
data_analyst = get_agent("data_analyst")
market_researcher = get_agent("market_researcher")

# Gaming
npc = get_agent("npc_agent")
game_master = get_agent("game_master")
```

---

## 🧮 Token Counting & Cost

```python
from pyagent.tokens import count_tokens, calculate_cost

# Count tokens
tokens = count_tokens("Hello, how are you?", model="gpt-4o-mini")
print(f"Tokens: {tokens}")

# Calculate cost
cost = calculate_cost(
    input_tokens=1000,
    output_tokens=500,
    model="gpt-4o-mini"
)
print(f"Cost: ${cost:.4f}")
```

---

## 📊 Feature Comparison

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
| Code Execution | ✅ | ✅ | ❌ | ❌ | ✅ |
| Vector DB Connectors | ✅ | ❌ | ❌ | ✅ | ✅ |

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
├── tokens/         # 🧮 Token counting
├── models/         # 🤖 Multi-provider models
├── instructions/   # 📝 Persona and guidelines
├── code_executor/  # 💻 Safe code execution
├── integrations/   # 🔗 LangChain, SK adapters
├── usecases/       # 🎯 Pre-built agents
├── devui/          # 🖥️ Development UI
├── cli/            # ⌨️ Command line interface
└── errors/         # ❌ Error hierarchy
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

---

## 📚 Documentation

- [Getting Started](docs/QUICKSTART.md)
- [API Reference](docs/API_REFERENCE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Azure Setup](docs/AZURE_SETUP.md)
- [Examples](examples/)
- [Changelog](docs/CHANGELOG.md)

---

## 🙏 Acknowledgements

PYAI builds on the excellent work of:

- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) - Runner pattern
- [Microsoft Semantic Kernel](https://github.com/microsoft/semantic-kernel) - Kernel & plugins
- [Google ADK](https://github.com/google/adk-python) - A2A protocol
- [Strands Agents](https://github.com/strands-agents/sdk-python) - Tool discovery
- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) - Token utilities

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>🐼 PYAI</strong><br/>
  <em>Because AI development should be as simple as <code>import pandas as pd</code></em>
</p>

<p align="center">
  <strong>25+ Modules • 150+ Classes • 671 Tests • Infinite Possibilities</strong>
</p>
