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
  <a href="https://pypi.org/project/pyai/"><img src="https://img.shields.io/badge/pypi-v0.4.0-blue" alt="PyPI"/></a>
  <a href="https://python.org/"><img src="https://img.shields.io/badge/python-3.10+-green" alt="Python"/></a>
  <a href="#"><img src="https://img.shields.io/badge/tests-671%20passing-brightgreen" alt="Tests"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-yellow" alt="License"/></a>
  <a href="#"><img src="https://img.shields.io/badge/modules-25+-orange" alt="Modules"/></a>
  <a href="#"><img src="https://img.shields.io/badge/classes-150+-red" alt="Classes"/></a>
</p>

<p align="center">
  <a href="#-what-is-pyai">What is PYAI</a> •
  <a href="#-the-three-dimensions">Three Dimensions</a> •
  <a href="#-why-pyai-one-stop-intelligence-solution">Why PYAI</a> •
  <a href="#-software-factories">Software Factories</a> •
  <a href="#-complete-module-reference">Modules</a> •
  <a href="#-the-pyai-product-suite">Ecosystem</a>
</p>

---

## 🎯 What is PYAI?

**PYAI is not just another AI library. It's an Intelligence Engine.**

While other frameworks help you *call* AI models, PYAI embeds intelligence *into* your software architecture. It's the foundation for building **Software Factories** — systems that don't just use AI, but think, adapt, and create.

> *"The best code is the code you never had to write. The best software is the software that writes itself."*

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': 'transparent', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#ffffff', 'lineColor': '#ffffff', 'secondaryColor': 'transparent', 'tertiaryColor': 'transparent', 'background': 'transparent', 'mainBkg': 'transparent', 'nodeBorder': '#ffffff', 'clusterBkg': 'transparent', 'clusterBorder': '#ffffff', 'titleColor': '#ffffff', 'edgeLabelBackground': 'transparent', 'nodeTextColor': '#ffffff'}}}%%
flowchart LR
    subgraph Traditional["Traditional AI Libraries"]
        A["Your Code"] -->|calls| B["AI API"]
        B -->|returns| A
    end
    
    subgraph PYAI["PYAI Intelligence Engine"]
        C["Application"] <-->|embedded| D["🧠 PYAI"]
        D <-->|orchestrates| E["Agents"]
        D <-->|manages| F["Memory"]
        D <-->|executes| G["Workflows"]
        D -->|connects| H["LLM Providers"]
    end
```

Built on **pyai**, our core SDK, PYAI provides **25+ modules** with **150+ classes** covering every AI use case.

---

## 🔺 The Three Dimensions

PYAI operates across **three dimensions of intelligence**, each building upon the last:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': 'transparent', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#ffffff', 'lineColor': '#ffffff', 'secondaryColor': 'transparent', 'tertiaryColor': 'transparent', 'background': 'transparent', 'mainBkg': 'transparent', 'nodeBorder': '#ffffff', 'clusterBkg': 'transparent', 'clusterBorder': '#ffffff', 'titleColor': '#ffffff', 'edgeLabelBackground': 'transparent', 'nodeTextColor': '#ffffff'}}}%%
flowchart TB
    subgraph D3["🏭 DIMENSION 3: CREATION"]
        direction LR
        C1["Self-generating<br/>Systems"]
        C2["Code Synthesis<br/>Engines"]
        C3["Autonomous<br/>Development"]
    end
    
    subgraph D2["🔗 DIMENSION 2: ORCHESTRATION"]
        direction LR
        O1["Agent<br/>Coordination"]
        O2["Workflow<br/>Automation"]
        O3["Knowledge<br/>Synthesis"]
    end
    
    subgraph D1["🧠 DIMENSION 1: COGNITION"]
        direction LR
        K1["ask • research"]
        K2["summarize • analyze"]
        K3["extract • generate"]
    end
    
    D1 -->|"builds"| D2
    D2 -->|"enables"| D3
```

| Dimension | Purpose | Key Components |
|-----------|---------|----------------|
| **🧠 Cognition** | Single AI operations | `ask()`, `research()`, `summarize()`, `extract()` |
| **🔗 Orchestration** | Multi-agent coordination | `Agent`, `Workflow`, `Handoff`, `Patterns` |
| **🏭 Creation** | Self-generating systems | `code.write()`, `code.review()`, Software Factories |

### Dimension 1️⃣ — Cognition
The foundation. Single-purpose AI operations that **just work**.

```python
from pyai import ask, summarize, extract

# Instant intelligence
answer = ask("Explain quantum entanglement")
summary = summarize(long_document)
entities = extract(text, fields=["names", "dates", "amounts"])
```

### Dimension 2️⃣ — Orchestration
Coordinated intelligence. Multiple agents working in harmony.

```python
from pyai import Agent, Runner
from pyai.blueprint import Workflow, Step

# Create specialized agents
researcher = Agent(name="Researcher", instructions="Find information.")
analyst = Agent(name="Analyst", instructions="Analyze data deeply.")
writer = Agent(name="Writer", instructions="Write compelling content.")

# Build workflow
workflow = (Workflow("ResearchPipeline")
    .add_step(Step("research", researcher))
    .add_step(Step("analyze", analyst))
    .add_step(Step("write", writer))
    .build())
```

### Dimension 3️⃣ — Creation
Self-generating systems. **The Software Factory.**

```python
from pyai import code

# Generate code from description
api_code = code.write("REST API for user management with JWT auth")

# Review and improve
review = code.review(existing_code)
improved = code.refactor(old_code, goal="async architecture")

# Generate tests
tests = code.test(my_function)
```

---

## ✨ Why PYAI: One-Stop Intelligence Solution

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': 'transparent', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#ffffff', 'lineColor': '#ffffff', 'secondaryColor': 'transparent', 'tertiaryColor': 'transparent', 'background': 'transparent', 'mainBkg': 'transparent', 'nodeBorder': '#ffffff', 'clusterBkg': 'transparent', 'clusterBorder': '#ffffff', 'titleColor': '#ffffff', 'edgeLabelBackground': 'transparent', 'nodeTextColor': '#ffffff'}}}%%
flowchart TB
    subgraph PYAI["🧠 PYAI - One-Stop Solution"]
        subgraph Cognition["Cognition"]
            ASK["ask"]
            RES["research"]
            SUM["summarize"]
            RAG["rag"]
            GEN["generate"]
        end
        
        subgraph Orchestration["Orchestration"]
            AGT["Agents"]
            WRK["Workflows"]
            HND["Handoffs"]
            PAT["Patterns"]
        end
        
        subgraph Enterprise["Enterprise"]
            AUTH["Azure AD"]
            SESS["Sessions"]
            EVAL["Evaluation"]
            TRACE["Tracing"]
        end
        
        subgraph Integrations["Integrations"]
            VEC["Vector DBs"]
            API["OpenAPI"]
            PLG["Plugins"]
            MCP["MCP/A2A"]
        end
    end
```

### The Problem with Current Frameworks

| Challenge | LangChain | CrewAI | PYAI Solution |
|-----------|-----------|--------|---------------|
| Simple question | 10+ lines of setup | N/A | `ask("question")` |
| RAG system | 15+ lines, multiple classes | N/A | 2 lines |
| Agent with tools | Complex chains | YAML configs | 5 lines Python |
| Multi-agent | 40+ lines | 50+ lines | 10 lines |
| Memory | External setup | Limited | Built-in |
| Production | DIY | DIY | Included |

### Lines of Code Comparison

| Task | LangChain | LlamaIndex | CrewAI | **PYAI** |
|------|-----------|------------|--------|----------|
| Question Answering | 15 | 12 | N/A | **1** |
| RAG System | 25 | 20 | N/A | **2** |
| Agent with Tools | 30 | 25 | 30 | **5** |
| Multi-Agent Pipeline | 50 | 40 | 60 | **10** |
| Research Assistant | 45 | 35 | 50 | **1** |

---

## 🏭 Software Factories

A **Software Factory** is a system that generates software, not just code snippets.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': 'transparent', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#ffffff', 'lineColor': '#ffffff', 'secondaryColor': 'transparent', 'tertiaryColor': 'transparent', 'background': 'transparent', 'mainBkg': 'transparent', 'nodeBorder': '#ffffff', 'clusterBkg': 'transparent', 'clusterBorder': '#ffffff', 'titleColor': '#ffffff', 'edgeLabelBackground': 'transparent', 'nodeTextColor': '#ffffff'}}}%%
flowchart LR
    subgraph Traditional["Traditional Development"]
        T1["📝 Write"] --> T2["🐛 Debug"] --> T3["📋 Test"] --> T4["📖 Document"]
    end
    
    subgraph Factory["Software Factory"]
        F1["💬 Describe"] --> F2["🏭 Generate"] --> F3["✅ Validate"] --> F4["🚀 Deploy"]
    end
```

| Aspect | Traditional | Software Factory |
|--------|-------------|------------------|
| **Input** | Code | Natural Language |
| **Process** | Manual Writing | AI Generation |
| **Testing** | Manual | Auto-generated |
| **Debugging** | Line by line | Self-healing |
| **Time** | Hours/Days | Seconds/Minutes |

---

## 📚 Architecture Overview

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': 'transparent', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#ffffff', 'lineColor': '#ffffff', 'secondaryColor': 'transparent', 'tertiaryColor': 'transparent', 'background': 'transparent', 'mainBkg': 'transparent', 'nodeBorder': '#ffffff', 'clusterBkg': 'transparent', 'clusterBorder': '#ffffff', 'titleColor': '#ffffff', 'edgeLabelBackground': 'transparent', 'nodeTextColor': '#ffffff'}}}%%
flowchart TB
    subgraph Application["YOUR APPLICATION"]
        APP["🖥️ App Layer"]
    end
    
    subgraph SDK["PYAI SDK - src/pyai/"]
        subgraph Easy["🚀 easy/"]
            E1["ask • research • summarize"]
            E2["rag • generate • translate"]
            E3["fetch • analyze • code"]
            E4["handoff • guardrails • trace"]
        end
        
        subgraph Core["🧠 core/"]
            C1["Agent"]
            C2["LLMProvider"]
            C3["Memory"]
        end
        
        subgraph Runner["⚡ runner/"]
            R1["Runner"]
            R2["StreamingRunner"]
        end
        
        subgraph Blueprint["🔗 blueprint/"]
            B1["Workflow"]
            B2["Patterns"]
        end
        
        subgraph Skills["🛠️ skills/"]
            S1["@tool decorator"]
            S2["SkillRegistry"]
        end
        
        subgraph Kernel["🔌 kernel/"]
            K1["Kernel"]
            K2["ServiceRegistry"]
        end
    end
    
    subgraph Providers["LLM PROVIDERS"]
        P1["Azure OpenAI"]
        P2["OpenAI"]
        P3["Anthropic"]
        P4["Ollama"]
    end
    
    Application --> SDK
    SDK --> Providers
```

---

## 📦 Complete Module Reference

### File Structure

```
src/pyai/
├── easy/           # One-liner APIs (15+ functions)
├── core/           # Agent, Memory, LLM providers
├── runner/         # Execution engine
├── blueprint/      # Workflows and patterns
├── skills/         # Tools and skills system
├── kernel/         # Service registry (SK pattern)
├── sessions/       # SQLite/Redis persistence
├── evaluation/     # Agent testing framework
├── voice/          # Real-time voice
├── multimodal/     # Image, audio, video
├── vectordb/       # Vector database connectors
├── openapi/        # OpenAPI tool generation
├── plugins/        # Plugin architecture
├── a2a/            # Agent-to-Agent protocol
├── config/         # YAML configuration
├── tokens/         # Token counting
└── tools/          # Built-in tools
```

---

## 🎯 One-Liner APIs (`easy/` module)

The `easy/` module provides **15+ one-liner APIs** that handle complex AI tasks with zero setup.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': 'transparent', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#ffffff', 'lineColor': '#ffffff', 'secondaryColor': 'transparent', 'tertiaryColor': 'transparent', 'background': 'transparent', 'mainBkg': 'transparent', 'nodeBorder': '#ffffff', 'clusterBkg': 'transparent', 'clusterBorder': '#ffffff', 'titleColor': '#ffffff', 'edgeLabelBackground': 'transparent', 'nodeTextColor': '#ffffff'}}}%%
flowchart TB
    subgraph QA["Question Answering"]
        ASK["ask()"]
        RES["research()"]
    end
    
    subgraph Content["Content Processing"]
        SUM["summarize()"]
        TRANS["translate()"]
        EXT["extract()"]
        GEN["generate()"]
    end
    
    subgraph Knowledge["Knowledge Management"]
        RAGA["rag.index()"]
        RAGQ["rag.ask()"]
    end
    
    subgraph RealTime["Real-Time Data"]
        FW["fetch.weather()"]
        FN["fetch.news()"]
        FS["fetch.stock()"]
    end
    
    subgraph Code["Code Operations"]
        CW["code.write()"]
        CR["code.review()"]
        CD["code.debug()"]
        CT["code.test()"]
    end
    
    subgraph Analysis["Analysis"]
        AS["analyze.sentiment()"]
        AE["analyze.entities()"]
        AC["analyze.classify()"]
    end
```

### ask() — Universal Question Answering

The foundation of PYAI. Ask any question, get an intelligent answer.

```python
from pyai import ask

# Simple questions
answer = ask("What is Python?")

# Detailed responses
answer = ask("Explain quantum computing", detailed=True)

# Formatted output
answer = ask("List 5 programming tips", format="bullet")

# With context
answer = ask("What does this code do?", context=my_code)

# Async version
answer = await ask_async("What is AI?")
```

### research() — Deep Topic Research

Multi-step research with automatic source gathering and synthesis.

```python
from pyai import research

# Basic research
result = research("AI trends in enterprise software")

# Access structured results
print(result.summary)        # Executive summary
print(result.key_points)     # Bullet points
print(result.insights)       # Deep analysis
print(result.sources)        # References

# Research with specific focus
result = research(
    topic="Machine learning in healthcare",
    depth="comprehensive",
    max_sources=10
)
```

### summarize() — Document Summarization

Summarize any content: text, files, URLs.

```python
from pyai import summarize

# Text summarization
summary = summarize(long_document)

# File summarization (PDF, Word, etc.)
summary = summarize("./report.pdf")

# URL summarization
summary = summarize("https://example.com/article")

# Custom length
summary = summarize(text, length="short")    # ~2 sentences
summary = summarize(text, length="medium")   # ~1 paragraph
summary = summarize(text, length="long")     # Detailed
```

### rag — Retrieval-Augmented Generation

Production-ready RAG in 2 lines.

```python
from pyai import rag

# Index documents
knowledge = rag.index("./documents")

# Query the knowledge base
answer = knowledge.ask("What is the main conclusion?")

# With source attribution
result = knowledge.ask("What were the key findings?", return_sources=True)
print(result.answer)
print(result.sources)

# Multiple document types
rag.index(["./pdfs", "./markdown", "./code"])
```

### generate() — Content Generation

Generate any type of content.

```python
from pyai import generate

# Code generation
code = generate("fibonacci function", type="code")
api = generate("REST API for user management", type="code", language="python")

# Email generation
email = generate("polite rejection email", type="email")

# Article generation
article = generate("Introduction to AI", type="article", length="1000 words")

# Custom types
plan = generate("project plan for mobile app", type="plan")
```

### translate() — Language Translation

```python
from pyai import translate

# Simple translation
spanish = translate("Hello, how are you?", to="spanish")
japanese = translate("Good morning", to="japanese")

# Detect and translate
result = translate(unknown_text, to="english")
print(result.detected_language)  # "french"
print(result.translated)         # English text

# Preserve formatting
translated_doc = translate(markdown_text, to="german", preserve_format=True)
```

### extract() — Structured Data Extraction

Extract structured data from unstructured text.

```python
from pyai import extract

# Extract specific fields
data = extract(email_text, fields=["sender", "date", "subject", "action_items"])

# With types
data = extract(invoice, fields={
    "vendor": "string",
    "amount": "float",
    "date": "date",
    "line_items": "list"
})

# Entity extraction
entities = extract(article, fields=["people", "organizations", "locations"])
```

### fetch — Real-Time Data

Access live data feeds.

```python
from pyai import fetch

# Weather data
weather = fetch.weather("New York")
print(weather.temperature)
print(weather.conditions)

# News
headlines = fetch.news("artificial intelligence")
for article in headlines:
    print(article.title, article.source)

# Stock data
stock = fetch.stock("AAPL")
print(stock.price, stock.change)

# Web content
content = fetch.url("https://example.com")
```

### analyze — Data Analysis

```python
from pyai import analyze

# Sentiment analysis
result = analyze.sentiment("I love this product!")
print(result.label)     # "positive"
print(result.score)     # 0.95

# Entity recognition
entities = analyze.entities("Apple CEO Tim Cook announced...")
# [{"text": "Apple", "type": "ORG"}, {"text": "Tim Cook", "type": "PERSON"}]

# Classification
category = analyze.classify(text, categories=["tech", "sports", "politics"])

# Comparison
comparison = analyze.compare(text1, text2)
print(comparison.similarity)
print(comparison.differences)
```

### code — Code Operations

AI-powered code assistant.

```python
from pyai import code

# Write code
implementation = code.write("binary search tree in Python")
api = code.write("FastAPI CRUD endpoints for users", framework="fastapi")

# Review code
review = code.review(my_code)
print(review.issues)
print(review.suggestions)
print(review.score)

# Debug errors
fix = code.debug("TypeError: 'NoneType' object is not subscriptable", context=my_code)
print(fix.explanation)
print(fix.solution)

# Generate tests
tests = code.test(my_function)
print(tests.test_cases)

# Refactor
improved = code.refactor(legacy_code, goal="async/await pattern")

# Explain code
explanation = code.explain(complex_function)
```

### handoff() — Agent Delegation

Transfer tasks between agents.

```python
from pyai import handoff

# Transfer to specialist
result = handoff(
    task="Complex legal analysis",
    to_agent=legal_specialist,
    context=case_details
)

# With routing
result = handoff(
    task=user_request,
    routes={
        "code": coder_agent,
        "math": calculator_agent,
        "writing": writer_agent
    }
)
```

### guardrails() — Safety Wrappers

```python
from pyai.easy import guardrails

# Wrap any function with safety
safe_ask = guardrails.wrap(ask, block_pii=True, block_harmful=True)

# Custom validators
safe_generate = guardrails.wrap(generate, 
    validators=[no_code_execution, family_friendly])

# Rate limiting
limited_ask = guardrails.wrap(ask, rate_limit="10/minute")
```

### trace() — Debugging & Observability

```python
from pyai.easy import trace

# Enable tracing
trace.enable()

# Run your code
result = ask("What is AI?")
research_result = research("Machine learning")

# View traces
trace.show()
# Displays: tokens used, latency, model calls, cost

# Export for analysis
trace.export("traces.json")
```

---

## 🤖 Agent Framework (`core/` module)

The `core/` module provides the foundational building blocks for intelligent agents.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': 'transparent', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#ffffff', 'lineColor': '#ffffff', 'secondaryColor': 'transparent', 'tertiaryColor': 'transparent', 'background': 'transparent', 'mainBkg': 'transparent', 'nodeBorder': '#ffffff', 'clusterBkg': 'transparent', 'clusterBorder': '#ffffff', 'titleColor': '#ffffff', 'edgeLabelBackground': 'transparent', 'nodeTextColor': '#ffffff'}}}%%
classDiagram
    class Agent {
        +name: str
        +instructions: str
        +tools: List~Tool~
        +memory: Memory
        +model: str
        +run(input) RunResult
    }
    
    class AgentConfig {
        +model: str
        +temperature: float
        +max_tokens: int
        +tools: List
    }
    
    class Memory {
        +add(message)
        +get_context()
        +clear()
    }
    
    class LLMProvider {
        +generate(prompt) Response
        +stream(prompt) AsyncIterator
    }
    
    Agent --> AgentConfig
    Agent --> Memory
    Agent --> LLMProvider
```

### Agent Execution Flow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': 'transparent', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#ffffff', 'lineColor': '#ffffff', 'secondaryColor': 'transparent', 'tertiaryColor': 'transparent', 'background': 'transparent', 'mainBkg': 'transparent', 'nodeBorder': '#ffffff', 'clusterBkg': 'transparent', 'clusterBorder': '#ffffff', 'titleColor': '#ffffff', 'edgeLabelBackground': 'transparent', 'nodeTextColor': '#ffffff'}}}%%
sequenceDiagram
    participant User
    participant Runner
    participant Agent
    participant Memory
    participant LLM
    participant Tools
    
    User->>Runner: run_sync(agent, "Query")
    Runner->>Agent: execute(input)
    Agent->>Memory: get_context()
    Memory-->>Agent: conversation_history
    Agent->>LLM: generate(prompt + context)
    LLM-->>Agent: response + tool_calls
    
    alt Has Tool Calls
        loop For each tool call
            Agent->>Tools: execute(tool_call)
            Tools-->>Agent: result
        end
        Agent->>LLM: generate(with tool results)
        LLM-->>Agent: final response
    end
    
    Agent->>Memory: add(input, response)
    Agent-->>Runner: RunResult
    Runner-->>User: result.final_output
```

### Creating Agents

```python
from pyai import Agent, Runner
from pyai.skills import tool

# Define custom tools
@tool(description="Get current weather for a city")
async def get_weather(city: str) -> str:
    """Fetch weather data for the specified city."""
    return f"Weather in {city}: Sunny, 72°F"

@tool(description="Search the knowledge base")
async def search_kb(query: str) -> str:
    """Search internal knowledge base."""
    return f"Found 3 results for '{query}'"

# Create the agent
agent = Agent(
    name="WeatherBot",
    instructions="""You are a helpful weather assistant.
    Always provide accurate weather information.
    If asked about other topics, politely redirect to weather.""",
    tools=[get_weather, search_kb],
    model="gpt-4o-mini"
)

# Run synchronously
result = Runner.run_sync(agent, "What's the weather in Tokyo?")
print(result.final_output)

# Run asynchronously
result = await Runner.run(agent, "Weather in Paris?")
print(result.final_output)
```

### Agent Configuration

```python
from pyai import Agent
from pyai.core import AgentConfig

# Detailed configuration
config = AgentConfig(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9,
    presence_penalty=0.1,
    frequency_penalty=0.1
)

agent = Agent(
    name="Analyst",
    instructions="Analyze data thoroughly.",
    config=config
)
```

### Memory Management

```python
from pyai import Agent
from pyai.core import ConversationMemory, SlidingWindowMemory

# Conversation memory (keeps all messages)
agent = Agent(
    name="Assistant",
    instructions="Help users.",
    memory=ConversationMemory()
)

# Sliding window (keeps last N messages)
agent = Agent(
    name="Assistant",
    instructions="Help users.",
    memory=SlidingWindowMemory(window_size=10)
)

# Access memory
agent.memory.add("user", "Hello")
agent.memory.add("assistant", "Hi there!")
context = agent.memory.get_context()
```

### Streaming Responses

```python
from pyai import Agent, Runner

agent = Agent(name="Assistant", instructions="Be helpful.")

# Stream tokens as they arrive
async for chunk in Runner.stream(agent, "Tell me a story"):
    print(chunk, end="", flush=True)
```

---

## 🔗 Multi-Agent Systems (`blueprint/` module)

The `blueprint/` module enables sophisticated multi-agent orchestration.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': 'transparent', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#ffffff', 'lineColor': '#ffffff', 'secondaryColor': 'transparent', 'tertiaryColor': 'transparent', 'background': 'transparent', 'mainBkg': 'transparent', 'nodeBorder': '#ffffff', 'clusterBkg': 'transparent', 'clusterBorder': '#ffffff', 'titleColor': '#ffffff', 'edgeLabelBackground': 'transparent', 'nodeTextColor': '#ffffff'}}}%%
flowchart TB
    subgraph Patterns["Available Patterns"]
        direction TB
        P1["🔗 Chain<br/>Sequential Processing"]
        P2["🔀 Router<br/>Dynamic Routing"]
        P3["📊 MapReduce<br/>Parallel Processing"]
        P4["👔 Supervisor<br/>Managed Workers"]
        P5["🔄 Loop<br/>Iterative Refinement"]
    end
```

### Architecture Patterns

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': 'transparent', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#ffffff', 'lineColor': '#ffffff', 'secondaryColor': 'transparent', 'tertiaryColor': 'transparent', 'background': 'transparent', 'mainBkg': 'transparent', 'nodeBorder': '#ffffff', 'clusterBkg': 'transparent', 'clusterBorder': '#ffffff', 'titleColor': '#ffffff', 'edgeLabelBackground': 'transparent', 'nodeTextColor': '#ffffff'}}}%%
flowchart LR
    subgraph Chain["Chain Pattern"]
        CA1["📝 Draft"] --> CA2["✏️ Edit"] --> CA3["✅ Review"]
    end
    
    subgraph Router["Router Pattern"]
        RR["🔀 Router"] --> RA1["💻 Code"]
        RR --> RA2["📐 Math"]
        RR --> RA3["📝 Writing"]
    end
```

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': 'transparent', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#ffffff', 'lineColor': '#ffffff', 'secondaryColor': 'transparent', 'tertiaryColor': 'transparent', 'background': 'transparent', 'mainBkg': 'transparent', 'nodeBorder': '#ffffff', 'clusterBkg': 'transparent', 'clusterBorder': '#ffffff', 'titleColor': '#ffffff', 'edgeLabelBackground': 'transparent', 'nodeTextColor': '#ffffff'}}}%%
flowchart TB
    subgraph MapReduce["MapReduce Pattern"]
        MR1["📄 Doc 1"] --> MAP1["Analyzer"]
        MR2["📄 Doc 2"] --> MAP2["Analyzer"]
        MR3["📄 Doc 3"] --> MAP3["Analyzer"]
        MAP1 --> RED["Synthesizer"]
        MAP2 --> RED
        MAP3 --> RED
    end
    
    subgraph Supervisor["Supervisor Pattern"]
        SUP["👔 Manager"] --> SW1["Worker 1"]
        SUP --> SW2["Worker 2"]
        SUP --> SW3["Worker 3"]
        SW1 -.-> SUP
        SW2 -.-> SUP
        SW3 -.-> SUP
    end
```

### Workflow Definition

```python
from pyai import Agent
from pyai.blueprint import Workflow, Step

# Create specialized agents
researcher = Agent(
    name="Researcher",
    instructions="Research topics thoroughly. Return structured findings."
)

writer = Agent(
    name="Writer",
    instructions="Write engaging content based on research."
)

editor = Agent(
    name="Editor",
    instructions="Edit and polish content for clarity."
)

# Build sequential workflow
workflow = (Workflow("ContentPipeline")
    .add_step(Step("research", researcher, output_key="research"))
    .add_step(Step("write", writer, input_key="research", output_key="draft"))
    .add_step(Step("edit", editor, input_key="draft", output_key="final"))
    .build())

# Execute
result = await workflow.run("Write about AI in healthcare")
print(result.outputs["final"])
```

### Chain Pattern

```python
from pyai.blueprint import ChainPattern

# Create a chain of agents
chain = ChainPattern([
    ("draft", drafter),
    ("review", reviewer),
    ("polish", editor)
])

# Output of each agent feeds into the next
result = await chain.run("Create a product announcement")
```

### Router Pattern

```python
from pyai.blueprint import RouterPattern

# Create router with specialized agents
router = RouterPattern()
router.add_route("code", code_agent, keywords=["python", "javascript", "bug"])
router.add_route("math", math_agent, keywords=["calculate", "equation", "number"])
router.add_route("writing", writer_agent, keywords=["write", "essay", "email"])
router.add_route("default", general_agent)

# Router automatically selects the right agent
result = await router.run("Fix this Python bug: ...")
# -> Routes to code_agent

result = await router.run("Calculate 234 * 567")
# -> Routes to math_agent
```

### MapReduce Pattern

```python
from pyai.blueprint import MapReducePattern

# Analyze multiple documents in parallel
analyzer = Agent(name="Analyzer", instructions="Analyze document content.")
synthesizer = Agent(name="Synthesizer", instructions="Synthesize findings.")

map_reduce = MapReducePattern(
    mapper=analyzer,
    reducer=synthesizer
)

documents = ["doc1.txt", "doc2.txt", "doc3.txt"]
result = await map_reduce.run(documents)
# Analyzes all docs in parallel, then synthesizes
```

### Supervisor Pattern

```python
from pyai.blueprint import SupervisorPattern

# Manager delegates to workers
manager = Agent(
    name="Manager",
    instructions="Delegate tasks and synthesize results."
)

workers = [
    Agent(name="Coder", instructions="Write code."),
    Agent(name="Tester", instructions="Write tests."),
    Agent(name="Documenter", instructions="Write docs.")
]

supervisor = SupervisorPattern(manager=manager, workers=workers)
result = await supervisor.run("Build a calculator module")

---

## 🔌 Kernel Registry (`kernel/` module)

Microsoft Semantic Kernel-style service management:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': 'transparent', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#ffffff', 'lineColor': '#ffffff', 'secondaryColor': 'transparent', 'tertiaryColor': 'transparent', 'background': 'transparent', 'mainBkg': 'transparent', 'nodeBorder': '#ffffff', 'clusterBkg': 'transparent', 'clusterBorder': '#ffffff', 'titleColor': '#ffffff', 'edgeLabelBackground': 'transparent', 'nodeTextColor': '#ffffff'}}}%%
flowchart TB
    subgraph Kernel["Kernel"]
        SR["ServiceRegistry"]
        FR["FilterRegistry"]
        PR["PluginRegistry"]
        
        SR --> LLM1["GPT-4"]
        SR --> LLM2["Claude"]
        SR --> MEM["Redis Memory"]
        
        PR --> P1["WeatherPlugin"]
        PR --> P2["SearchPlugin"]
        
        FR --> F1["LoggingFilter"]
        FR --> F2["ValidationFilter"]
    end
```

```python
from pyai.kernel import Kernel, KernelBuilder

kernel = (KernelBuilder()
    .add_llm(openai_client, name="gpt4", is_default=True)
    .add_llm(azure_client, name="azure")
    .add_memory(redis_memory)
    .add_plugin(WeatherPlugin())
    .build())

result = await kernel.invoke("weather", "get_weather", city="NYC")
```

---

## 🏢 Enterprise Features

PYAI is built for production. Every feature you need to deploy AI at scale.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': 'transparent', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#ffffff', 'lineColor': '#ffffff', 'secondaryColor': 'transparent', 'tertiaryColor': 'transparent', 'background': 'transparent', 'mainBkg': 'transparent', 'nodeBorder': '#ffffff', 'clusterBkg': 'transparent', 'clusterBorder': '#ffffff', 'titleColor': '#ffffff', 'edgeLabelBackground': 'transparent', 'nodeTextColor': '#ffffff'}}}%%
flowchart TB
    subgraph Enterprise["Enterprise Features"]
        direction TB
        AUTH["🔐 Azure AD<br/>Authentication"]
        SESS["💾 Session<br/>Management"]
        EVAL["📊 Testing &<br/>Evaluation"]
        TRACE["📍 Tracing &<br/>Observability"]
        GUARD["🛡️ Guardrails &<br/>Safety"]
        MONITOR["📈 Monitoring &<br/>Analytics"]
    end
```

### 🔐 Azure AD Authentication

Seamless integration with Azure Active Directory. No API keys needed in production.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': 'transparent', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#ffffff', 'lineColor': '#ffffff', 'secondaryColor': 'transparent', 'tertiaryColor': 'transparent', 'background': 'transparent', 'mainBkg': 'transparent', 'nodeBorder': '#ffffff', 'clusterBkg': 'transparent', 'clusterBorder': '#ffffff', 'titleColor': '#ffffff', 'edgeLabelBackground': 'transparent', 'nodeTextColor': '#ffffff'}}}%%
sequenceDiagram
    participant App
    participant PYAI
    participant AzureAD["Azure AD"]
    participant AOAI["Azure OpenAI"]
    
    App->>PYAI: ask("question")
    PYAI->>AzureAD: Get token (DefaultAzureCredential)
    AzureAD-->>PYAI: Bearer token
    PYAI->>AOAI: API call with token
    AOAI-->>PYAI: Response
    PYAI-->>App: Answer
```

```python
import os

# Configure Azure OpenAI (no API key needed!)
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource.openai.azure.com/"
os.environ["AZURE_OPENAI_DEPLOYMENT"] = "gpt-4o-mini"

from pyai import ask

# Uses your az login credentials or Managed Identity automatically
answer = ask("Hello from Azure!")
```

**Supported Authentication Methods:**
- `az login` (Developer workstations)
- Managed Identity (Azure VMs, App Service, AKS)
- Service Principal (CI/CD pipelines)
- Workload Identity (Kubernetes)

### 💾 Session Management

Persistent conversation history with SQLite or Redis backends.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': 'transparent', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#ffffff', 'lineColor': '#ffffff', 'secondaryColor': 'transparent', 'tertiaryColor': 'transparent', 'background': 'transparent', 'mainBkg': 'transparent', 'nodeBorder': '#ffffff', 'clusterBkg': 'transparent', 'clusterBorder': '#ffffff', 'titleColor': '#ffffff', 'edgeLabelBackground': 'transparent', 'nodeTextColor': '#ffffff'}}}%%
flowchart LR
    subgraph App["Application"]
        U1["User A"]
        U2["User B"]
        U3["User C"]
    end
    
    subgraph Sessions["SessionManager"]
        SM["Session<br/>Manager"]
    end
    
    subgraph Storage["Storage Backend"]
        SQL["SQLite<br/>sessions.db"]
        RED["Redis<br/>Cluster"]
    end
    
    U1 --> SM
    U2 --> SM
    U3 --> SM
    SM --> SQL
    SM --> RED
```

```python
from pyai.sessions import SessionManager, SQLiteSessionStore, RedisSessionStore

# SQLite for development
manager = SessionManager(store=SQLiteSessionStore("sessions.db"))

# Redis for production
manager = SessionManager(store=RedisSessionStore(
    host="redis.example.com",
    port=6379,
    password="secret"
))

# Create and use sessions
session = await manager.create(user_id="user123")
session.add_message("user", "Hello")
session.add_message("assistant", "Hi there!")

# Resume later
session = await manager.get(session_id="abc123")
history = session.get_messages()

# Session with agent
from pyai import Agent, Runner

agent = Agent(name="Assistant", instructions="Be helpful.")
result = await Runner.run(agent, "Hello", session=session)
# Automatically maintains conversation history
```

### 📊 Evaluation Framework

Test your agents systematically.

```python
from pyai.evaluation import Evaluator, EvalSet, TestCase, metrics

# Define test cases
eval_set = EvalSet([
    TestCase(
        input="What is 2+2?",
        expected="4",
        tags=["math"]
    ),
    TestCase(
        input="Capital of France?",
        expected="Paris",
        tags=["geography"]
    ),
    TestCase(
        input="Write a haiku about coding",
        expected_pattern=r".*\n.*\n.*",  # 3 lines
        tags=["creative"]
    )
])

# Run evaluation
evaluator = Evaluator(agent)
results = await evaluator.run(eval_set)

# View results
print(f"Pass rate: {results.pass_rate}%")
print(f"Average latency: {results.avg_latency}ms")

for result in results.failed:
    print(f"Failed: {result.input}")
    print(f"Expected: {result.expected}")
    print(f"Got: {result.actual}")
```

### 📍 Tracing & Observability

Full visibility into agent operations.

```python
from pyai.easy import trace

# Enable tracing
trace.enable()

# Run operations
result = ask("Explain quantum computing")
research_result = research("AI in healthcare")

# View traces
trace.show()
# Output:
# ┌─ ask("Explain quantum computing")
# │  Model: gpt-4o-mini
# │  Tokens: 45 in, 230 out
# │  Latency: 1.2s
# │  Cost: $0.0012
# └─

# Export for external tools
trace.export("traces.json")
trace.export_to_opentelemetry()
```

### 🛡️ Guardrails & Safety

Built-in protection for production deployments.

```python
from pyai.easy import guardrails

# PII protection
safe_ask = guardrails.wrap(ask, block_pii=True)
# Blocks: SSNs, credit cards, phone numbers

# Content filtering
safe_generate = guardrails.wrap(generate, 
    block_harmful=True,
    block_adult=True
)

# Custom validators
def no_financial_advice(response):
    if "invest" in response.lower():
        return False, "Cannot provide investment advice"
    return True, None

safe_ask = guardrails.wrap(ask, validators=[no_financial_advice])

# Rate limiting
limited_ask = guardrails.wrap(ask, rate_limit="100/hour")

# Token limits
bounded_ask = guardrails.wrap(ask, max_tokens=500)
```

---

## 🔗 Integrations

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': 'transparent', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#ffffff', 'lineColor': '#ffffff', 'secondaryColor': 'transparent', 'tertiaryColor': 'transparent', 'background': 'transparent', 'mainBkg': 'transparent', 'nodeBorder': '#ffffff', 'clusterBkg': 'transparent', 'clusterBorder': '#ffffff', 'titleColor': '#ffffff', 'edgeLabelBackground': 'transparent', 'nodeTextColor': '#ffffff'}}}%%
flowchart TB
    PYAI["🧠 PYAI"] --> VEC["Vector DBs"]
    PYAI --> FRAME["Frameworks"]
    PYAI --> PROTO["Protocols"]
    
    VEC --> CH["ChromaDB"]
    VEC --> PC["Pinecone"]
    VEC --> QD["Qdrant"]
    VEC --> AZ["Azure AI Search"]
    
    FRAME --> LC["LangChain"]
    FRAME --> SK["Semantic Kernel"]
    
    PROTO --> MCP["MCP Protocol"]
    PROTO --> A2A["A2A Protocol"]
    PROTO --> OAPI["OpenAPI"]
```

---

## 📊 Feature Comparison

| Feature | PYAI | OpenAI Agents | Google ADK | Semantic Kernel | LangChain |
|---------|:----:|:-------------:|:----------:|:---------------:|:---------:|
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
pip install pyai                # Basic
pip install pyai[openai]        # OpenAI
pip install pyai[azure]         # Azure + Azure AD
pip install pyai[all]           # Everything
```

### Hello World

```python
from pyai import ask

answer = ask("What is the capital of France?")
print(answer)  # Paris
```

### Configuration

```bash
# OpenAI
export OPENAI_API_KEY=sk-your-key

# Azure OpenAI (Azure AD - no key needed!)
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

---

## 💡 Design Philosophy

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': 'transparent', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#ffffff', 'lineColor': '#ffffff', 'secondaryColor': 'transparent', 'tertiaryColor': 'transparent', 'background': 'transparent', 'mainBkg': 'transparent', 'nodeBorder': '#ffffff', 'clusterBkg': 'transparent', 'clusterBorder': '#ffffff', 'titleColor': '#ffffff', 'edgeLabelBackground': 'transparent', 'nodeTextColor': '#ffffff'}}}%%
flowchart LR
    subgraph Philosophy["Design Principles"]
        P1["🎯 Simplicity First"]
        P2["🔋 Batteries Included"]
        P3["📐 Progressive Complexity"]
        P4["🧠 Intelligence as Infrastructure"]
        P5["🔧 Composability"]
    end
```

| Principle | Description |
|-----------|-------------|
| **Simplicity First** | One line should accomplish one task |
| **Batteries Included** | Everything you need, out of the box |
| **Progressive Complexity** | Start simple, scale up when needed |
| **Intelligence as Infrastructure** | AI is foundation, not feature |
| **Composability** | Small pieces combine into powerful systems |

---

## 👥 Community & Documentation

- 📖 **[Wiki Documentation](https://github.com/gitpavleenbali/PYAI/wiki)** — Comprehensive guides
- 🐛 **[Report Issues](https://github.com/gitpavleenbali/PYAI/issues)** — Bug reports
- 💡 **[Feature Requests](https://github.com/gitpavleenbali/PYAI/discussions)** — Ideas
- 🤝 **[Contributing Guide](./docs/CONTRIBUTING.md)** — Get involved

---

## 🔮 The PYAI Product Suite

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': 'transparent', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#ffffff', 'lineColor': '#ffffff', 'secondaryColor': 'transparent', 'tertiaryColor': 'transparent', 'background': 'transparent', 'mainBkg': 'transparent', 'nodeBorder': '#ffffff', 'clusterBkg': 'transparent', 'clusterBorder': '#ffffff', 'titleColor': '#ffffff', 'edgeLabelBackground': 'transparent', 'nodeTextColor': '#ffffff'}}}%%
flowchart LR
    subgraph Available["✅ Available Now"]
        PA["🤖 pyai<br/>Core SDK"]
    end
    
    subgraph Soon["🔜 Coming Soon"]
        PF["🔄 PyFlow<br/>Visual Workflows"]
        PV["👁️ PyVision<br/>Computer Vision"]
        PVO["🎤 PyVoice<br/>Speech & Audio"]
    end
    
    subgraph Future["🔮 Future"]
        PFAC["🏭 PyFactory<br/>Software Generation"]
        PM["🧠 PyMind<br/>Autonomous Reasoning"]
    end
    
    Available --> Soon --> Future
```

| Product | Purpose | Dimension | Status |
|---------|---------|-----------|--------|
| **🤖 pyai** | Core Intelligence SDK | All | ✅ Available |
| **🔄 PyFlow** | Visual AI Workflows | Orchestration | 🔜 Coming Soon |
| **👁️ PyVision** | Computer Vision | Cognition | 🔜 Coming Soon |
| **🎤 PyVoice** | Speech & Audio | Cognition | 🔜 Coming Soon |
| **🏭 PyFactory** | Software Generation | Creation | 🔮 Future |
| **🧠 PyMind** | Autonomous Reasoning | Creation | 🔮 Future |

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
