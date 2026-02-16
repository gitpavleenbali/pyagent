# 🐼🤖 PyAgent - The Pandas of AI Agents

**Build AI-powered applications in 3 lines or less.**

PyAgent is a revolutionary Python library that brings pandas-like simplicity to AI agent development. No boilerplate. No configuration hell. Just results.

[![PyPI version](https://img.shields.io/badge/pypi-v0.4.0-blue)](https://pypi.org/project/pyagent/)
[![Python](https://img.shields.io/badge/python-3.10+-green)](https://python.org/)
[![Tests](https://img.shields.io/badge/tests-671%20passing-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **What pandas did for data, PyAgent does for intelligence.**

---

## ✨ Why PyAgent?

| Framework | Lines for RAG | Lines for Weather Agent | Lines for Research |
|-----------|--------------|------------------------|-------------------|
| LangChain | 15+ | 20+ | 25+ |
| LlamaIndex | 10+ | 15+ | 20+ |
| CrewAI | 30+ | 25+ | 35+ |
| **PyAgent** | **2** | **1** | **1** |

### What makes it legendary:

- **🚀 One-liner operations** for common AI tasks
- **📦 Batteries included** - prebuilt agents ready to use  
- **🐼 Pandas-like API** - if you know pandas, you know PyAgent
- **⚙️ Zero configuration** - sensible defaults that just work
- **🔧 Power when needed** - full access to low-level components
- **🏢 Enterprise ready** - Azure AD auth, sessions, evaluation tools

---

## 📦 Installation

```bash
pip install pyagent

# With optional dependencies
pip install pyagent[openai]      # For OpenAI models
pip install pyagent[anthropic]   # For Anthropic models
pip install pyagent[azure]       # For Azure OpenAI + Azure AD
pip install pyagent[all]         # Everything
```

---

## 🚀 Quick Start

### One-Liner API

```python
from pyagent import ask, summarize, research

# Ask anything
answer = ask("What is the capital of France?")  # 'Paris'

# Summarize documents
summary = summarize("./report.pdf", length="short")

# Research any topic
result = research("AI trends 2024")
print(result.summary)
print(result.key_points)
```

### RAG in 2 Lines

```python
from pyagent import rag

docs = rag.index("./documents")
answer = docs.ask("What is the main conclusion?")
```

### Generate Content

```python
from pyagent import generate

code = generate("fibonacci function", type="code")
email = generate("welcome email for new users", type="email")
article = generate("blog post about AI", type="article")
```

---

## 🤖 Agent Framework

When you need more control, use the full Agent API:

```python
from pyagent import Agent, Runner
from pyagent.skills import tool

@tool(description="Get weather for a city")
async def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny, 72°F"

agent = Agent(
    name="WeatherBot",
    instructions="Help users with weather information.",
    tools=[get_weather]
)

result = Runner.run_sync(agent, "What's the weather in Tokyo?")
print(result.final_output)
```

### Multi-Agent Systems

```python
from pyagent import Agent
from pyagent.blueprint import Workflow, Step

# Specialized agents
researcher = Agent(name="Researcher", instructions="Find information.")
writer = Agent(name="Writer", instructions="Write engaging content.")
editor = Agent(name="Editor", instructions="Review and improve.")

# Chain them
workflow = (Workflow("ContentPipeline")
    .add_step(Step("research", researcher))
    .add_step(Step("write", writer))
    .add_step(Step("edit", editor))
    .build())
```

### Agent Handoffs

```python
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

---

## 🔧 More Powerful Features

### Fetch Real-Time Data

```python
from pyagent import fetch

weather = fetch.weather("New York")
news = fetch.news("artificial intelligence")
stock = fetch.stock("AAPL")
```

### Extract Structured Data

```python
from pyagent import extract

text = "John is 30 years old and lives in New York"
data = extract(text, ["name", "age", "city"])
# {"name": "John", "age": 30, "city": "New York"}
```

### Analyze & Code

```python
from pyagent import analyze, code

# Sentiment analysis
sentiment = analyze.sentiment("I love this product!")

# Write, review, debug code
python_code = code.write("REST API for todo app")
review = code.review(my_code)
solution = code.debug("TypeError: cannot unpack...")
```

---

## ⚙️ Configuration

Works out of the box with environment variables:

```bash
export OPENAI_API_KEY=sk-...
# or for Azure
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

Or configure programmatically:

```python
import pyagent
pyagent.configure(api_key="sk-...", model="gpt-4o", temperature=0.7)
```

---

## 🏢 Enterprise Features

| Feature | Description |
|---------|-------------|
| **Model Agnostic** | OpenAI, Azure OpenAI, Anthropic, Ollama, custom providers |
| **Azure AD Auth** | No API keys needed - uses `az login` credentials |
| **Sessions** | SQLite and Redis session persistence |
| **Kernel Registry** | MS Semantic Kernel-style service management |
| **Streaming** | Real-time streaming responses |
| **Voice Support** | Voice input/output with OpenAI Realtime API |
| **Evaluation** | Built-in agent evaluation framework |
| **Tracing** | Full execution tracing for debugging |

### Azure OpenAI with Azure AD

```python
import os
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource.openai.azure.com/"
os.environ["AZURE_OPENAI_DEPLOYMENT"] = "gpt-4o-mini"

from pyagent import ask
# Automatically uses your Azure login - no API key needed!
answer = ask("Hello!")
```

---

## 📊 Real Comparison: PyAgent vs Others

### RAG Example

```python
# LangChain: 15+ lines
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

loader = DirectoryLoader('./docs')
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)
qa = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=vectorstore.as_retriever())
result = qa.run("What is the conclusion?")

# PyAgent: 2 lines
from pyagent import rag
answer = rag.ask("./docs", "What is the conclusion?")
```

### Research Agent Example

```python
# CrewAI: 25+ lines
from crewai import Agent, Task, Crew, Process
researcher = Agent(role="Senior Researcher", goal="Research thoroughly", backstory="...", verbose=True)
task = Task(description="Research AI trends", expected_output="Report", agent=researcher)
crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential)
result = crew.kickoff()

# PyAgent: 1 line
from pyagent import research
result = research("AI trends")
```

---

## 📚 Documentation

- [Getting Started Guide](docs/QUICKSTART.md)
- [API Reference](docs/API_REFERENCE.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Azure Setup Guide](docs/AZURE_SETUP.md)
- [Examples](examples/)

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

---

## 🙏 Acknowledgements

PyAgent builds on the excellent work of the open-source community:

- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) - Runner pattern
- [Microsoft Semantic Kernel](https://github.com/microsoft/semantic-kernel) - Plugin architecture
- [Google ADK](https://github.com/google/adk-python) - Agent configuration
- [Strands Agents](https://github.com/strands-agents/sdk-python) - Tool discovery
- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) - Token utilities

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>PyAgent</strong> - <em>Because AI development should be as simple as <code>import pandas as pd</code></em>
</p>
