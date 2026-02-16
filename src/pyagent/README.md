# 🐼🤖 PyAgent - The Pandas of AI Agents

**Build AI-powered applications in 3 lines or less.**

PyAgent is a revolutionary Python library that brings pandas-like simplicity to AI agent development. No boilerplate. No configuration hell. Just results.

[![PyPI version](https://badge.fury.io/py/pyagent.svg)](https://badge.fury.io/py/pyagent)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
- **🐼 Pandas-like API** - if you know pandas, you know pyagent
- **⚙️ Zero configuration** - sensible defaults that just work
- **🔧 Power when needed** - full access to low-level components

## 📦 Installation

```bash
pip install pyagent

# With optional dependencies
pip install pyagent[openai]      # For OpenAI models
pip install pyagent[anthropic]   # For Anthropic models
pip install pyagent[all]         # Everything
```

## 🚀 Quick Start

### Ask anything in one line

```python
from pyagent import ask

answer = ask("What is the capital of France?")
# 'Paris'

answer = ask("Explain quantum computing", detailed=True)
# 'Quantum computing is a type of computation that harnesses...'

answer = ask("List 5 Python tips", format="bullet")
# '• Use list comprehensions...\n• Leverage f-strings...'
```

### RAG in 2 lines

```python
from pyagent import rag

# Index and query
docs = rag.index("./documents")
answer = docs.ask("What is the main conclusion?")

# Or even simpler - one line!
answer = rag.ask("./research_paper.pdf", "What methodology was used?")
```

### Research any topic

```python
from pyagent import research

result = research("AI trends 2024")
print(result.summary)
print(result.key_points)
print(result.insights)
```

### Fetch real-time data

```python
from pyagent import fetch

# Weather
weather = fetch.weather("New York")
print(f"{weather.temperature}°C, {weather.conditions}")

# News
news = fetch.news("artificial intelligence")
for article in news:
    print(article.title)

# Stocks
stock = fetch.stock("AAPL")
print(f"${stock.price} ({stock.change_percent}%)")
```

### Generate content

```python
from pyagent import generate

# Code
code = generate("fibonacci function", type="code")

# Email
email = generate("welcome email for new users", type="email")

# Article
article = generate("blog post about AI", type="article")
```

### Create custom agents

```python
from pyagent import agent

# Custom agent
coder = agent("You are an expert Python developer")
result = coder("Write a REST API for a todo app")

# Prebuilt personas
researcher = agent(persona="researcher")
findings = researcher("Research the latest in quantum computing")

# With memory (remembers conversation)
assistant = agent("You are a helpful data analyst")
assistant("Load the sales data")
assistant("What are the top trends?")  # Remembers context
```

### Chat sessions

```python
from pyagent import chat

session = chat(persona="teacher")
session("Explain machine learning")
session("What about deep learning?")  # Continues conversation
session("Give me an example")          # Still has context
```

### Summarize anything

```python
from pyagent import summarize

# From text
summary = summarize("Long article text here...")

# From file
summary = summarize("./report.pdf")

# From URL
summary = summarize("https://example.com/article")

# Options
summary = summarize(text, length="short")
summary = summarize(text, bullet_points=True)
summary = summarize(text, style="executive")
```

### Extract structured data

```python
from pyagent import extract

text = "John is 30 years old and lives in New York"

# Extract specific fields
data = extract(text, ["name", "age", "city"])
# {"name": "John", "age": 30, "city": "New York"}

# Natural language extraction
emails = extract(document, "all email addresses")
# ["john@email.com", "jane@company.com"]
```

### Translate

```python
from pyagent import translate

spanish = translate("Hello, how are you?", to="spanish")
# "¡Hola, ¿cómo estás?"

japanese = translate("Welcome", to="japanese", formal=True)
# "ようこそ"
```

### Analyze data

```python
from pyagent import analyze

# Analyze any data
insights = analyze.data(sales_data)
print(insights.summary)
print(insights.recommendations)

# Sentiment analysis
sentiment = analyze.sentiment("I love this product!")
# {"sentiment": "positive", "confidence": 0.95}
```

### Code operations

```python
from pyagent import code

# Write code
python_code = code.write("function to parse JSON files")

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

## ⚙️ Configuration

PyAgent works out of the box with environment variables:

```bash
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=...
# or 
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://...
```

Or configure programmatically:

```python
import pyagent

pyagent.configure(
    api_key="sk-...",
    model="gpt-4o",
    temperature=0.7
)
```

## 🔧 Power User Mode

When you need full control, access the complete low-level API:

```python
from pyagent import Agent, Blueprint, Workflow, Memory

# Full control over agent
agent = Agent(
    instructions=Instruction("You are a specialized assistant"),
    skills=[CustomSkill(), WebSkill()],
    memory=VectorMemory(provider="chromadb")
)

# Complex workflows
workflow = Workflow()
workflow.add_step("research", ResearchAgent())
workflow.add_step("analyze", AnalysisAgent())
workflow.add_step("report", ReportAgent())
result = workflow.run("Analyze market trends")
```

## 📊 Comparison with Other Frameworks

### LangChain (RAG Example)

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
```

```python
# PyAgent: 2 lines
from pyagent import rag
answer = rag.ask("./docs", "What is the conclusion?")
```

### CrewAI (Research Agent Example)

```python
# CrewAI: 25+ lines (plus YAML config files)
from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="Senior Researcher",
    goal="Research the topic thoroughly",
    backstory="You are an expert researcher...",
    verbose=True
)
task = Task(
    description="Research AI trends",
    expected_output="Comprehensive report",
    agent=researcher
)
crew = Crew(
    agents=[researcher],
    tasks=[task],
    process=Process.sequential
)
result = crew.kickoff()
```

```python
# PyAgent: 1 line
from pyagent import research
result = research("AI trends")
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**PyAgent** - *Because AI development should be as simple as `import pandas as pd`*
