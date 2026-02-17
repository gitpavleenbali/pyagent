# 🚀 pyai Quick Start Guide

Get up and running with pyai in under 5 minutes.

## Installation

```bash
pip install pyai
```

## Setup

### Option 1: Environment Variable (Recommended)

```bash
# Linux/macOS
export OPENAI_API_KEY=sk-your-key-here

# Windows PowerShell
$env:OPENAI_API_KEY = "sk-your-key-here"

# Windows CMD
set OPENAI_API_KEY=sk-your-key-here
```

### Option 2: Programmatic Configuration

```python
import pyai

pyai.configure(api_key="sk-your-key-here")
```

### Option 3: Azure OpenAI

```python
import pyai

pyai.configure(
    provider="azure",
    azure_endpoint="https://YOUR-RESOURCE.openai.azure.com/",
    azure_deployment="gpt-4o-mini",
    api_key="your-azure-api-key"
)
```

---

## Your First 5 Lines

```python
from pyai import ask, agent, rag

# 1. Ask anything
answer = ask("What is machine learning?")
print(answer)

# 2. Create an expert agent
coder = agent(persona="coder")
code = coder("Write a Python function to reverse a string")
print(code)

# 3. RAG in 2 lines
docs = rag.index(["My company sells AI solutions.", "We are based in Seattle."])
answer = docs.ask("Where is the company located?")
print(answer)  # "Seattle"
```

---

## 5 Minute Tutorial

### 1. Simple Q&A

```python
from pyai import ask

# Basic question
answer = ask("What is the capital of Japan?")
# "Tokyo"

# Detailed answer
explanation = ask("Explain photosynthesis", detailed=True)

# Concise answer
brief = ask("What is DNA?", concise=True)

# Formatted output
tips = ask("Give me 5 Python tips", format="bullet")

# JSON output
profile = ask("Generate a sample user profile", as_json=True)
# {"name": "John Doe", "age": 30, "email": "john@example.com"}
```

### 2. Custom Agents

```python
from pyai import agent

# Create a Python expert
python_expert = agent(persona="python_expert")
result = python_expert("How do I use list comprehensions?")

# Create a custom agent
math_tutor = agent("You are a patient math tutor who explains concepts simply")
result = math_tutor("What is calculus?")

# Agent with memory
assistant = agent("You are helpful", name="Alex", memory=True)
assistant("My favorite color is blue")
assistant("What's my favorite color?")  # "Blue!"
```

### 3. RAG (Retrieval-Augmented Generation)

```python
from pyai import rag

# From files
answer = rag.ask("./documents/report.pdf", "What is the main conclusion?")

# Index multiple documents
docs = rag.index([
    "./research.pdf",
    "./notes.txt",
    "./data/"  # Entire folder
])
answer1 = docs.ask("What methodology was used?")
answer2 = docs.ask("What are the limitations?")

# From URL
answer = rag.from_url("https://example.com/article", "Summarize this")

# From text
long_text = "..."
answer = rag.from_text(long_text, "What are the key points?")
```

### 4. Real-Time Data

```python
from pyai import fetch

# Weather
weather = fetch.weather("New York")
print(f"{weather.temperature}°C, {weather.conditions}")

# News
news = fetch.news("artificial intelligence")
for article in news[:3]:
    print(f"- {article.title}")

# Stocks
stock = fetch.stock("AAPL")
print(f"Apple: ${stock.price} ({stock.change_percent}%)")

# Crypto
btc = fetch.crypto("BTC")
print(f"Bitcoin: ${btc.price}")
```

### 5. Code Operations

```python
from pyai import code

# Generate code
func = code.write("function to calculate fibonacci numbers")

# Review code
my_code = """
def calculate(x):
    result = x * 2
    return result
"""
review = code.review(my_code)
print(f"Score: {review.score}/10")
print(f"Issues: {review.issues}")

# Debug errors
fix = code.debug("TypeError: 'NoneType' object is not subscriptable")

# Convert languages
js_code = code.convert(python_code, from_lang="python", to_lang="javascript")
```

---

## Common Use Cases

### Research Assistant

```python
from pyai import research

result = research("quantum computing applications")
print("Summary:", result.summary)
print("Key Points:", result.key_points)
print("Insights:", result.insights)
```

### Document Summarizer

```python
from pyai import summarize

# Summarize anything
summary = summarize("./research_paper.pdf")
summary = summarize("https://example.com/article")
summary = summarize("Long text here...", length="short")
```

### Data Extractor

```python
from pyai import extract
from pydantic import BaseModel

class Invoice(BaseModel):
    vendor: str
    amount: float
    date: str
    items: list

invoice = extract(invoice_text, Invoice)
print(f"Vendor: {invoice.vendor}, Amount: ${invoice.amount}")
```

### Chat Session

```python
from pyai import chat

session = chat(persona="teacher")
session.say("What is machine learning?")
session.say("Give me an example")  # Remembers context
session.say("How can I learn more?")  # Still has full context
```

### Multi-Language Support

```python
from pyai import translate

# English to Spanish
spanish = translate("Hello, how are you?", to="es")
# "Hola, ¿cómo estás?"

# With formal register
formal_german = translate("Please help me", to="de", formal=True)
```

---

## What's Next?

- 📖 [Full API Reference](./API_REFERENCE.md)
- 🏗️ [Architecture Guide](./ARCHITECTURE.md)
- 🤝 [Contributing Guide](./CONTRIBUTING.md)
- 📝 [Changelog](./CHANGELOG.md)

---

## Comparison

Task | Other Frameworks | pyai
-----|-----------------|--------
Simple Q&A | 10+ lines | 1 line
RAG | 20+ lines | 2 lines
Custom Agent | 25+ lines | 1 line
Weather Fetch | 15+ lines | 1 line
Code Review | Custom implementation | 1 line

**pyai: The Pandas of AI** 🐼🤖
