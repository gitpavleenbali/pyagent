# 📚 pyai API Reference

## Table of Contents
- [Configuration](#configuration)
- [One-Liner Functions](#one-liner-functions)
- [Modules](#modules)
- [Advanced Components](#advanced-components)

---

## Configuration

### `configure()`

Configure pyai globally. Call once at application startup.

```python
import pyai

pyai.configure(
    api_key="sk-...",           # API key (or use OPENAI_API_KEY env var)
    provider="openai",          # "openai" | "anthropic" | "azure"
    model="gpt-4o-mini",        # Default model
    azure_endpoint="...",       # Azure OpenAI endpoint (if using Azure)
    azure_deployment="...",     # Azure deployment name
    temperature=0.7,            # Default temperature
    max_tokens=2048,            # Default max tokens
)
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT` | Azure OpenAI deployment name |

---

## One-Liner Functions

### `ask()`

Ask any question, get an intelligent answer.

```python
from pyai import ask

# Basic
answer = ask("What is the capital of France?")

# With options
answer = ask("Explain quantum computing",
    detailed=True,    # Comprehensive answer
    concise=True,     # Brief answer (mutually exclusive with detailed)
    format="bullet",  # "bullet" | "numbered" | "markdown"
    creative=True,    # More creative response
    as_json=True,     # Return as dict
    model="gpt-4"     # Specific model
)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | `str` | required | The question to ask |
| `detailed` | `bool` | `False` | Get comprehensive answer |
| `concise` | `bool` | `False` | Get brief answer |
| `format` | `str` | `None` | Output format |
| `creative` | `bool` | `False` | Creative/varied response |
| `as_json` | `bool` | `False` | Return as dict |
| `model` | `str` | `None` | Override model |

**Returns:** `str` or `dict` (if `as_json=True`)

---

### `research()`

Deep research on any topic.

```python
from pyai import research

# Full research
result = research("quantum computing applications")
print(result.summary)       # Executive summary
print(result.key_points)    # List of key points
print(result.insights)      # Derived insights
print(result.sources)       # Referenced sources
print(result.confidence)    # Confidence score

# Quick summary only
summary = research("meditation benefits", quick=True)

# Insights only
insights = research("remote work future", as_insights=True)

# Focused research
result = research("climate change", focus="economic impact")
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `topic` | `str` | required | Topic to research |
| `quick` | `bool` | `False` | Return summary only |
| `as_insights` | `bool` | `False` | Return insights list only |
| `focus` | `str` | `None` | Focus area |
| `depth` | `str` | `"medium"` | "shallow" \| "medium" \| "deep" |

**Returns:** `ResearchResult`, `str`, or `List[str]`

---

### `summarize()`

Summarize text, files, or URLs.

```python
from pyai import summarize

# Text
summary = summarize("Long article text here...")

# File
summary = summarize("./report.pdf")
summary = summarize("./document.docx")

# URL
summary = summarize("https://example.com/article")

# With options
summary = summarize(content,
    length="short",      # "short" | "medium" | "long"
    focus="key findings",
    as_bullets=True
)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | `str` | required | Text, file path, or URL |
| `length` | `str` | `"medium"` | Summary length |
| `focus` | `str` | `None` | Focus area |
| `as_bullets` | `bool` | `False` | Return as bullet points |

**Returns:** `str`

---

### `extract()`

Extract structured data from text.

```python
from pyai import extract
from pydantic import BaseModel

# With Pydantic schema
class Person(BaseModel):
    name: str
    age: int
    email: str

person = extract(
    "John Doe is 30 years old. Contact: john@email.com",
    Person
)
print(person.name)  # "John Doe"
print(person.age)   # 30

# With dict schema
data = extract(text, {"name": str, "skills": list})
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | `str` | required | Text to extract from |
| `schema` | `type` | required | Pydantic model or dict |
| `strict` | `bool` | `False` | Strict validation |

**Returns:** Instance of schema type

---

### `generate()`

Generate content of various types.

```python
from pyai import generate

# Text content
blog = generate("blog post about AI trends", type="blog")

# Code
code = generate("fibonacci function", type="code", language="python")

# Email
email = generate("follow-up email after meeting", type="email")

# Documentation
docs = generate("API documentation for user service", type="docs")
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | required | What to generate |
| `type` | `str` | `"text"` | "text" \| "code" \| "email" \| "blog" \| "docs" |
| `length` | `str` | `"medium"` | "short" \| "medium" \| "long" |
| `style` | `str` | `None` | Style guidance |
| `language` | `str` | `None` | Programming language (for code) |

**Returns:** `str`

---

### `translate()`

Translate text between languages.

```python
from pyai import translate

# Basic
spanish = translate("Hello, how are you?", to="es")

# With options
formal = translate(text,
    to="de",
    from_lang="en",     # Auto-detected if omitted
    formal=True,        # Formal register
    preserve_formatting=True
)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | Text to translate |
| `to` | `str` | required | Target language code |
| `from_lang` | `str` | `None` | Source language (auto-detected) |
| `formal` | `bool` | `False` | Use formal register |
| `preserve_formatting` | `bool` | `True` | Keep formatting |

**Returns:** `str`

---

### `chat()`

Create an interactive chat session with memory.

```python
from pyai import chat

# Basic session
session = chat("You are a helpful assistant")
response1 = session.say("What is Python?")
response2 = session.say("How do I learn it?")  # Remembers context!

# With persona
session = chat(persona="teacher")
session("Explain machine learning")  # Shorthand for .say()

# Prebuilt personas
# "teacher", "advisor", "coder", "researcher", "writer", 
# "analyst", "critic", "creative", "editor"
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `system_message` | `str` | `None` | Custom system prompt |
| `persona` | `str` | `None` | Prebuilt persona |
| `model` | `str` | `None` | Override model |

**Returns:** `ChatSession`

**ChatSession Methods:**
| Method | Description |
|--------|-------------|
| `.say(message)` | Send message, get response |
| `(message)` | Shorthand for `.say()` |
| `.reset()` | Clear conversation history |
| `.history` | Get conversation history |

---

### `agent()`

Create a custom AI agent.

```python
from pyai import agent

# Custom agent
coder = agent("You are an expert Python developer")
result = coder("Write a function to parse JSON")

# Prebuilt persona
researcher = agent(persona="researcher")
result = researcher("Research quantum computing")

# Named agent with memory
assistant = agent(
    "You are a helpful assistant",
    name="Alex",
    memory=True
)
assistant("My name is John")
assistant("What's my name?")  # Returns "John"

# Available personas:
# "coder", "researcher", "writer", "analyst", "teacher",
# "advisor", "critic", "creative", "editor", "python_expert"
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `system_message` | `str` | `None` | Custom system prompt |
| `persona` | `str` | `None` | Prebuilt persona |
| `name` | `str` | `None` | Agent name |
| `model` | `str` | `None` | Override model |
| `memory` | `bool` | `True` | Enable memory |

**Returns:** `Agent`

---

## Modules

### `rag` Module

RAG (Retrieval-Augmented Generation) operations.

```python
from pyai import rag

# One-shot RAG
answer = rag.ask("./docs/paper.pdf", "What is the conclusion?")

# Index documents for multiple queries
docs = rag.index(["doc1.txt", "doc2.pdf", "./folder"])
answer1 = docs.ask("What is the main finding?")
answer2 = docs.ask("What methodology was used?")

# From URL
answer = rag.from_url("https://example.com", "Summarize this")

# From raw text
answer = rag.from_text(long_text, "What are the key points?")

# Index options
docs = rag.index(sources,
    chunk_size=500,   # Characters per chunk
    overlap=50        # Overlap between chunks
)
```

**Functions:**
| Function | Description |
|----------|-------------|
| `rag.index(sources, ...)` | Index documents |
| `rag.ask(source, question)` | One-shot RAG query |
| `rag.from_url(url, question)` | RAG from URL |
| `rag.from_text(text, question)` | RAG from text |

---

### `fetch` Module

Real-time data fetching.

```python
from pyai import fetch

# Weather
weather = fetch.weather("Tokyo")
print(weather.temperature)  # 22.5
print(weather.conditions)   # "Partly Cloudy"
print(weather.humidity)     # 65
print(weather.wind_speed)   # 12.3

# News
articles = fetch.news("artificial intelligence", limit=5)
for article in articles:
    print(article.title)
    print(article.source)
    print(article.url)

# Stocks
stock = fetch.stock("AAPL")
print(stock.price)          # 175.50
print(stock.change)         # +2.30
print(stock.change_percent) # +1.33
print(stock.volume)         # 52000000

# Crypto
btc = fetch.crypto("BTC")
print(btc.price)            # 45000.00
print(btc.change_24h)       # +3.5%
print(btc.market_cap)       # 850000000000

# Facts
facts = fetch.facts("black holes", count=3)
for fact in facts:
    print(f"- {fact}")
```

---

### `analyze` Module

Data and text analysis.

```python
from pyai import analyze
import pandas as pd

# Data analysis
df = pd.DataFrame(...)
result = analyze.data(df, goal="find anomalies")
print(result.summary)
print(result.insights)
print(result.statistics)
print(result.recommendations)

# Text analysis
analysis = analyze.text(article, aspects=["tone", "complexity"])

# Sentiment
sentiment = analyze.sentiment("I love this product!")
print(sentiment.sentiment)  # "positive"
print(sentiment.score)      # 0.95
print(sentiment.aspects)    # {"product": "positive"}

# Compare items
comparison = analyze.compare(
    "Python", "JavaScript", "Rust",
    criteria=["performance", "ease of use", "ecosystem"]
)
```

---

### `code` Module

Code generation and analysis.

```python
from pyai import code

# Write code
python_code = code.write("function to calculate fibonacci")
js_code = code.write("react component for login form", language="javascript")

# Review code
review = code.review(my_code)
print(review.score)           # 8/10
print(review.issues)          # ["unused variable", ...]
print(review.suggestions)     # ["consider using list comprehension", ...]
print(review.security_concerns)

# Debug errors
fix = code.debug("TypeError: cannot unpack...", code=buggy_code)

# Explain code
explanation = code.explain(complex_function, level="beginner")

# Refactor
improved = code.refactor(old_code, goal="performance")
improved = code.refactor(old_code, goal="readability")
improved = code.refactor(old_code, goal="type-safety")

# Convert between languages
js_code = code.convert(python_code, from_lang="python", to_lang="javascript")
```

---

## Advanced Components

For advanced use cases, access the underlying components:

```python
from pyai.core import Agent, Memory, ConversationMemory, VectorMemory
from pyai.core import LLMProvider, OpenAIProvider, AnthropicProvider
from pyai.instructions import Instruction, SystemPrompt, Persona
from pyai.skills import Skill, ToolSkill, ActionSkill, SkillRegistry
from pyai.blueprint import Blueprint, Workflow, Pipeline
```

See [ARCHITECTURE.md](./ARCHITECTURE.md) for advanced usage patterns.

---

## Error Handling

```python
from pyai import ask
from pyai.exceptions import (
    pyaiError,      # Base exception
    ConfigError,       # Configuration issues
    LLMError,          # LLM provider errors
    RateLimitError,    # Rate limiting
    TokenLimitError,   # Token limit exceeded
)

try:
    answer = ask("...", model="nonexistent-model")
except ConfigError as e:
    print(f"Configuration issue: {e}")
except LLMError as e:
    print(f"LLM error: {e}")
except pyaiError as e:
    print(f"General error: {e}")
```

---

## Type Safety

pyai is fully typed. Install type stubs are included:

```python
# Full IDE support for:
from pyai import ask, agent, rag, fetch, code, chat

# Hover documentation works
# Autocomplete works
# Type checking works
```

---

*For architectural details, see [ARCHITECTURE.md](./ARCHITECTURE.md)*
