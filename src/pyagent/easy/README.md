# PyAgent Easy Module

**One-Liner Intelligence for Python Developers**

The `easy` module is the heart of PyAgent - providing simple, single-function interfaces to powerful AI capabilities. If you can write `print()`, you can use PyAgent.

## Philosophy

```python
# Instead of this (traditional approach):
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is AI?"}]
)
answer = response.choices[0].message.content

# Just do this:
from pyagent import ask
answer = ask("What is AI?")
```

## Available Functions

### Core Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `ask()` | Get answers to questions | `ask("What is machine learning?")` |
| `chat()` | Multi-turn conversations | `chat("Hi!", history=messages)` |
| `agent()` | Create intelligent agents | `agent("You are helpful", name="Bot")` |

### Content Generation

| Function | Purpose | Example |
|----------|---------|---------|
| `generate()` | Generate text content | `generate("blog post about AI")` |
| `summarize()` | Summarize long text | `summarize(article, style="bullet")` |
| `translate()` | Translate between languages | `translate("Hello", to="Spanish")` |
| `code()` | Generate/explain code | `code("sort algorithm in Python")` |

### Data Processing

| Function | Purpose | Example |
|----------|---------|---------|
| `extract()` | Extract structured data | `extract(text, schema={"name": str})` |
| `analyze()` | Analyze data/text | `analyze(data, goal="find trends")` |
| `fetch()` | Fetch and process URLs | `fetch("https://example.com")` |

### Advanced Features

| Function | Purpose | Example |
|----------|---------|---------|
| `research()` | Deep research with sources | `research("quantum computing")` |
| `rag()` | Retrieval Augmented Generation | `rag("question", documents=docs)` |
| `handoff()` | Transfer between agents | `handoff(agent1, agent2, task)` |
| `mcp()` | Model Context Protocol | `mcp.connect("server")` |
| `guardrails()` | Safety and validation | `guardrails.validate(content)` |
| `trace()` | Execution tracing | `trace.start("session")` |

## Detailed Usage

### ask() - The Simplest AI Call

```python
from pyagent import ask

# Basic usage
answer = ask("What is the capital of France?")

# With model selection
answer = ask("Complex question...", model="gpt-4")

# With system context
answer = ask(
    "How do I fix this?",
    system="You are a senior Python developer"
)
```

### agent() - Create Intelligent Agents

```python
from pyagent import agent

# Simple agent
assistant = agent("You are a helpful assistant")
response = assistant("Help me plan my day")

# Agent with memory
chatbot = agent(
    "You are a friendly chatbot",
    name="Buddy",
    memory=True
)
chatbot("My name is Alice")
chatbot("What's my name?")  # Remembers: "Alice"

# Agent with skills
from pyagent.skills import web_search

researcher = agent(
    "You are a researcher with web access",
    skills=[web_search]
)
```

### chat() - Conversational AI

```python
from pyagent import chat

# Single turn
response, history = chat("Hello!")

# Multi-turn with history
response, history = chat("What about Python?", history=history)

# With persona
response, _ = chat(
    "Tell me a joke",
    persona="comedian",
    style="witty"
)
```

### research() - Deep Investigation

```python
from pyagent import research

# Basic research
report = research("impacts of climate change")

# With specific depth
report = research(
    "quantum computing applications",
    depth="comprehensive",  # quick, standard, comprehensive
    sources=5
)

# Output includes:
# - Executive summary
# - Key findings
# - Detailed analysis
# - Sources with citations
```

### extract() - Structured Data Extraction

```python
from pyagent import extract

text = "John Smith, CEO of TechCorp, announced $10M funding"

# Extract with schema
data = extract(text, schema={
    "name": str,
    "title": str,
    "company": str,
    "amount": str
})
# Returns: {"name": "John Smith", "title": "CEO", ...}

# Extract list of items
emails = extract(document, extract_type="emails")
dates = extract(document, extract_type="dates")
```

### rag() - Knowledge-Grounded Responses

```python
from pyagent import rag

# With documents
response = rag(
    "What is our refund policy?",
    documents=[
        "Our refund policy allows returns within 30 days...",
        "Refunds are processed within 5 business days..."
    ]
)

# With vector store
from pyagent.integrations import vector_db
store = vector_db.connect("chroma", path="./my_docs")

response = rag(
    "What are the system requirements?",
    store=store
)
```

### code() - Code Generation & Explanation

```python
from pyagent import code

# Generate code
result = code("quicksort in Python", task="generate")

# Explain code
explanation = code(my_function_code, task="explain")

# Review code
review = code(my_code, task="review")

# Fix code
fixed = code(buggy_code, task="fix", error=error_message)
```

### guardrails() - Safety & Validation

```python
from pyagent import guardrails

# Create guardrails
rails = guardrails()

# Add input validation
rails.add_input_rule("block_pii", detect_pii, action="block")
rails.add_input_rule("profanity", detect_profanity, action="warn")

# Add output validation
rails.add_output_rule("factual", fact_check, action="flag")

# Use with agent
safe_agent = rails.wrap(my_agent)
```

### handoff() - Agent Coordination

```python
from pyagent import agent, handoff

# Create specialized agents
sales = agent("You are a sales expert")
support = agent("You are technical support")

# Handoff when needed
result = handoff(
    from_agent=sales,
    to_agent=support,
    context="Customer needs technical help",
    reason="Technical question beyond sales scope"
)
```

## Configuration

### Global Settings

```python
from pyagent.easy import config

# Set default model
config.set_model("gpt-4o")

# Set API key
config.set_api_key("sk-...")

# Use Azure OpenAI
config.use_azure(
    endpoint="https://your-resource.openai.azure.com",
    api_version="2024-02-15-preview"
)

# Enable tracing
config.enable_tracing(True)
```

## Module Structure

```
easy/
├── __init__.py          # Main exports
├── ask.py               # ask() function
├── chat.py              # chat() function  
├── agent_factory.py     # agent() function
├── generate.py          # generate() function
├── summarize.py         # summarize() function
├── translate.py         # translate() function
├── extract.py           # extract() function
├── analyze.py           # analyze() function
├── research.py          # research() function
├── code.py              # code() function
├── fetch.py             # fetch() function
├── rag.py               # rag() function
├── handoff.py           # handoff() function
├── mcp.py               # MCP protocol
├── guardrails.py        # Safety features
├── trace.py             # Execution tracing
├── config.py            # Configuration
└── llm_interface.py     # LLM backend
```

## Best Practices

1. **Start Simple**: Use `ask()` first, graduate to `agent()` when needed
2. **Use Memory Wisely**: Enable memory only for conversational agents
3. **Configure Once**: Set up config at app startup
4. **Handle Errors**: Wrap calls in try/except for production
5. **Monitor Usage**: Use `trace()` in production

## See Also

- [PyAgent Quickstart](../../docs/QUICKSTART.md)
- [API Reference](../../docs/API_REFERENCE.md)
- [Examples](../../examples/)
