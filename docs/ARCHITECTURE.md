# 🏗️ pyai Architecture Blueprint

## Vision Statement

**pyai** aims to be the **pandas of AI** - making AI development as simple as data manipulation. Just as pandas revolutionized data analysis by making complex operations one-liners, pyai revolutionizes AI development.

---

## Architectural Philosophy

### The 3-Dimensional Library Concept

Traditional libraries are **2-dimensional**:
- Function → Result
- Input → Output

pyai is **3-dimensional**:
- Function → **Context** → **Intelligence** → Result
- Single call embeds: configuration, memory, reasoning, output formatting

```
┌─────────────────────────────────────────────────────────────────┐
│                    pyai 3D ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    USER LAYER (Dimension 1 - Surface)                           │
│    ═══════════════════════════════════                          │
│    • ask("question") → answer                                   │
│    • research("topic") → insights                               │
│    • agent("persona") → intelligent assistant                   │
│                                                                  │
│    ─────────────────────────────────────────────────────────    │
│                                                                  │
│    INTELLIGENCE LAYER (Dimension 2 - Depth)                     │
│    ═══════════════════════════════════════                       │
│    • Auto-configuration (zero-config)                           │
│    • Smart defaults (model selection)                           │
│    • Memory management (conversation context)                   │
│    • RAG indexing (document understanding)                      │
│                                                                  │
│    ─────────────────────────────────────────────────────────    │
│                                                                  │
│    FOUNDATION LAYER (Dimension 3 - Infrastructure)             │
│    ═══════════════════════════════════════════════              │
│    • Multi-provider LLM support (OpenAI, Anthropic, Azure)     │
│    • Skill system (extensible capabilities)                     │
│    • Blueprint patterns (complex workflows)                     │
│    • Memory stores (conversation, vector, hybrid)              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Principles

### 1. **ZERO FRICTION**
```python
# Bad: Other frameworks require 10+ lines
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
template = PromptTemplate(input_variables=["question"], template="{question}")
chain = LLMChain(llm=llm, prompt=template)
result = chain.run("What is AI?")

# Good: pyai - one line
from pyai import ask
answer = ask("What is AI?")
```

### 2. **SENSIBLE DEFAULTS**
- Auto-detects API keys from environment
- Uses optimal model for each task type
- Manages memory automatically
- Handles errors gracefully

### 3. **PROGRESSIVE COMPLEXITY**
```python
# Level 1: One-liner (80% of use cases)
answer = ask("What is AI?")

# Level 2: Options (15% of use cases)
answer = ask("What is AI?", detailed=True, model="gpt-4")

# Level 3: Full control (5% of use cases)
from pyai import Agent, Memory, SystemPrompt
agent = Agent(
    llm=OpenAIProvider(model="gpt-4"),
    memory=VectorMemory(size=1000),
    system_prompt=SystemPrompt("You are an expert...")
)
```

---

## Module Architecture

```
pyai/
├── __init__.py          # Main entry point with lazy imports
├── __init__.pyi         # Type stubs for IDE support
├── py.typed             # PEP 561 marker
│
├── easy/                # THE REVOLUTIONARY SIMPLE API
│   ├── ask.py           # ask() - Question answering
│   ├── research.py      # research() - Deep research
│   ├── summarize.py     # summarize() - Text summarization
│   ├── extract.py       # extract() - Data extraction
│   ├── generate.py      # generate() - Content generation
│   ├── translate.py     # translate() - Translation
│   ├── chat.py          # chat() - Interactive sessions
│   ├── agent_factory.py # agent() - Custom agents
│   ├── rag.py           # rag module - RAG operations
│   ├── fetch.py         # fetch module - Real-time data
│   ├── analyze.py       # analyze module - Analysis
│   ├── code.py          # code module - Code operations
│   ├── config.py        # Configuration management
│   └── llm_interface.py # Unified LLM interface
│
├── core/                # FOUNDATION COMPONENTS
│   ├── agent.py         # Base Agent class
│   ├── base.py          # Abstract base classes
│   ├── llm.py           # LLM providers
│   └── memory.py        # Memory implementations
│
├── instructions/        # PROMPT ENGINEERING
│   ├── instruction.py   # Base instruction
│   ├── system_prompt.py # System prompts
│   ├── context.py       # Context injection
│   ├── persona.py       # Agent personas
│   └── guidelines.py    # Behavioral guidelines
│
├── skills/              # CAPABILITIES
│   ├── skill.py         # Base skill class
│   ├── tool_skill.py    # Function-as-tool
│   ├── action_skill.py  # Discrete actions
│   ├── registry.py      # Skill registry
│   └── builtin.py       # Built-in skills
│
└── blueprint/           # COMPLEX WORKFLOWS
    ├── blueprint.py     # Workflow blueprints
    ├── orchestrator.py  # Multi-agent orchestration
    ├── patterns.py      # Common patterns
    ├── pipeline.py      # Sequential pipelines
    └── workflow.py      # Workflow definitions
```

---

## Data Flow Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER INPUT                                │
│                    ask("What is Python?")                        │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      CONFIGURATION LAYER                          │
│  • Auto-detect API key from environment                          │
│  • Select optimal model (gpt-4o-mini default)                    │
│  • Apply sensible defaults                                        │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                     PROMPT CONSTRUCTION                           │
│  • Build system message                                          │
│  • Apply formatting rules (concise, detailed, etc.)              │
│  • Inject context if available                                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      LLM INTERFACE                                │
│  ┌─────────┐   ┌──────────┐   ┌─────────┐                       │
│  │ OpenAI  │   │ Anthropic │   │  Azure  │                       │
│  └────┬────┘   └─────┬────┘   └────┬────┘                       │
│       └──────────────┼─────────────┘                             │
│                      ▼                                            │
│            Unified Response                                       │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    OUTPUT PROCESSING                              │
│  • Parse response                                                │
│  • Format as requested (JSON, bullet, etc.)                      │
│  • Apply post-processing                                         │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                         RESULT                                    │
│         "Python is a high-level programming language..."         │
└──────────────────────────────────────────────────────────────────┘
```

---

## Comparison with Competitors

| Feature | LangChain | LlamaIndex | AutoGen | CrewAI | **pyai** |
|---------|-----------|------------|---------|--------|-------------|
| Lines for simple Q&A | 10+ | 8+ | 15+ | 12+ | **1** |
| Lines for RAG | 20+ | 15+ | 25+ | 20+ | **2** |
| Zero-config | ❌ | ❌ | ❌ | ❌ | **✅** |
| Type hints | Partial | Partial | Partial | Partial | **Full** |
| Memory auto-manage | ❌ | ❌ | ❌ | ❌ | **✅** |
| Learning curve | Steep | Moderate | Steep | Moderate | **Flat** |

---

## Design Patterns Used

### 1. **Lazy Loading Pattern**
```python
# __init__.py uses __getattr__ for lazy imports
def __getattr__(name):
    if name == "ask":
        from pyai.easy.ask import ask
        return ask
```

### 2. **Factory Pattern**
```python
# agent() is a factory that creates Agent instances
def agent(persona="coder"):
    return Agent(get_persona_config(persona))
```

### 3. **Facade Pattern**
```python
# ask() is a facade hiding complex LLM interaction
def ask(question):
    config = get_config()
    llm = create_llm(config)
    prompt = build_prompt(question)
    return llm.complete(prompt)
```

### 4. **Strategy Pattern**
```python
# Different LLM providers implement same interface
class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt): ...

class OpenAIProvider(LLMProvider): ...
class AnthropicProvider(LLMProvider): ...
class AzureProvider(LLMProvider): ...
```

---

## Extension Points

### Custom Skills
```python
from pyai import Skill, SkillResult

class MyCustomSkill(Skill):
    name = "my_skill"
    description = "Does something custom"
    
    async def execute(self, input: str) -> SkillResult:
        result = do_custom_thing(input)
        return SkillResult.ok(result)
```

### Custom Personas
```python
from pyai import agent

# Register custom persona
agent.register_persona(
    name="data_scientist",
    system_prompt="You are an expert data scientist...",
    skills=["pandas", "visualization", "statistics"]
)
```

### Custom Memory
```python
from pyai import Memory

class RedisMemory(Memory):
    def __init__(self, redis_url):
        self.client = redis.from_url(redis_url)
    
    def add_message(self, role, content):
        self.client.lpush("messages", json.dumps({role: content}))
```

---

## Future Roadmap

### Phase 1 (Current): Foundation ✅
- Core one-liner functions
- Basic RAG support
- Multi-provider LLM support

### Phase 2: Intelligence
- Automatic model selection based on task
- Smart caching and rate limiting
- Advanced memory with vector search

### Phase 3: Scale
- Async/concurrent operations
- Distributed agent swarms
- Cloud-native deployment

### Phase 4: Ecosystem
- Plugin marketplace
- Pre-trained agent templates
- Community contributions

---

## Performance Considerations

| Operation | Target Latency | Notes |
|-----------|---------------|-------|
| Import | < 100ms | Lazy loading ensures fast imports |
| ask() | < 1s | Network bound by LLM |
| rag.index() | O(n) | Linear with document count |
| rag.ask() | < 1s | Depends on index size |

---

## Security Architecture

### API Key Management
- Environment variables (recommended)
- Programmatic configuration
- Never logged or printed

### Code Execution
- Sandboxed execution for code skills
- Whitelist for allowed operations
- No file system access by default

### Data Privacy
- No data sent to external services except LLM
- Memory stored locally by default
- Optional encryption for sensitive data

---

*This document is the architectural blueprint for pyai. For API reference, see [API_REFERENCE.md](./API_REFERENCE.md).*
