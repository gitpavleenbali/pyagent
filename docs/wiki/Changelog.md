# Changelog

All notable changes to PyAgent are documented here.

## [0.4.0] - 2026-02-16

### Release Summary
Major release achieving feature parity with industry-leading AI agent SDKs.

### Added

#### Phase 1 Features (Core)
- **Runner Pattern** - Structured agent execution (29 tests)
- **Agent Config YAML** - No-code agent configuration (24 tests)
- **Agents as Plugins** - Plugin architecture (22 tests)
- **OpenAPI Tools** - Auto-generate tools from specs (40 tests)
- **Token Counting** - Token utilities (40 tests)

#### Phase 2 Features (Advanced)
- **Tool Auto-Discovery** - Automatic tool loading (12 tests)
- **Context Caching** - Performance optimization (7 tests)
- **Session Rewind** - Checkpoint and rollback (6 tests)
- **Multimodal Vision** - Image/vision support (12 tests)
- **Vector DB Connectors** - ChromaDB, Pinecone, Weaviate, Qdrant (7 tests)

#### Phase 3 Features (Enterprise)
- **A2A Protocol** - Agent-to-Agent communication (12 tests)
- **Development UI** - Debugging and testing UI (9 tests)
- **Voice Streaming** - Real-time voice I/O (10 tests)

#### Phase 4 Features (Architecture)
- **Kernel Registry Pattern** - Service registry (35 tests)

### Changed
- Azure AD Authentication auto-detection
- Professional repository structure

### Test Coverage
- **671 tests passing**

---

## [0.3.0] - 2026-02-14

### Added
- Evaluation module for agent testing
- EvalSet, TestCase, Evaluator classes
- Multiple evaluation criteria
- Benchmark reporting

---

## [0.2.0] - 2026-02-13

### Added
- Workflow patterns (Chain, Parallel, Supervisor)
- Blueprint module for complex orchestration
- Session management (SQLite, Redis)
- Guardrails for input/output validation
- Tracing and observability

---

## [0.1.0] - 2026-02-12

### 🎉 Initial Release - "The Pandas of AI"

#### One-Liner Functions
- `ask()` - Ask questions
- `research()` - Deep research
- `summarize()` - Summarize content
- `extract()` - Extract data
- `generate()` - Generate content
- `translate()` - Translate text
- `chat()` - Interactive chat
- `agent()` - Create agents

#### Modules
- `rag` - RAG operations
- `fetch` - Real-time data
- `analyze` - Data analysis
- `code` - Code operations

#### Core Features
- Zero-configuration setup
- Multi-provider support (OpenAI, Anthropic, Azure)
- Full type hints
- Lazy imports

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.4.0 | 2026-02-16 | Enterprise features, A2A protocol |
| 0.3.0 | 2026-02-14 | Evaluation module |
| 0.2.0 | 2026-02-13 | Workflows, sessions, guardrails |
| 0.1.0 | 2026-02-12 | Initial release |

---

## Upgrade Guide

### From 0.3.x to 0.4.x

```python
# No breaking changes, new features are additive

# New: Runner pattern
from pyagent import Runner
result = Runner.run_sync(agent, "query")

# New: A2A protocol
from pyagent.a2a import A2AServer
server = A2AServer(agent)
```

### From 0.2.x to 0.3.x

```python
# New: Evaluation
from pyagent.evaluation import Evaluator, EvalSet
evaluator = Evaluator(agent)
results = evaluator.evaluate(eval_set)
```

### From 0.1.x to 0.2.x

```python
# New: Workflows
from pyagent.blueprint import Workflow
workflow = Workflow().add_step(analyze).add_step(summarize)

# New: Sessions
from pyagent.sessions import SQLiteSession
session = SQLiteSession("history.db")
```

---

## See Also

- [Installation](Installation) - Install PyAgent
- [Quick-Start](Quick-Start) - Getting started
- [Contributing](Contributing) - How to contribute
