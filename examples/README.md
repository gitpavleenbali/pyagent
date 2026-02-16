# PyAgent Examples

This directory contains working examples demonstrating PyAgent's capabilities.

## Prerequisites

### Authentication Setup

Choose one of the following authentication methods:

**Option 1: OpenAI**
```bash
export OPENAI_API_KEY=sk-your-key
```

**Option 2: Azure OpenAI with API Key**
```bash
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_API_KEY=your-key
export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

**Option 3: Azure OpenAI with Azure AD (Recommended for Enterprise)**
```bash
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
# No API key needed - uses your Azure login automatically
az login  # Login to Azure first
```

## Examples

### Basic Agent (`basic_agent.py`)

Demonstrates the core Agent API with system prompts, skills, and LLM configuration.

```bash
python examples/basic_agent.py
```

### Weather App (`weather_app.py`)

A complete weather assistant showing:
- Simple weather queries
- Multi-city comparison
- Weather-based recommendations

```bash
python examples/weather_app.py
```

### Smart Research Assistant (`smart_research_assistant.py`)

Showcases advanced features:
- `research()` - Deep research on topics
- `rag` - Document Q&A
- `agent()` - Custom personas
- `code.*` - Code operations
- `chat()` - Conversations with memory
- `analyze.*` - Text analysis

```bash
python examples/smart_research_assistant.py
```

### Multi-Agent Workflow (`multi_agent_workflow.py`)

Demonstrates multi-agent collaboration:
- Workflow patterns
- Agent chaining
- Supervisor pattern

```bash
python examples/multi_agent_workflow.py
```

### Custom Skills (`custom_skills.py`)

Shows how to create custom tools:
- `@tool` decorator
- Skill classes
- Action skills for CRUD

```bash
python examples/custom_skills.py
```

### Comprehensive Examples (`comprehensive_examples.py`)

Full showcase of all PyAgent capabilities:
- Simple Q&A
- Custom agents
- Chat with memory
- RAG pipeline
- Code generation
- Research & summarization
- Data extraction

```bash
python examples/comprehensive_examples.py
```

### Quick Start (`quick_start.py`)

Reference file showing the simplest API patterns.

```bash
python examples/quick_start.py
```

## Configuration Helper

All examples use `config_helper.py` which automatically:
1. Detects your authentication method
2. Configures PyAgent appropriately
3. Falls back to Azure AD if no API key is set

## Running Examples

From the repository root:

```bash
# Run any example
python examples/basic_agent.py

# Or from the examples directory
cd examples
python basic_agent.py
```

## Troubleshooting

### "Please configure credentials"
Set the environment variables as shown above.

### SSL/Network Errors
Some corporate networks may block external requests. Try:
- Using a VPN
- Checking proxy settings
- Using Azure endpoints within your network

### Import Errors
Ensure PyAgent is installed:
```bash
pip install -e .
```
