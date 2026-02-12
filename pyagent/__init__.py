"""
PyAgent - The Pandas of AI Agents 🐼🤖

Build AI-powered applications in 3 lines or less. No boilerplate.
No configuration hell. Just results.

REVOLUTIONARY DESIGN PHILOSOPHY:
- One-liner operations for common AI tasks
- Batteries included - prebuilt agents ready to use
- Pandas-like API - if you know pandas, you know pyagent
- Zero configuration - sensible defaults that just work

QUICK START:
    # Research anything in one line
    >>> from pyagent import research
    >>> papers = research("quantum computing breakthroughs 2024")
    
    # RAG in two lines
    >>> from pyagent import rag
    >>> answer = rag.ask("./documents", "What is the main conclusion?")
    
    # Fetch real-time data instantly
    >>> from pyagent import fetch
    >>> weather = fetch.weather("New York")
    >>> news = fetch.news("AI startups")
    
    # Create custom agents effortlessly
    >>> from pyagent import agent
    >>> my_agent = agent("You are a data analyst", model="gpt-4")
    >>> result = my_agent("Analyze this sales data: ...")

ADVANCED USAGE (when you need more control):
    >>> from pyagent import Agent, Blueprint, Workflow
    >>> # Full access to underlying components
"""

__version__ = "0.1.0"
__author__ = "PyAgent Contributors"

import importlib

# Cache for loaded modules
_cache = {}


def __getattr__(name):
    """Lazy import for all modules to avoid circular dependencies."""
    if name in _cache:
        return _cache[name]
    
    # One-liner functions
    if name == "ask":
        from pyagent.easy.ask import ask as _ask
        _cache[name] = _ask
        return _ask
    elif name == "research":
        from pyagent.easy.research import research as _research
        _cache[name] = _research
        return _research
    elif name == "summarize":
        from pyagent.easy.summarize import summarize as _summarize
        _cache[name] = _summarize
        return _summarize
    elif name == "extract":
        from pyagent.easy.extract import extract as _extract
        _cache[name] = _extract
        return _extract
    elif name == "generate":
        from pyagent.easy.generate import generate as _generate
        _cache[name] = _generate
        return _generate
    elif name == "translate":
        from pyagent.easy.translate import translate as _translate
        _cache[name] = _translate
        return _translate
    elif name == "chat":
        from pyagent.easy.chat import chat as _chat
        _cache[name] = _chat
        return _chat
    elif name == "agent":
        from pyagent.easy.agent_factory import agent as _agent
        _cache[name] = _agent
        return _agent
    
    # Modules (use importlib to avoid recursion)
    elif name == "rag":
        _mod = importlib.import_module("pyagent.easy.rag")
        _cache[name] = _mod
        return _mod
    elif name == "fetch":
        _mod = importlib.import_module("pyagent.easy.fetch")
        _cache[name] = _mod
        return _mod
    elif name == "analyze":
        _mod = importlib.import_module("pyagent.easy.analyze")
        _cache[name] = _mod
        return _mod
    elif name == "code":
        _mod = importlib.import_module("pyagent.easy.code")
        _cache[name] = _mod
        return _mod
    
    # NEW: Advanced features (competitive with OpenAI Agents, Strands)
    elif name == "handoff":
        from pyagent.easy.handoff import handoff as _handoff
        _cache[name] = _handoff
        return _handoff
    elif name == "mcp":
        from pyagent.easy.mcp import mcp as _mcp
        _cache[name] = _mcp
        return _mcp
    elif name == "guardrails":
        from pyagent.easy.guardrails import guardrails as _guardrails
        _cache[name] = _guardrails
        return _guardrails
    elif name == "trace":
        from pyagent.easy.trace import trace as _trace
        _cache[name] = _trace
        return _trace
    
    # Core components
    elif name == "Agent":
        from pyagent.core.agent import Agent as _Agent
        _cache[name] = _Agent
        return _Agent
    elif name == "Memory":
        from pyagent.core.memory import Memory as _Memory
        _cache[name] = _Memory
        return _Memory
    elif name == "ConversationMemory":
        from pyagent.core.memory import ConversationMemory as _ConversationMemory
        _cache[name] = _ConversationMemory
        return _ConversationMemory
    elif name == "VectorMemory":
        from pyagent.core.memory import VectorMemory as _VectorMemory
        _cache[name] = _VectorMemory
        return _VectorMemory
    elif name in ("LLM", "OpenAIProvider"):
        from pyagent.core.llm import OpenAIProvider as _OpenAIProvider
        _cache[name] = _OpenAIProvider
        return _OpenAIProvider
    elif name == "AzureProvider":
        from pyagent.core.llm import OpenAIProvider as _Azure
        _cache[name] = _Azure
        return _Azure  # Azure uses same client
    elif name == "AnthropicProvider":
        from pyagent.core.llm import AnthropicProvider as _Anthropic
        _cache[name] = _Anthropic
        return _Anthropic
    
    # Instructions
    elif name == "Instruction":
        from pyagent.instructions import Instruction as _Inst
        _cache[name] = _Inst
        return _Inst
    elif name == "SystemPrompt":
        from pyagent.instructions import SystemPrompt as _SP
        _cache[name] = _SP
        return _SP
    elif name == "Context":
        from pyagent.instructions import Context as _Ctx
        _cache[name] = _Ctx
        return _Ctx
    elif name == "Persona":
        from pyagent.instructions import Persona as _Persona
        _cache[name] = _Persona
        return _Persona
    elif name == "Guidelines":
        from pyagent.instructions import Guidelines as _Guide
        _cache[name] = _Guide
        return _Guide
    
    # Skills
    elif name == "Skill":
        from pyagent.skills import Skill as _Skill
        _cache[name] = _Skill
        return _Skill
    elif name == "ToolSkill":
        from pyagent.skills import ToolSkill as _TS
        _cache[name] = _TS
        return _TS
    elif name == "ActionSkill":
        from pyagent.skills import ActionSkill as _AS
        _cache[name] = _AS
        return _AS
    elif name == "SkillRegistry":
        from pyagent.skills import SkillRegistry as _SR
        _cache[name] = _SR
        return _SR
    
    # Blueprint
    elif name == "Blueprint":
        from pyagent.blueprint import Blueprint as _BP
        _cache[name] = _BP
        return _BP
    elif name == "Workflow":
        from pyagent.blueprint import Workflow as _WF
        _cache[name] = _WF
        return _WF
    elif name == "Pipeline":
        from pyagent.blueprint import Pipeline as _PL
        _cache[name] = _PL
        return _PL
    elif name == "Orchestrator":
        from pyagent.blueprint import Orchestrator as _Orch
        _cache[name] = _Orch
        return _Orch
    
    raise AttributeError(f"module 'pyagent' has no attribute '{name}'")


__all__ = [
    # =========================================================================
    # HIGH-LEVEL API (Use these for 90% of tasks)
    # =========================================================================
    
    # One-liner functions
    "ask",           # Ask anything, get intelligent answers
    "research",      # Deep research on any topic
    "summarize",     # Summarize text, documents, URLs
    "extract",       # Extract structured data
    "generate",      # Generate content (text, code, etc.)
    "translate",     # Translate between languages
    "chat",          # Interactive chat session
    
    # Prebuilt modules
    "rag",           # RAG operations (index, ask, search)
    "fetch",         # Data fetching (weather, news, stocks, etc.)
    "analyze",       # Data analysis (describe, insights, visualize)
    "code",          # Code operations (write, review, debug, explain)
    
    # Quick agent factory
    "agent",         # Create custom agents in one line
    
    # Advanced features (competitive with OpenAI Agents, Strands)
    "handoff",       # Multi-agent handoffs and team routing
    "mcp",           # Model Context Protocol server support
    "guardrails",    # Input/output validation and safety
    "trace",         # Tracing and observability
    
    # =========================================================================
    # LOW-LEVEL API (For power users)
    # =========================================================================
    
    # Core
    "Agent",
    "Memory",
    "ConversationMemory", 
    "VectorMemory",
    "LLM",
    "OpenAIProvider",
    "AzureProvider",
    "AnthropicProvider",
    
    # Instructions
    "Instruction",
    "SystemPrompt", 
    "Context",
    "Persona",
    "Guidelines",
    
    # Skills
    "Skill",
    "ToolSkill",
    "ActionSkill",
    "SkillRegistry",
    
    # Blueprint
    "Blueprint",
    "Workflow",
    "Pipeline",
    "Orchestrator",
]


# =============================================================================
# CONFIGURATION (Optional - works without any config)
# =============================================================================

def configure(
    api_key: str = None,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    **kwargs
):
    """
    Configure pyagent globally. Optional - pyagent works with env vars.
    
    Args:
        api_key: Your API key (or set OPENAI_API_KEY env var)
        model: Default model to use
        provider: Default provider (openai, azure, anthropic)
    
    Example:
        >>> import pyagent
        >>> pyagent.configure(api_key="sk-...", model="gpt-4o")
    """
    from pyagent.easy.config import set_config
    set_config(api_key=api_key, model=model, provider=provider, **kwargs)


def version():
    """Return the pyagent version."""
    return __version__
