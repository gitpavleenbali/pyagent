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
from __future__ import annotations

__version__ = "0.3.0"
__author__ = "PyAgent Contributors"

import importlib
from typing import TYPE_CHECKING, Any, Optional

# Type stubs for static analysis - these imports are never executed at runtime
# but satisfy type checkers like Pylance/mypy for __all__ exports
if TYPE_CHECKING:
    # One-liner functions
    from pyagent.easy.ask import ask as ask
    from pyagent.easy.research import research as research
    from pyagent.easy.summarize import summarize as summarize
    from pyagent.easy.extract import extract as extract
    from pyagent.easy.generate import generate as generate
    from pyagent.easy.translate import translate as translate
    from pyagent.easy.chat import chat as chat
    from pyagent.easy.agent_factory import agent as agent
    
    # Modules
    from pyagent.easy import rag as rag
    from pyagent.easy import fetch as fetch
    from pyagent.easy import analyze as analyze
    from pyagent.easy import code as code
    
    # Advanced features
    from pyagent.easy.handoff import handoff as handoff
    from pyagent.easy.mcp import mcp as mcp
    from pyagent.easy.guardrails import guardrails as guardrails
    from pyagent.easy.trace import trace as trace
    
    # Core components
    from pyagent.core.agent import Agent as Agent
    from pyagent.core.memory import Memory as Memory
    from pyagent.core.memory import ConversationMemory as ConversationMemory
    from pyagent.core.memory import VectorMemory as VectorMemory
    from pyagent.core.llm import OpenAIProvider as OpenAIProvider
    from pyagent.core.llm import OpenAIProvider as LLM
    from pyagent.core.llm import OpenAIProvider as AzureProvider
    from pyagent.core.llm import AnthropicProvider as AnthropicProvider
    
    # Instructions
    from pyagent.instructions import Instruction as Instruction
    from pyagent.instructions import SystemPrompt as SystemPrompt
    from pyagent.instructions import Context as Context
    from pyagent.instructions import Persona as Persona
    from pyagent.instructions import Guidelines as Guidelines
    
    # Skills
    from pyagent.skills import Skill as Skill
    from pyagent.skills import ToolSkill as ToolSkill
    from pyagent.skills import ActionSkill as ActionSkill
    from pyagent.skills import SkillRegistry as SkillRegistry
    
    # Blueprint
    from pyagent.blueprint import Blueprint as Blueprint
    from pyagent.blueprint import Workflow as Workflow
    from pyagent.blueprint import Pipeline as Pipeline
    from pyagent.blueprint import Orchestrator as Orchestrator
    
    # Models module
    from pyagent import models as models
    from pyagent.models import get_model as get_model
    
    # Sessions module
    from pyagent import sessions as sessions
    from pyagent.sessions import Session as Session
    from pyagent.sessions import SessionManager as SessionManager
    
    # Evaluation module
    from pyagent import evaluation as evaluation
    from pyagent.evaluation.evaluator import evaluate_agent as evaluate_agent
    from pyagent.evaluation.base import EvalSet as EvalSet
    from pyagent.evaluation.base import TestCase as TestCase
    
    # CLI module
    from pyagent import cli as cli
    
    # Code executor module
    from pyagent import code_executor as code_executor
    from pyagent.code_executor.executor import execute_python as execute_python
    
    # Runner module
    from pyagent import runner as runner
    from pyagent.runner import Runner as Runner
    from pyagent.runner import RunConfig as RunConfig
    from pyagent.runner import RunResult as RunResult
    
    # Config module
    from pyagent import config as config
    from pyagent.config import load_agent as load_agent
    from pyagent.config import AgentConfig as AgentConfig
    from pyagent.config import AgentBuilder as AgentBuilder
    
    # Plugins module
    from pyagent import plugins as plugins
    from pyagent.plugins import Plugin as Plugin
    from pyagent.plugins import PluginRegistry as PluginRegistry
    
    # Kernel module
    from pyagent import kernel as kernel
    from pyagent.kernel import Kernel as Kernel
    from pyagent.kernel import KernelBuilder as KernelBuilder
    from pyagent.kernel import ServiceRegistry as ServiceRegistry
    
    # OpenAPI module
    from pyagent import openapi as openapi
    from pyagent.openapi import OpenAPITools as OpenAPITools
    from pyagent.openapi import create_tools_from_openapi as create_tools_from_openapi
    
    # Tokens module
    from pyagent import tokens as tokens
    from pyagent.tokens import TokenCounter as TokenCounter
    from pyagent.tokens import count_tokens as count_tokens
    from pyagent.tokens import calculate_cost as calculate_cost
    from pyagent.tokens import CostTracker as CostTracker
    
    # Errors module
    from pyagent import errors as errors
    from pyagent.errors import PyAgentError as PyAgentError
    
    # Tools module
    from pyagent import tools as tools
    from pyagent.skills import Tool as Tool
    from pyagent.skills import ToolDiscovery as ToolDiscovery
    from pyagent.skills import discover_tools as discover_tools
    from pyagent.skills import ToolWatcher as ToolWatcher
    
    # Context caching
    from pyagent.core.context_cache import ContextCache as ContextCache
    from pyagent.core.context_cache import cache_context as cache_context
    
    # Multimodal module
    from pyagent import multimodal as multimodal
    from pyagent.multimodal import Image as Image
    from pyagent.multimodal import Audio as Audio
    from pyagent.multimodal import Video as Video
    from pyagent.multimodal import MultimodalContent as MultimodalContent
    
    # VectorDB module
    from pyagent import vectordb as vectordb
    from pyagent.vectordb import VectorStore as VectorStore
    from pyagent.vectordb import ChromaStore as ChromaStore
    from pyagent.vectordb import PineconeStore as PineconeStore
    from pyagent.vectordb import MemoryVectorStore as MemoryVectorStore
    from pyagent.vectordb import ChromaConnector as ChromaConnector
    from pyagent.vectordb import PineconeConnector as PineconeConnector
    
    # A2A module
    from pyagent import a2a as a2a
    from pyagent.a2a import A2AServer as A2AServer
    from pyagent.a2a import A2AClient as A2AClient
    from pyagent.a2a import RemoteAgent as RemoteAgent
    from pyagent.a2a import AgentCard as AgentCard
    
    # DevUI module
    from pyagent import devui as devui
    from pyagent.devui import DevUI as DevUI
    from pyagent.devui import launch_ui as launch_ui
    from pyagent.devui import AgentDashboard as AgentDashboard
    from pyagent.devui import AgentDebugger as AgentDebugger
    
    # Voice module
    from pyagent import voice as voice
    from pyagent.voice import VoiceSession as VoiceSession
    from pyagent.voice import AudioStream as AudioStream
    from pyagent.voice import Transcriber as Transcriber
    from pyagent.voice import Synthesizer as Synthesizer

# Cache for loaded modules
_cache: dict[str, Any] = {}


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

    # =========================================================================
    # NEW: Multi-provider models (inspired by Google ADK)
    # =========================================================================
    elif name == "models":
        _mod = importlib.import_module("pyagent.models")
        _cache[name] = _mod
        return _mod
    elif name == "get_model":
        from pyagent.models import get_model as _get_model
        _cache[name] = _get_model
        return _get_model

    # =========================================================================
    # NEW: Session management (inspired by Google ADK / OpenAI Agents)
    # =========================================================================
    elif name == "sessions":
        _mod = importlib.import_module("pyagent.sessions")
        _cache[name] = _mod
        return _mod
    elif name == "Session":
        from pyagent.sessions import Session as _Session
        _cache[name] = _Session
        return _Session
    elif name == "SessionManager":
        from pyagent.sessions import SessionManager as _SM
        _cache[name] = _SM
        return _SM

    # =========================================================================
    # NEW: Evaluation (similar to Google ADK eval)
    # =========================================================================
    elif name == "evaluation":
        _mod = importlib.import_module("pyagent.evaluation")
        _cache[name] = _mod
        return _mod
    elif name == "evaluate_agent":
        from pyagent.evaluation import evaluate_agent as _eval
        _cache[name] = _eval
        return _eval
    elif name == "EvalSet":
        from pyagent.evaluation import EvalSet as _EvalSet
        _cache[name] = _EvalSet
        return _EvalSet
    elif name == "TestCase":
        from pyagent.evaluation import TestCase as _TestCase
        _cache[name] = _TestCase
        return _TestCase

    # =========================================================================
    # NEW: CLI (similar to Google ADK CLI)
    # =========================================================================
    elif name == "cli":
        _mod = importlib.import_module("pyagent.cli")
        _cache[name] = _mod
        return _mod

    # =========================================================================
    # NEW: Code executor (similar to Google ADK code_executors)
    # =========================================================================
    elif name == "code_executor":
        _mod = importlib.import_module("pyagent.code_executor")
        _cache[name] = _mod
        return _mod
    elif name == "execute_python":
        from pyagent.code_executor import execute_python as _exec
        _cache[name] = _exec
        return _exec

    # =========================================================================
    # NEW: Runner pattern (similar to OpenAI Agents SDK)
    # =========================================================================
    elif name == "runner":
        _mod = importlib.import_module("pyagent.runner")
        _cache[name] = _mod
        return _mod
    elif name == "Runner":
        from pyagent.runner import Runner as _Runner
        _cache[name] = _Runner
        return _Runner
    elif name == "RunConfig":
        from pyagent.runner import RunConfig as _RunConfig
        _cache[name] = _RunConfig
        return _RunConfig
    elif name == "RunResult":
        from pyagent.runner import RunResult as _RunResult
        _cache[name] = _RunResult
        return _RunResult

    # =========================================================================
    # NEW: Agent config (similar to Google ADK agent.yaml)
    # =========================================================================
    elif name == "config":
        _mod = importlib.import_module("pyagent.config")
        _cache[name] = _mod
        return _mod
    elif name == "load_agent":
        from pyagent.config import load_agent as _load
        _cache[name] = _load
        return _load
    elif name == "AgentConfig":
        from pyagent.config import AgentConfig as _AC
        _cache[name] = _AC
        return _AC
    elif name == "AgentBuilder":
        from pyagent.config import AgentBuilder as _AB
        _cache[name] = _AB
        return _AB

    # =========================================================================
    # NEW: Plugins (similar to MS Semantic Kernel)
    # =========================================================================
    elif name == "plugins":
        _mod = importlib.import_module("pyagent.plugins")
        _cache[name] = _mod
        return _mod
    elif name == "Plugin":
        from pyagent.plugins import Plugin as _Plugin
        _cache[name] = _Plugin
        return _Plugin
    elif name == "PluginRegistry":
        from pyagent.plugins import PluginRegistry as _PR
        _cache[name] = _PR
        return _PR

    # =========================================================================
    # NEW: Kernel (similar to MS Semantic Kernel)
    # =========================================================================
    elif name == "kernel":
        _mod = importlib.import_module("pyagent.kernel")
        _cache[name] = _mod
        return _mod
    elif name == "Kernel":
        from pyagent.kernel import Kernel as _K
        _cache[name] = _K
        return _K
    elif name == "KernelBuilder":
        from pyagent.kernel import KernelBuilder as _KB
        _cache[name] = _KB
        return _KB
    elif name == "ServiceRegistry":
        from pyagent.kernel import ServiceRegistry as _SR
        _cache[name] = _SR
        return _SR

    # =========================================================================
    # NEW: OpenAPI tools (similar to Google ADK)
    # =========================================================================
    elif name == "openapi":
        _mod = importlib.import_module("pyagent.openapi")
        _cache[name] = _mod
        return _mod
    elif name == "OpenAPITools":
        from pyagent.openapi.tools import OpenAPITools as _OAT
        _cache[name] = _OAT
        return _OAT
    elif name == "create_tools_from_openapi":
        from pyagent.openapi import create_tools_from_openapi as _ctfo
        _cache[name] = _ctfo
        return _ctfo

    # =========================================================================
    # NEW: Token counting (like Anthropic)
    # =========================================================================
    elif name == "tokens":
        _mod = importlib.import_module("pyagent.tokens")
        _cache[name] = _mod
        return _mod
    elif name == "TokenCounter":
        from pyagent.tokens.counter import TokenCounter as _TC
        _cache[name] = _TC
        return _TC
    elif name == "count_tokens":
        from pyagent.tokens.counter import count_tokens as _ct
        _cache[name] = _ct
        return _ct
    elif name == "calculate_cost":
        from pyagent.tokens.cost import calculate_cost as _cc
        _cache[name] = _cc
        return _cc
    elif name == "CostTracker":
        from pyagent.tokens.cost import CostTracker as _CT
        _cache[name] = _CT
        return _CT

    # =========================================================================
    # NEW: Error types
    # =========================================================================
    elif name == "errors":
        _mod = importlib.import_module("pyagent.errors")
        _cache[name] = _mod
        return _mod
    elif name == "PyAgentError":
        from pyagent.errors import PyAgentError as _PAE
        _cache[name] = _PAE
        return _PAE

    # =========================================================================
    # NEW: Tool Auto-Discovery (like Strands Agents)
    # =========================================================================
    elif name == "tools":
        _mod = importlib.import_module("pyagent.tools")
        _cache[name] = _mod
        return _mod
    elif name == "Tool":
        from pyagent.tools import Tool as _Tool
        _cache[name] = _Tool
        return _Tool
    elif name == "ToolDiscovery":
        from pyagent.tools import ToolDiscovery as _TD
        _cache[name] = _TD
        return _TD
    elif name == "discover_tools":
        from pyagent.tools import discover_tools as _dt
        _cache[name] = _dt
        return _dt
    elif name == "ToolWatcher":
        from pyagent.tools import ToolWatcher as _TW
        _cache[name] = _TW
        return _TW

    # =========================================================================
    # NEW: Context Caching (like Google ADK)
    # =========================================================================
    elif name == "ContextCache":
        from pyagent.core.cache import ContextCache as _CC
        _cache[name] = _CC
        return _CC
    elif name == "cache_context":
        from pyagent.core.cache import cache_context as _cc
        _cache[name] = _cc
        return _cc

    # =========================================================================
    # NEW: Multimodal (images, audio, video)
    # =========================================================================
    elif name == "multimodal":
        _mod = importlib.import_module("pyagent.multimodal")
        _cache[name] = _mod
        return _mod
    elif name == "Image":
        from pyagent.multimodal import Image as _Image
        _cache[name] = _Image
        return _Image
    elif name == "Audio":
        from pyagent.multimodal import Audio as _Audio
        _cache[name] = _Audio
        return _Audio
    elif name == "Video":
        from pyagent.multimodal import Video as _Video
        _cache[name] = _Video
        return _Video
    elif name == "MultimodalContent":
        from pyagent.multimodal import MultimodalContent as _MC
        _cache[name] = _MC
        return _MC

    # =========================================================================
    # NEW: Vector Database Connectors
    # =========================================================================
    elif name == "vectordb":
        _mod = importlib.import_module("pyagent.vectordb")
        _cache[name] = _mod
        return _mod
    elif name == "VectorStore":
        from pyagent.vectordb import VectorStore as _VS
        _cache[name] = _VS
        return _VS
    elif name == "ChromaStore":
        from pyagent.vectordb import ChromaStore as _CS
        _cache[name] = _CS
        return _CS
    elif name == "PineconeStore":
        from pyagent.vectordb import PineconeStore as _PS
        _cache[name] = _PS
        return _PS
    elif name == "MemoryVectorStore":
        from pyagent.vectordb import MemoryVectorStore as _MVS
        _cache[name] = _MVS
        return _MVS

    # =========================================================================
    # NEW: A2A Protocol (Agent-to-Agent)
    # =========================================================================
    elif name == "a2a":
        _mod = importlib.import_module("pyagent.a2a")
        _cache[name] = _mod
        return _mod
    elif name == "A2AServer":
        from pyagent.a2a import A2AServer as _A2AS
        _cache[name] = _A2AS
        return _A2AS
    elif name == "A2AClient":
        from pyagent.a2a import A2AClient as _A2AC
        _cache[name] = _A2AC
        return _A2AC
    elif name == "RemoteAgent":
        from pyagent.a2a import RemoteAgent as _RA
        _cache[name] = _RA
        return _RA
    elif name == "AgentCard":
        from pyagent.a2a import AgentCard as _AC2
        _cache[name] = _AC2
        return _AC2

    # =========================================================================
    # NEW: Development UI
    # =========================================================================
    elif name == "devui":
        _mod = importlib.import_module("pyagent.devui")
        _cache[name] = _mod
        return _mod
    elif name == "DevUI":
        from pyagent.devui import DevUI as _DUI
        _cache[name] = _DUI
        return _DUI
    elif name == "launch_ui":
        from pyagent.devui import launch_ui as _lui
        _cache[name] = _lui
        return _lui
    elif name == "AgentDashboard":
        from pyagent.devui import AgentDashboard as _AD
        _cache[name] = _AD
        return _AD
    elif name == "AgentDebugger":
        from pyagent.devui import AgentDebugger as _ADbg
        _cache[name] = _ADbg
        return _ADbg

    # =========================================================================
    # NEW: Voice Streaming
    # =========================================================================
    elif name == "voice":
        _mod = importlib.import_module("pyagent.voice")
        _cache[name] = _mod
        return _mod
    elif name == "VoiceSession":
        from pyagent.voice import VoiceSession as _VS2
        _cache[name] = _VS2
        return _VS2
    elif name == "AudioStream":
        from pyagent.voice import AudioStream as _AS2
        _cache[name] = _AS2
        return _AS2
    elif name == "Transcriber":
        from pyagent.voice import Transcriber as _Trans
        _cache[name] = _Trans
        return _Trans
    elif name == "Synthesizer":
        from pyagent.voice import Synthesizer as _Synth
        _cache[name] = _Synth
        return _Synth

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
    # NEW: Multi-provider models (inspired by Google ADK)
    # =========================================================================
    "models",        # Multi-provider model module
    "get_model",     # Get model by name (auto-detects provider)

    # =========================================================================
    # NEW: Session management
    # =========================================================================
    "sessions",      # Session management module
    "Session",       # Session class
    "SessionManager",# Session manager

    # =========================================================================
    # NEW: Evaluation (like Google ADK eval)
    # =========================================================================
    "evaluation",    # Evaluation module
    "evaluate_agent",# Evaluate agent on test cases
    "EvalSet",       # Evaluation test set
    "TestCase",      # Single test case

    # =========================================================================
    # NEW: CLI and code execution
    # =========================================================================
    "cli",           # CLI module (pyagent run, eval, serve)
    "code_executor", # Safe code execution
    "execute_python",# Execute Python code safely

    # =========================================================================
    # NEW: Runner pattern (like OpenAI Agents SDK)
    # =========================================================================
    "runner",        # Runner module
    "Runner",        # Main runner class
    "RunConfig",     # Run configuration
    "RunResult",     # Run result

    # =========================================================================
    # NEW: Agent config (like Google ADK agent.yaml)
    # =========================================================================
    "config",        # Config module
    "load_agent",    # Load agent from YAML/JSON
    "AgentConfig",   # Agent configuration
    "AgentBuilder",  # Build agents from config

    # =========================================================================
    # NEW: Plugins (like MS Semantic Kernel)
    # =========================================================================
    "plugins",       # Plugins module
    "Plugin",        # Base plugin class
    "PluginRegistry",# Plugin registry

    # =========================================================================
    # NEW: Kernel (like MS Semantic Kernel)
    # =========================================================================
    "kernel",        # Kernel module
    "Kernel",        # Central kernel class
    "KernelBuilder", # Kernel builder
    "ServiceRegistry",# Service registry

    # =========================================================================
    # NEW: OpenAPI tools (like Google ADK)
    # =========================================================================
    "openapi",       # OpenAPI module
    "OpenAPITools",  # OpenAPI tools class
    "create_tools_from_openapi",  # Create tools from spec

    # =========================================================================
    # NEW: Token counting (like Anthropic)
    # =========================================================================
    "tokens",        # Token counting module
    "TokenCounter",  # Token counter class
    "count_tokens",  # Count tokens function
    "calculate_cost",# Calculate cost function
    "CostTracker",   # Cost tracking class

    # =========================================================================
    # NEW: Error types
    # =========================================================================
    "errors",        # Error module
    "PyAgentError",  # Base error class

    # =========================================================================
    # NEW: Tool Auto-Discovery (like Strands Agents)
    # =========================================================================
    "tools",         # Tools module
    "Tool",          # Tool class
    "ToolDiscovery", # Auto-discovery
    "discover_tools",# Discover tools from directory
    "ToolWatcher",   # Hot-reload watcher

    # =========================================================================
    # NEW: Context Caching (like Google ADK)
    # =========================================================================
    "ContextCache",  # Context cache class
    "cache_context", # Cache decorator

    # =========================================================================
    # NEW: Multimodal (images, audio, video)
    # =========================================================================
    "multimodal",    # Multimodal module
    "Image",         # Image class
    "Audio",         # Audio class
    "Video",         # Video class
    "MultimodalContent",  # Combined content

    # =========================================================================
    # NEW: Vector Database Connectors
    # =========================================================================
    "vectordb",      # Vector DB module
    "VectorStore",   # Base store class
    "ChromaStore",   # ChromaDB connector
    "PineconeStore", # Pinecone connector
    "MemoryVectorStore",  # In-memory store

    # =========================================================================
    # NEW: A2A Protocol (Agent-to-Agent)
    # =========================================================================
    "a2a",           # A2A module
    "A2AServer",     # Expose agent over HTTP
    "A2AClient",     # Connect to remote agent
    "RemoteAgent",   # Remote agent wrapper
    "AgentCard",     # Agent capability card

    # =========================================================================
    # NEW: Development UI
    # =========================================================================
    "devui",         # Dev UI module
    "DevUI",         # Web UI class
    "launch_ui",     # Quick launch function
    "AgentDashboard",# Monitoring dashboard
    "AgentDebugger", # Step-through debugger

    # =========================================================================
    # NEW: Voice Streaming
    # =========================================================================
    "voice",         # Voice module
    "VoiceSession",  # Real-time voice session
    "AudioStream",   # Audio streaming
    "Transcriber",   # Speech-to-text
    "Synthesizer",   # Text-to-speech

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
    api_key: Optional[str] = None,
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
