"""
PyAgent Type Stubs - For static type checking support

This file provides type hints for IDE/Pylance support with lazy imports.
"""

from typing import (
    Any, Dict, List, Optional, Union, Callable, TypeVar, 
    Literal, overload, TYPE_CHECKING
)
from dataclasses import dataclass

# Version
__version__: str

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PyAgentConfig:
    """Global configuration for PyAgent"""
    api_key: Optional[str] = None
    provider: Literal["openai", "anthropic", "azure"] = "openai"
    model: str = "gpt-4o-mini"
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60
    retry_attempts: int = 3

def configure(
    api_key: Optional[str] = None,
    provider: Literal["openai", "anthropic", "azure"] = "openai",
    model: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    azure_deployment: Optional[str] = None,
    **kwargs: Any
) -> PyAgentConfig: ...

def get_config() -> PyAgentConfig: ...

# =============================================================================
# One-Liner Functions
# =============================================================================

def ask(
    question: str,
    *,
    detailed: bool = False,
    concise: bool = False,
    format: Optional[Literal["bullet", "numbered", "markdown"]] = None,
    creative: bool = False,
    as_json: bool = False,
    model: Optional[str] = None,
) -> Union[str, Dict[str, Any]]:
    """
    Ask any question, get an intelligent answer.
    
    Args:
        question: The question to ask
        detailed: Get a comprehensive answer
        concise: Get a brief answer
        format: Output format (bullet, numbered, markdown)
        creative: More creative/varied response
        as_json: Return structured JSON
        model: Override default model
        
    Returns:
        Answer string or JSON dict
    """
    ...

@dataclass
class ResearchResult:
    """Result from research operation"""
    summary: str
    key_points: List[str]
    insights: List[str]
    sources: List[str]
    confidence: float

def research(
    topic: str,
    *,
    quick: bool = False,
    as_insights: bool = False,
    focus: Optional[str] = None,
    depth: Literal["shallow", "medium", "deep"] = "medium",
) -> Union[ResearchResult, str, List[str]]:
    """
    Deep research on any topic.
    
    Args:
        topic: The topic to research
        quick: Quick summary only
        as_insights: Return insights list only
        focus: Focus area within the topic
        depth: Research depth
        
    Returns:
        ResearchResult, summary string, or insights list
    """
    ...

def summarize(
    content: str,
    *,
    length: Literal["short", "medium", "long"] = "medium",
    focus: Optional[str] = None,
    as_bullets: bool = False,
) -> str:
    """
    Summarize text, files, or URLs.
    
    Args:
        content: Text, file path, or URL to summarize
        length: Summary length
        focus: Focus area for summary
        as_bullets: Return as bullet points
        
    Returns:
        Summary string
    """
    ...

T = TypeVar('T')

def extract(
    content: str,
    schema: type[T],
    *,
    strict: bool = False,
) -> T:
    """
    Extract structured data from text.
    
    Args:
        content: Text to extract from
        schema: Pydantic model or dict schema
        strict: Strict validation
        
    Returns:
        Instance of schema type
    """
    ...

def generate(
    prompt: str,
    *,
    type: Literal["text", "code", "email", "blog", "docs"] = "text",
    length: Literal["short", "medium", "long"] = "medium",
    style: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """
    Generate content of various types.
    
    Args:
        prompt: What to generate
        type: Content type
        length: Content length
        style: Style guidance
        language: Programming language (for code)
        
    Returns:
        Generated content
    """
    ...

def translate(
    text: str,
    to: str,
    *,
    from_lang: Optional[str] = None,
    formal: bool = False,
    preserve_formatting: bool = True,
) -> str:
    """
    Translate text between languages.
    
    Args:
        text: Text to translate
        to: Target language code (en, es, fr, etc.)
        from_lang: Source language (auto-detected if None)
        formal: Use formal register
        preserve_formatting: Keep original formatting
        
    Returns:
        Translated text
    """
    ...

# =============================================================================
# Chat Sessions
# =============================================================================

class ChatSession:
    """Interactive chat session with memory"""
    
    def __init__(
        self,
        system_message: Optional[str] = None,
        persona: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None: ...
    
    def say(self, message: str) -> str:
        """Send a message and get response"""
        ...
    
    def __call__(self, message: str) -> str:
        """Shorthand for say()"""
        ...
    
    def reset(self) -> None:
        """Clear conversation history"""
        ...
    
    @property
    def history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        ...

def chat(
    system_message: Optional[str] = None,
    *,
    persona: Optional[str] = None,
    model: Optional[str] = None,
) -> ChatSession:
    """
    Create an interactive chat session.
    
    Args:
        system_message: Custom system prompt
        persona: Prebuilt persona (teacher, advisor, etc.)
        model: Override default model
        
    Returns:
        ChatSession instance
    """
    ...

# =============================================================================
# Agent Factory
# =============================================================================

class Agent:
    """Custom AI agent"""
    
    name: str
    
    def __init__(
        self,
        system_message: Optional[str] = None,
        *,
        persona: Optional[str] = None,
        name: Optional[str] = None,
        model: Optional[str] = None,
        memory: bool = True,
    ) -> None: ...
    
    def __call__(self, message: str) -> str:
        """Send message to agent"""
        ...
    
    def reset(self) -> None:
        """Reset agent memory"""
        ...

@overload
def agent(
    system_message: str,
    *,
    name: Optional[str] = None,
    model: Optional[str] = None,
    memory: bool = True,
) -> Agent: ...

@overload
def agent(
    *,
    persona: str,
    name: Optional[str] = None,
    model: Optional[str] = None,
    memory: bool = True,
) -> Agent: ...

def agent(
    system_message: Optional[str] = None,
    *,
    persona: Optional[str] = None,
    name: Optional[str] = None,
    model: Optional[str] = None,
    memory: bool = True,
) -> Agent:
    """
    Create a custom AI agent.
    
    Args:
        system_message: Custom system prompt
        persona: Prebuilt persona name
        name: Agent name
        model: Model to use
        memory: Enable conversation memory
        
    Returns:
        Agent instance
    """
    ...

# =============================================================================
# RAG Module
# =============================================================================

class rag:
    """RAG (Retrieval-Augmented Generation) operations"""
    
    @staticmethod
    def index(
        sources: Union[str, List[str]],
        *,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> "IndexedDocuments":
        """Index documents for RAG"""
        ...
    
    @staticmethod
    def ask(
        source: Union[str, "IndexedDocuments"],
        question: str,
        *,
        top_k: int = 5,
    ) -> str:
        """Ask a question against documents"""
        ...
    
    @staticmethod
    def from_url(url: str, question: str) -> str:
        """RAG from a URL"""
        ...
    
    @staticmethod
    def from_text(text: str, question: str) -> str:
        """RAG from raw text"""
        ...

class IndexedDocuments:
    """Indexed documents for RAG queries"""
    
    source: str
    chunk_count: int
    
    def ask(self, question: str, *, top_k: int = 5) -> str:
        """Query the indexed documents"""
        ...

# =============================================================================
# Fetch Module
# =============================================================================

@dataclass
class WeatherResult:
    """Weather data result"""
    location: str
    temperature: float
    conditions: str
    humidity: int
    wind_speed: float
    forecast: List[Dict[str, Any]]

@dataclass
class NewsArticle:
    """News article"""
    title: str
    source: str
    url: str
    published: str
    summary: Optional[str]

@dataclass
class StockData:
    """Stock market data"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int

@dataclass
class CryptoData:
    """Cryptocurrency data"""
    symbol: str
    price: float
    change_24h: float
    market_cap: float

class fetch:
    """Real-time data fetching"""
    
    @staticmethod
    def weather(location: str) -> WeatherResult:
        """Get weather for a location"""
        ...
    
    @staticmethod
    def news(topic: str, *, limit: int = 10) -> List[NewsArticle]:
        """Get news articles on a topic"""
        ...
    
    @staticmethod
    def stock(symbol: str) -> StockData:
        """Get stock data"""
        ...
    
    @staticmethod
    def crypto(symbol: str) -> CryptoData:
        """Get cryptocurrency data"""
        ...
    
    @staticmethod
    def facts(topic: str, *, count: int = 5) -> List[str]:
        """Get facts about a topic"""
        ...

# =============================================================================
# Analyze Module
# =============================================================================

@dataclass
class AnalysisResult:
    """Data analysis result"""
    summary: str
    insights: List[str]
    statistics: Dict[str, Any]
    recommendations: List[str]

@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    sentiment: Literal["positive", "negative", "neutral"]
    score: float
    aspects: Dict[str, str]

class analyze:
    """Data and text analysis"""
    
    @staticmethod
    def data(data: Any, *, goal: Optional[str] = None) -> AnalysisResult:
        """Analyze data (DataFrame, dict, list)"""
        ...
    
    @staticmethod
    def text(text: str, *, aspects: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze text content"""
        ...
    
    @staticmethod
    def sentiment(text: str) -> SentimentResult:
        """Analyze sentiment of text"""
        ...
    
    @staticmethod
    def compare(*items: Any, criteria: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare multiple items"""
        ...

# =============================================================================
# Code Module
# =============================================================================

@dataclass
class CodeReview:
    """Code review result"""
    score: int
    issues: List[str]
    suggestions: List[str]
    security_concerns: List[str]

class code:
    """Code generation and analysis"""
    
    @staticmethod
    def write(
        description: str,
        *,
        language: str = "python",
        style: Optional[str] = None,
    ) -> str:
        """Generate code from description"""
        ...
    
    @staticmethod
    def review(code: str, *, focus: Optional[str] = None) -> CodeReview:
        """Review code for issues"""
        ...
    
    @staticmethod
    def debug(error: str, *, code: Optional[str] = None) -> str:
        """Debug an error"""
        ...
    
    @staticmethod
    def explain(code: str, *, level: Literal["beginner", "intermediate", "expert"] = "intermediate") -> str:
        """Explain code"""
        ...
    
    @staticmethod
    def refactor(code: str, *, goal: str = "readability") -> str:
        """Refactor code"""
        ...
    
    @staticmethod
    def convert(
        code: str,
        *,
        from_lang: str,
        to_lang: str,
    ) -> str:
        """Convert code between languages"""
        ...

# =============================================================================
# Core Classes (Advanced Usage)
# =============================================================================

class Blueprint:
    """Blueprint for complex agent workflows"""
    ...

class Workflow:
    """Multi-step workflow orchestration"""
    ...

class Pipeline:
    """Data processing pipeline"""
    ...

# Core components
from pyagent.core.agent import Agent as CoreAgent
from pyagent.core.memory import Memory, ConversationMemory, VectorMemory
from pyagent.core.llm import LLMProvider, OpenAIProvider, AnthropicProvider

# Instructions
from pyagent.instructions.instruction import Instruction
from pyagent.instructions.system_prompt import SystemPrompt
from pyagent.instructions.context import Context
from pyagent.instructions.persona import Persona
from pyagent.instructions.guidelines import Guidelines

# Skills
from pyagent.skills.skill import Skill
from pyagent.skills.tool_skill import ToolSkill
from pyagent.skills.action_skill import ActionSkill
from pyagent.skills.registry import SkillRegistry
