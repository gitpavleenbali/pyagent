"""
agent() - Create custom agents in one line

The simplest way to create purpose-built AI agents.

Examples:
    >>> from pyagent import agent
    >>> 
    >>> # Create a custom agent
    >>> coder = agent("You are an expert Python developer")
    >>> result = coder("Write a function to parse JSON")
    >>> 
    >>> # Use prebuilt personas
    >>> researcher = agent(persona="researcher")
    >>> findings = researcher("Research quantum computing trends")
    >>>
    >>> # Create with specific model
    >>> gpt4_agent = agent("Be helpful", model="gpt-4o")
"""

from typing import Union, List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

from pyagent.easy.llm_interface import get_llm, LLMInterface
from pyagent.easy.chat import ChatSession


# Prebuilt personas
PERSONAS = {
    "assistant": "You are a helpful AI assistant. Be concise and accurate.",
    
    "coder": """You are an expert programmer. Write clean, efficient, well-documented code.
When asked to write code, return code only unless explanation is requested.
Follow best practices for the relevant programming language.""",
    
    "researcher": """You are a thorough research analyst. 
Provide comprehensive, well-organized research with multiple perspectives.
Always consider credibility and include relevant context.""",
    
    "writer": """You are a professional writer. 
Create engaging, well-structured content. Adapt your style to the task.
Focus on clarity, flow, and impact.""",
    
    "analyst": """You are a data analyst. 
Provide data-driven insights with clear reasoning.
Identify patterns, anomalies, and actionable conclusions.""",
    
    "teacher": """You are a patient, encouraging teacher.
Explain concepts clearly with examples. Check understanding.
Adapt explanations to the learner's level.""",
    
    "advisor": """You are a wise, thoughtful advisor.
Provide balanced, well-reasoned guidance.
Consider multiple angles and potential consequences.""",
    
    "critic": """You are a constructive critic.
Provide honest, detailed feedback aimed at improvement.
Balance criticism with recognition of strengths.""",
    
    "creative": """You are a creative thinker.
Think outside the box. Generate novel ideas and perspectives.
Don't be afraid to suggest unconventional approaches.""",
    
    "editor": """You are a meticulous editor.
Improve writing for clarity, conciseness, and impact.
Fix errors and enhance style while preserving voice.""",
}


@dataclass
class SimpleAgent:
    """
    A simple, callable agent that handles conversations.
    
    Usage:
        >>> agent = SimpleAgent("You are a helpful assistant")
        >>> agent("Hello!")
        'Hello! How can I help you?'
        
        >>> agent("What's 2+2?")  # Remembers previous context
        '4'
    """
    
    instructions: str
    model: str = None
    temperature: float = None
    name: str = "Agent"
    memory: bool = True
    max_history: int = 20
    
    _messages: List[Dict[str, str]] = field(default_factory=list, repr=False)
    
    def __post_init__(self):
        """Initialize message history with system message."""
        self._messages = [{"role": "system", "content": self.instructions}]
    
    def __call__(
        self,
        message: str,
        *,
        remember: bool = None,
        **kwargs
    ) -> str:
        """
        Send a message and get a response.
        
        Args:
            message: Your message/prompt
            remember: Override memory setting for this call
            **kwargs: Additional LLM parameters
            
        Returns:
            str: The agent's response
        """
        use_memory = remember if remember is not None else self.memory
        
        # Get LLM
        llm_kwargs = {}
        if self.model:
            llm_kwargs["model"] = self.model
        llm = get_llm(**llm_kwargs)
        
        if use_memory:
            # Add to history
            self._messages.append({"role": "user", "content": message})
            
            # Get response
            response = llm.chat(
                self._messages,
                temperature=self.temperature,
                **kwargs
            )
            
            # Add assistant response
            self._messages.append({"role": "assistant", "content": response.content})
            
            # Trim history if needed
            if len(self._messages) > self.max_history + 1:
                self._messages = self._messages[:1] + self._messages[-(self.max_history):]
        else:
            # One-shot, no memory
            response = llm.complete(
                message,
                system=self.instructions,
                temperature=self.temperature,
                **kwargs
            )
        
        return response.content
    
    def reset(self) -> None:
        """Clear conversation history."""
        self._messages = [{"role": "system", "content": self.instructions}]
    
    def clear(self) -> None:
        """Clear conversation history. Alias for reset()."""
        self.reset()
    
    def run(self, task: str, **kwargs) -> str:
        """Alias for __call__, useful for explicit task running."""
        return self(task, **kwargs)
    
    @property
    def history(self) -> List[Dict[str, str]]:
        """Get conversation history (excluding system message)."""
        return [m for m in self._messages if m["role"] != "system"]
    
    @property
    def messages(self) -> List[Dict[str, str]]:
        """Get all messages including system message."""
        return self._messages.copy()
    
    def __repr__(self) -> str:
        return f"Agent(name='{self.name}', messages={len(self._messages)})"


def agent(
    instructions: str = None,
    *,
    persona: str = None,
    name: str = None,
    model: str = None,
    temperature: float = None,
    memory: bool = True,
    **kwargs
) -> SimpleAgent:
    """
    Create a custom AI agent in one line.
    
    Args:
        instructions: What the agent should do / how it should behave
        persona: Use a prebuilt persona ("coder", "researcher", "writer", etc.)
        name: Name for the agent
        model: LLM model to use
        temperature: Response temperature
        memory: Whether to remember conversation history
        **kwargs: Additional parameters
        
    Returns:
        SimpleAgent: A callable agent
    
    Examples:
        >>> # Custom agent
        >>> helper = agent("You are a helpful assistant")
        >>> helper("What is Python?")
        'Python is a programming language...'
        
        >>> # Prebuilt persona
        >>> coder = agent(persona="coder")
        >>> coder("Write a hello world function")
        'def hello_world():\\n    print("Hello, World!")'
        
        >>> # Named agent with specific model
        >>> analyst = agent(
        ...     "You analyze business data",
        ...     name="DataBot",
        ...     model="gpt-4o"
        ... )
        
    Available personas:
        - assistant: General helpful assistant
        - coder: Expert programmer
        - researcher: Thorough research analyst
        - writer: Professional writer
        - analyst: Data analyst
        - teacher: Patient educator
        - advisor: Thoughtful advisor
        - critic: Constructive critic
        - creative: Creative thinker
        - editor: Meticulous editor
    """
    # Determine instructions
    if persona and not instructions:
        if persona.lower() in PERSONAS:
            instructions = PERSONAS[persona.lower()]
        else:
            instructions = f"You are a {persona}. Be helpful and professional."
    
    if not instructions:
        instructions = PERSONAS["assistant"]
    
    # Determine name
    if not name:
        if persona:
            name = persona.title()
        else:
            name = "Agent"
    
    return SimpleAgent(
        instructions=instructions,
        model=model,
        temperature=temperature,
        name=name,
        memory=memory
    )


# Convenience function for one-shot tasks
def run(
    task: str,
    *,
    as_agent: str = None,
    instructions: str = None,
    **kwargs
) -> str:
    """
    Run a one-shot task without creating an agent.
    
    Args:
        task: The task to perform
        as_agent: Persona to use
        instructions: Custom instructions
        **kwargs: Additional parameters
        
    Returns:
        str: The result
    
    Examples:
        >>> run("Write a haiku about coding")
        'Lines of logic flow...'
        
        >>> run("Review this code", as_agent="coder")
    """
    a = agent(instructions=instructions, persona=as_agent, memory=False, **kwargs)
    return a(task)


# Create preset agents for quick access
def coder(**kwargs) -> SimpleAgent:
    """Create a coding expert agent."""
    return agent(persona="coder", **kwargs)


def researcher(**kwargs) -> SimpleAgent:
    """Create a research analyst agent."""
    return agent(persona="researcher", **kwargs)


def writer(**kwargs) -> SimpleAgent:
    """Create a professional writer agent."""
    return agent(persona="writer", **kwargs)


def analyst(**kwargs) -> SimpleAgent:
    """Create a data analyst agent."""
    return agent(persona="analyst", **kwargs)


def teacher(**kwargs) -> SimpleAgent:
    """Create a patient teacher agent."""
    return agent(persona="teacher", **kwargs)
