"""
chat() - Interactive chat sessions

Start a conversation that maintains context.

Examples:
    >>> from pyai import chat
    >>> session = chat()
    >>> session("What is Python?")
    'Python is a programming language...'
    >>> session("What are its main features?")
    'Python has several key features including...'  # Remembers context
"""

from dataclasses import dataclass, field
from typing import Dict, List

from pyai.easy.llm_interface import get_llm


@dataclass
class ChatSession:
    """
    An interactive chat session with memory.

    Examples:
        >>> session = chat("You are a helpful coding assistant")
        >>> session("How do I read a file in Python?")
        'You can read a file using...'
        >>> session("What about writing?")
        'To write to a file...'  # Remembers previous context
    """

    system: str = "You are a helpful AI assistant."
    model: str = None
    temperature: float = None
    messages: List[Dict[str, str]] = field(default_factory=list)
    max_history: int = 50

    def __post_init__(self):
        """Initialize with system message."""
        if self.system:
            self.messages = [{"role": "system", "content": self.system}]

    def __call__(self, message: str, **kwargs) -> str:
        """Send a message and get a response."""
        return self.send(message, **kwargs)

    def send(self, message: str, **kwargs) -> str:
        """
        Send a message and get a response.

        Args:
            message: Your message
            **kwargs: Additional LLM parameters

        Returns:
            str: The assistant's response
        """
        # Add user message
        self.messages.append({"role": "user", "content": message})

        # Get LLM
        llm_kwargs = {}
        if self.model:
            llm_kwargs["model"] = self.model
        llm = get_llm(**llm_kwargs)

        # Get response
        response = llm.chat(
            self.messages, temperature=self.temperature or kwargs.get("temperature"), **kwargs
        )

        # Add assistant message
        self.messages.append({"role": "assistant", "content": response.content})

        # Trim history if needed
        if len(self.messages) > self.max_history + 1:
            # Keep system message + last max_history messages
            self.messages = self.messages[:1] + self.messages[-(self.max_history) :]

        return response.content

    def say(self, message: str, **kwargs) -> str:
        """Alias for send() - send a message and get a response."""
        return self.send(message, **kwargs)

    def clear(self) -> None:
        """Clear conversation history (keep system message)."""
        self.messages = [self.messages[0]] if self.messages else []

    def reset(self, system: str = None) -> None:
        """Reset with optional new system message."""
        self.system = system or self.system
        self.messages = [{"role": "system", "content": self.system}]

    @property
    def history(self) -> List[Dict[str, str]]:
        """Get conversation history (excluding system message)."""
        return [m for m in self.messages if m["role"] != "system"]

    def __repr__(self) -> str:
        return f"ChatSession(messages={len(self.messages)})"


def chat(
    system: str = None,
    *,
    persona: str = None,
    model: str = None,
    temperature: float = None,
    **kwargs,
) -> ChatSession:
    """
    Create an interactive chat session.

    Args:
        system: System message / instructions
        persona: Predefined persona ("coder", "teacher", "analyst", etc.)
        model: Override default model
        temperature: Default temperature for the session
        **kwargs: Additional parameters

    Returns:
        ChatSession: A callable chat session

    Examples:
        >>> session = chat()
        >>> session("Hello!")
        'Hello! How can I help you today?'

        >>> coder = chat(persona="coder")
        >>> coder("Write a hello world in Python")
        'print("Hello, World!")'

        >>> teacher = chat("You are a patient math teacher")
        >>> teacher("Explain calculus")
        'Calculus is the study of...'
    """
    # Predefined personas
    personas = {
        "coder": "You are an expert programmer. Help with coding questions and write clean, efficient code.",
        "teacher": "You are a patient, encouraging teacher. Explain concepts clearly with examples.",
        "analyst": "You are a data analyst. Provide data-driven insights and analysis.",
        "writer": "You are a professional writer. Help with writing tasks, editing, and creative content.",
        "researcher": "You are a thorough researcher. Provide comprehensive, well-sourced information.",
        "advisor": "You are a wise advisor. Provide thoughtful, balanced advice.",
        "coach": "You are a supportive coach. Motivate and guide toward improvement.",
    }

    if persona and not system:
        system = personas.get(persona, f"You are a {persona}.")

    if not system:
        system = "You are a helpful AI assistant."

    return ChatSession(system=system, model=model, temperature=temperature)


# Quick one-message chat (no session needed)
def quick_chat(message: str, system: str = None, **kwargs) -> str:
    """
    Send a single message without creating a session.

    Args:
        message: Your message
        system: Optional system message
        **kwargs: Additional parameters

    Returns:
        str: The response
    """
    session = chat(system=system, **kwargs)
    return session(message)
