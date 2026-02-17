# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Base Model Classes

Provides the abstract base class and common types for all LLM providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union


class ModelCapability(Enum):
    """Capabilities that models may support."""

    CHAT = "chat"
    COMPLETION = "completion"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    AUDIO = "audio"
    STREAMING = "streaming"
    JSON_MODE = "json_mode"
    STRUCTURED_OUTPUT = "structured_output"


@dataclass
class ModelConfig:
    """Configuration for a model instance.

    Attributes:
        model_id: The model identifier (e.g., "gpt-4o", "llama3.2")
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens in response
        top_p: Nucleus sampling parameter
        frequency_penalty: Penalty for token frequency
        presence_penalty: Penalty for token presence
        stop: Stop sequences
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        extra: Additional provider-specific parameters
    """

    model_id: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    timeout: float = 60.0
    max_retries: int = 3
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """A conversation message.

    Attributes:
        role: The role (system, user, assistant, tool)
        content: The message content
        name: Optional name for the message author
        tool_calls: List of tool calls (for assistant messages)
        tool_call_id: ID of the tool call this message responds to
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


@dataclass
class ToolCall:
    """A tool/function call from the model.

    Attributes:
        id: Unique identifier for the tool call
        name: Name of the tool/function
        arguments: Arguments as a JSON string or dict
    """

    id: str
    name: str
    arguments: Union[str, Dict[str, Any]]

    def get_arguments(self) -> Dict[str, Any]:
        """Get arguments as a dictionary."""
        if isinstance(self.arguments, str):
            import json

            return json.loads(self.arguments)
        return self.arguments


@dataclass
class Usage:
    """Token usage information.

    Attributes:
        prompt_tokens: Tokens in the prompt
        completion_tokens: Tokens in the completion
        total_tokens: Total tokens used
        cached_tokens: Tokens served from cache (if applicable)
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class ModelResponse:
    """Response from a model invocation.

    Attributes:
        content: The text content of the response
        role: The role (usually "assistant")
        tool_calls: Any tool calls requested
        usage: Token usage information
        model: The model that generated the response
        finish_reason: Why the model stopped (stop, length, tool_calls)
        raw_response: The raw provider response
    """

    content: str
    role: str = "assistant"
    tool_calls: Optional[List[ToolCall]] = None
    usage: Optional[Usage] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return bool(self.tool_calls)


class BaseModel(ABC):
    """Abstract base class for all LLM providers.

    Subclasses must implement:
    - generate(): Synchronous text generation
    - generate_async(): Asynchronous text generation
    - stream(): Streaming text generation
    - stream_async(): Async streaming text generation

    Example:
        class MyModel(BaseModel):
            def generate(self, messages, **kwargs):
                # Implementation
                pass
    """

    def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
        """Initialize the model.

        Args:
            config: Model configuration
            **kwargs: Additional configuration overrides
        """
        self.config = config or ModelConfig()
        # Apply kwargs to config
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.config.extra[key] = value

    @property
    def model_id(self) -> str:
        """Get the model identifier."""
        return self.config.model_id

    @property
    @abstractmethod
    def provider(self) -> str:
        """Get the provider name (e.g., 'azure', 'openai', 'ollama')."""
        pass

    @property
    def capabilities(self) -> List[ModelCapability]:
        """Get the capabilities this model supports."""
        return [ModelCapability.CHAT, ModelCapability.STREAMING]

    def supports(self, capability: ModelCapability) -> bool:
        """Check if this model supports a capability."""
        return capability in self.capabilities

    @abstractmethod
    def generate(
        self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs
    ) -> ModelResponse:
        """Generate a response synchronously.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool definitions
            **kwargs: Additional parameters

        Returns:
            ModelResponse with the generated content
        """
        pass

    @abstractmethod
    async def generate_async(
        self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs
    ) -> ModelResponse:
        """Generate a response asynchronously.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool definitions
            **kwargs: Additional parameters

        Returns:
            ModelResponse with the generated content
        """
        pass

    def stream(
        self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs
    ) -> Any:
        """Stream a response synchronously.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool definitions
            **kwargs: Additional parameters

        Yields:
            Chunks of the response
        """
        # Default implementation: just yield the full response
        response = self.generate(messages, tools, **kwargs)
        yield response

    async def stream_async(
        self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs
    ) -> AsyncIterator[Any]:
        """Stream a response asynchronously.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool definitions
            **kwargs: Additional parameters

        Yields:
            Chunks of the response
        """
        # Default implementation: just yield the full response
        response = await self.generate_async(messages, tools, **kwargs)
        yield response

    def count_tokens(self, text: str) -> int:
        """Count tokens in text (approximate).

        Args:
            text: The text to count tokens for

        Returns:
            Approximate token count
        """
        # Simple approximation: ~4 chars per token for English
        return len(text) // 4

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_id='{self.model_id}')"
