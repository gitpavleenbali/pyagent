"""
LLM - Language Model Provider interfaces and implementations
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class ModelProvider(Enum):
    """Supported model providers"""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    CUSTOM = "custom"


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    
    provider: ModelProvider = ModelProvider.OPENAI
    model: str = "gpt-4"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    
    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 60.0


@dataclass
class LLMResponse:
    """Response from an LLM call"""
    
    content: str
    finish_reason: str = "stop"
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Implement this interface to add support for different
    language model providers (OpenAI, Anthropic, local models, etc.)
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
    
    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Generate a completion from the model.
        
        Args:
            system_prompt: The system instruction
            messages: List of conversation messages
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    async def complete_with_tools(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion with tool/function calling support.
        
        Args:
            system_prompt: The system instruction
            messages: List of conversation messages
            tools: List of tool definitions
            **kwargs: Additional arguments
            
        Returns:
            LLMResponse with content and tool calls
        """
        pass
    
    @abstractmethod
    async def stream(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        **kwargs
    ):
        """
        Stream responses from the model.
        
        Yields chunks of the response as they're generated.
        """
        pass
    
    def validate_config(self) -> bool:
        """Validate provider configuration"""
        if not self.config.model:
            return False
        return True


class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider implementation.
    
    Example:
        >>> provider = OpenAIProvider(LLMConfig(
        ...     api_key="sk-...",
        ...     model="gpt-4-turbo",
        ... ))
        >>> response = await provider.complete(
        ...     system_prompt="You are helpful",
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        self._client = None
    
    async def _get_client(self):
        """Lazy initialization of OpenAI client"""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.api_base,
                    timeout=self.config.timeout,
                )
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        return self._client
    
    async def complete(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        client = await self._get_client()
        
        # Prepare messages with system prompt
        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)
        
        response = await client.chat.completions.create(
            model=self.config.model,
            messages=full_messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            top_p=kwargs.get("top_p", self.config.top_p),
        )
        
        return response.choices[0].message.content
    
    async def complete_with_tools(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        **kwargs
    ) -> LLMResponse:
        client = await self._get_client()
        
        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)
        
        response = await client.chat.completions.create(
            model=self.config.model,
            messages=full_messages,
            tools=tools,
            tool_choice=kwargs.get("tool_choice", "auto"),
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            model=response.model,
            metadata={"tool_calls": choice.message.tool_calls},
        )
    
    async def stream(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        **kwargs
    ):
        client = await self._get_client()
        
        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)
        
        stream = await client.chat.completions.create(
            model=self.config.model,
            messages=full_messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AzureOpenAIProvider(LLMProvider):
    """
    Azure OpenAI Service provider implementation.
    
    Supports both API Key and Azure AD (DefaultAzureCredential) authentication.
    If api_key is not provided, will use DefaultAzureCredential.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        self._client = None
    
    async def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncAzureOpenAI
                
                # Check if we should use Azure AD auth
                if not self.config.api_key:
                    try:
                        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
                        credential = DefaultAzureCredential()
                        token_provider = get_bearer_token_provider(
                            credential, "https://cognitiveservices.azure.com/.default"
                        )
                        self._client = AsyncAzureOpenAI(
                            azure_ad_token_provider=token_provider,
                            api_version=self.config.api_version or "2024-02-01",
                            azure_endpoint=self.config.api_base,
                            timeout=self.config.timeout,
                        )
                    except ImportError:
                        raise ImportError(
                            "azure-identity required for Azure AD auth. "
                            "Install with: pip install azure-identity"
                        )
                else:
                    # Use API key authentication
                    self._client = AsyncAzureOpenAI(
                        api_key=self.config.api_key,
                        api_version=self.config.api_version or "2024-02-01",
                        azure_endpoint=self.config.api_base,
                        timeout=self.config.timeout,
                    )
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        return self._client
    
    async def complete(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        client = await self._get_client()
        
        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)
        
        response = await client.chat.completions.create(
            model=self.config.model,  # deployment name in Azure
            messages=full_messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        
        return response.choices[0].message.content
    
    async def complete_with_tools(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        **kwargs
    ) -> LLMResponse:
        # Similar to OpenAI implementation
        client = await self._get_client()
        
        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)
        
        response = await client.chat.completions.create(
            model=self.config.model,
            messages=full_messages,
            tools=tools,
            tool_choice=kwargs.get("tool_choice", "auto"),
        )
        
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            model=response.model,
        )
    
    async def stream(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        **kwargs
    ):
        client = await self._get_client()
        
        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)
        
        stream = await client.chat.completions.create(
            model=self.config.model,
            messages=full_messages,
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        self._client = None
    
    async def _get_client(self):
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(
                    api_key=self.config.api_key,
                )
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")
        return self._client
    
    async def complete(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        client = await self._get_client()
        
        response = await client.messages.create(
            model=self.config.model,
            system=system_prompt,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
        )
        
        return response.content[0].text
    
    async def complete_with_tools(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        **kwargs
    ) -> LLMResponse:
        client = await self._get_client()
        
        response = await client.messages.create(
            model=self.config.model,
            system=system_prompt,
            messages=messages,
            tools=tools,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        
        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append(block)
        
        return LLMResponse(
            content=content,
            finish_reason=response.stop_reason,
            metadata={"tool_calls": tool_calls},
        )
    
    async def stream(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        **kwargs
    ):
        client = await self._get_client()
        
        async with client.messages.stream(
            model=self.config.model,
            system=system_prompt,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        ) as stream:
            async for text in stream.text_stream:
                yield text
