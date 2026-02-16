"""
LLM Interface - Unified interface for all LLM providers.

This module provides a simple, unified interface to any LLM provider.
Automatically handles provider selection based on configuration.
"""

import os
import json
import time
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

from pyagent.easy.config import get_config


def _log_llm_call(model: str, messages: List[Dict], response_content: str, duration_ms: float, usage: Dict):
    """Log an LLM call to the tracer if enabled."""
    try:
        from pyagent.easy.trace import _tracer, TraceEvent
        if _tracer.enabled:
            event = TraceEvent(
                type="llm_call",
                message=f"LLM call to {model}",
                duration_ms=duration_ms,
                metadata={
                    "model": model,
                    "input_messages": len(messages),
                    "output_length": len(response_content) if response_content else 0,
                    "usage": usage
                }
            )
            # Add to current span or create a temporary one
            if _tracer._current_span:
                _tracer._current_span.events.append(event)
            else:
                # Create implicit span for standalone calls
                with _tracer.span("llm_call") as span:
                    span.events.append(event)
    except ImportError:
        pass  # trace module not available


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass  
class LLMResponse:
    """Response from an LLM."""
    content: str
    model: str
    usage: Dict[str, int]
    raw: Any = None
    
    def __str__(self):
        return self.content
    
    def __repr__(self):
        return f"LLMResponse(content='{self.content[:50]}...')"


class LLMInterface:
    """
    Unified interface for LLM providers.
    
    Supports:
        - OpenAI (GPT-4, GPT-4o, GPT-3.5, etc.)
        - Anthropic (Claude 3.5, Claude 3, etc.)
        - Azure OpenAI
        - Local models via Ollama
    """
    
    def __init__(
        self,
        model: str = None,
        provider: str = None,
        api_key: str = None,
        **kwargs
    ):
        config = get_config()
        self.model = model or config.model
        self.provider = provider or config.provider
        self.api_key = api_key or config.get_api_key()
        self.kwargs = kwargs
        
        self._client = None
        
    def _get_client(self):
        """Lazy initialization of the appropriate client."""
        if self._client is not None:
            return self._client
            
        if self.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai"
                )
                
        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "Anthropic package not installed. Install with: pip install anthropic"
                )
                
        elif self.provider == "azure":
            try:
                from openai import AzureOpenAI
                config = get_config()
                
                # Try Azure AD authentication first if no API key
                if not self.api_key:
                    try:
                        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
                        credential = DefaultAzureCredential()
                        token_provider = get_bearer_token_provider(
                            credential, "https://cognitiveservices.azure.com/.default"
                        )
                        self._client = AzureOpenAI(
                            azure_ad_token_provider=token_provider,
                            azure_endpoint=config.get_azure_endpoint(),
                            api_version=config.azure_api_version
                        )
                    except ImportError:
                        raise ImportError(
                            "Azure Identity package not installed. Install with: pip install azure-identity"
                        )
                else:
                    # Use API key authentication
                    self._client = AzureOpenAI(
                        api_key=self.api_key,
                        azure_endpoint=config.get_azure_endpoint(),
                        api_version=config.azure_api_version
                    )
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai"
                )
                
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
            
        return self._client
    
    def complete(
        self,
        prompt: str,
        system: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a completion for the given prompt.
        
        Args:
            prompt: The user prompt
            system: Optional system message
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional provider-specific arguments
            
        Returns:
            LLMResponse with the generated content
        """
        config = get_config()
        temperature = temperature if temperature is not None else config.temperature
        max_tokens = max_tokens or config.max_tokens
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response for a chat conversation.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            LLMResponse with the generated content
        """
        config = get_config()
        temperature = temperature if temperature is not None else config.temperature
        max_tokens = max_tokens or config.max_tokens
        
        client = self._get_client()
        start_time = time.time()
        
        if self.provider in ("openai", "azure"):
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            duration_ms = (time.time() - start_time) * 1000
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            content = response.choices[0].message.content
            
            # Log to tracer if enabled
            _log_llm_call(response.model, messages, content, duration_ms, usage)
            
            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                raw=response
            )
            
        elif self.provider == "anthropic":
            # Convert messages format for Anthropic
            system_msg = None
            anthropic_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    anthropic_messages.append(msg)
            
            response = client.messages.create(
                model=self.model,
                messages=anthropic_messages,
                system=system_msg or "",
                max_tokens=max_tokens,
                **kwargs
            )
            duration_ms = (time.time() - start_time) * 1000
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
            content = response.content[0].text
            
            # Log to tracer if enabled
            _log_llm_call(response.model, messages, content, duration_ms, usage)
            
            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                raw=response
            )
    
    def json(
        self,
        prompt: str,
        schema: Dict[str, Any] = None,
        system: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a JSON response.
        
        Args:
            prompt: The user prompt
            schema: Optional JSON schema for structured output
            system: Optional system message
            
        Returns:
            Parsed JSON dict
        """
        json_system = (system or "") + "\n\nRespond with valid JSON only. No markdown, no explanation."
        
        if schema:
            json_system += f"\n\nFollow this schema:\n{json.dumps(schema, indent=2)}"
        
        if self.provider in ("openai", "azure"):
            response = self.complete(
                prompt,
                system=json_system,
                response_format={"type": "json_object"},
                **kwargs
            )
        else:
            response = self.complete(prompt, system=json_system, **kwargs)
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())


# Default global LLM instance
_default_llm: Optional[LLMInterface] = None


def get_llm(**kwargs) -> LLMInterface:
    """Get the default LLM instance, creating it if necessary."""
    global _default_llm
    
    if kwargs:
        # Return a new instance with custom settings
        return LLMInterface(**kwargs)
    
    if _default_llm is None:
        _default_llm = LLMInterface()
    
    return _default_llm


def reset_llm():
    """Reset the default LLM instance."""
    global _default_llm
    _default_llm = None
