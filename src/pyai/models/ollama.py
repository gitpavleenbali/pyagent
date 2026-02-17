# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Ollama Model Provider

Support for local models via Ollama.
Enables offline/private AI agent deployment.

Requires: ollama installed locally
Install: https://ollama.ai/download
"""

import os
from typing import Any, AsyncIterator, Dict, List, Optional

from .base import (
    BaseModel,
    Message,
    ModelCapability,
    ModelConfig,
    ModelResponse,
    Usage,
)


class OllamaModel(BaseModel):
    """Ollama local model provider.

    Run AI models locally without API keys or internet connection.
    Perfect for privacy-sensitive applications or offline use.

    Example:
        # Use llama3.2 locally
        model = OllamaModel(model_id="llama3.2")

        # Use a custom model
        model = OllamaModel(model_id="codellama:13b")

        # Custom host
        model = OllamaModel(
            model_id="llama3.2",
            host="http://my-server:11434"
        )
    """

    # Popular Ollama models for reference
    POPULAR_MODELS = [
        "llama3.2",  # Meta's latest
        "llama3.2:1b",  # Lightweight variant
        "llama3.2:3b",  # Balanced variant
        "codellama",  # Code-focused
        "mistral",  # Mistral AI
        "mixtral",  # Mixture of experts
        "phi3",  # Microsoft Phi-3
        "gemma2",  # Google Gemma 2
        "qwen2.5",  # Alibaba Qwen
        "deepseek-coder",  # DeepSeek for code
    ]

    def __init__(
        self,
        model_id: str = "llama3.2",
        host: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        **kwargs,
    ):
        """Initialize Ollama model.

        Args:
            model_id: Ollama model name (e.g., "llama3.2", "codellama:13b")
            host: Ollama server URL (default: http://localhost:11434)
            config: Model configuration
            **kwargs: Additional configuration
        """
        config = config or ModelConfig(model_id=model_id)
        config.model_id = model_id
        super().__init__(config, **kwargs)

        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client = None
        self._async_client = None

    @property
    def provider(self) -> str:
        return "ollama"

    @property
    def capabilities(self) -> List[ModelCapability]:
        caps = [
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
            ModelCapability.COMPLETION,
        ]
        # Some models support function calling
        if any(m in self.model_id.lower() for m in ["llama3", "mistral", "mixtral"]):
            caps.append(ModelCapability.FUNCTION_CALLING)
        # Vision models
        if any(m in self.model_id.lower() for m in ["llava", "bakllava", "moondream"]):
            caps.append(ModelCapability.VISION)
        return caps

    def _get_client(self):
        """Get or create the Ollama client."""
        if self._client is None:
            try:
                import ollama

                self._client = ollama.Client(host=self.host)
            except ImportError:
                raise ImportError(
                    "ollama package required. Install with: pip install ollama\n"
                    "Also ensure Ollama is installed: https://ollama.ai/download"
                )
        return self._client

    async def _get_async_client(self):
        """Get or create the async Ollama client."""
        if self._async_client is None:
            try:
                import ollama

                self._async_client = ollama.AsyncClient(host=self.host)
            except ImportError:
                raise ImportError("ollama package required. Install with: pip install ollama")
        return self._async_client

    def _prepare_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to Ollama format."""
        result = []
        for msg in messages:
            ollama_msg = {
                "role": msg.role,
                "content": msg.content,
            }
            result.append(ollama_msg)
        return result

    def _parse_response(self, response: Dict[str, Any]) -> ModelResponse:
        """Parse Ollama response to ModelResponse."""
        message = response.get("message", {})

        usage = None
        if "prompt_eval_count" in response or "eval_count" in response:
            usage = Usage(
                prompt_tokens=response.get("prompt_eval_count", 0),
                completion_tokens=response.get("eval_count", 0),
                total_tokens=response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
            )

        return ModelResponse(
            content=message.get("content", ""),
            role=message.get("role", "assistant"),
            usage=usage,
            model=response.get("model", self.model_id),
            finish_reason="stop" if response.get("done") else None,
            raw_response=response,
        )

    def generate(
        self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs
    ) -> ModelResponse:
        """Generate a response using Ollama."""
        client = self._get_client()

        options = {
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        if self.config.max_tokens:
            options["num_predict"] = self.config.max_tokens

        params = {
            "model": self.model_id,
            "messages": self._prepare_messages(messages),
            "options": options,
        }

        # Add tools if supported
        if tools and self.supports(ModelCapability.FUNCTION_CALLING):
            params["tools"] = tools

        response = client.chat(**params)
        return self._parse_response(response)

    async def generate_async(
        self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs
    ) -> ModelResponse:
        """Generate a response asynchronously."""
        client = await self._get_async_client()

        options = {
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        if self.config.max_tokens:
            options["num_predict"] = self.config.max_tokens

        params = {
            "model": self.model_id,
            "messages": self._prepare_messages(messages),
            "options": options,
        }

        if tools and self.supports(ModelCapability.FUNCTION_CALLING):
            params["tools"] = tools

        response = await client.chat(**params)
        return self._parse_response(response)

    def stream(
        self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs
    ):
        """Stream a response from Ollama."""
        client = self._get_client()

        options = {
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        params = {
            "model": self.model_id,
            "messages": self._prepare_messages(messages),
            "options": options,
            "stream": True,
        }

        response = client.chat(**params)

        for chunk in response:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]

    async def stream_async(
        self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs
    ) -> AsyncIterator[str]:
        """Stream a response asynchronously."""
        client = await self._get_async_client()

        options = {
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        params = {
            "model": self.model_id,
            "messages": self._prepare_messages(messages),
            "options": options,
            "stream": True,
        }

        response = await client.chat(**params)

        async for chunk in response:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]

    def pull(self, model_id: Optional[str] = None) -> None:
        """Pull/download a model from Ollama registry.

        Args:
            model_id: Model to pull (default: self.model_id)
        """
        client = self._get_client()
        model = model_id or self.model_id
        print(f"Pulling model '{model}'...")
        client.pull(model)
        print(f"Model '{model}' ready!")

    def list_models(self) -> List[str]:
        """List locally available models."""
        client = self._get_client()
        response = client.list()
        return [model["name"] for model in response.get("models", [])]

    @classmethod
    def is_available(cls, host: Optional[str] = None) -> bool:
        """Check if Ollama server is running.

        Args:
            host: Ollama server URL to check

        Returns:
            True if Ollama is available
        """
        try:
            import ollama

            client = ollama.Client(host=host or "http://localhost:11434")
            client.list()
            return True
        except Exception:
            return False
