# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
LiteLLM Model Provider

Unified interface for 100+ LLM providers via LiteLLM.
Supports OpenAI, Anthropic, Cohere, Replicate, HuggingFace, and more.
"""

from typing import Any, AsyncIterator, Dict, List, Optional

from .base import (
    BaseModel,
    Message,
    ModelCapability,
    ModelConfig,
    ModelResponse,
    ToolCall,
    Usage,
)


class LiteLLMModel(BaseModel):
    """LiteLLM unified model provider.

    Access 100+ LLM providers through a single interface.

    Supported providers include:
    - OpenAI, Azure OpenAI
    - Anthropic Claude
    - Google (Gemini, PaLM)
    - Cohere
    - Replicate
    - HuggingFace
    - AWS Bedrock
    - Ollama
    - vLLM
    - And many more...

    Example:
        # Use any provider with LiteLLM model format
        model = LiteLLMModel(model_id="gpt-4o")
        model = LiteLLMModel(model_id="claude-3-sonnet-20240229")
        model = LiteLLMModel(model_id="bedrock/anthropic.claude-v2")
        model = LiteLLMModel(model_id="ollama/llama3.2")
    """

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        **kwargs,
    ):
        """Initialize LiteLLM model.

        Args:
            model_id: LiteLLM model identifier (e.g., "gpt-4o", "claude-3-sonnet")
            api_key: API key (auto-detected from environment if not provided)
            api_base: Custom API base URL
            config: Model configuration
            **kwargs: Additional LiteLLM parameters
        """
        config = config or ModelConfig(model_id=model_id)
        config.model_id = model_id
        super().__init__(config, **kwargs)

        self.api_key = api_key
        self.api_base = api_base
        self.litellm_kwargs = kwargs

    @property
    def provider(self) -> str:
        return "litellm"

    @property
    def capabilities(self) -> List[ModelCapability]:
        # LiteLLM supports most capabilities depending on the underlying model
        return [
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION,
            ModelCapability.JSON_MODE,
        ]

    def _get_litellm(self):
        """Import and return litellm module."""
        try:
            import litellm

            return litellm
        except ImportError:
            raise ImportError("litellm package required. Install with: pip install litellm")

    def _prepare_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to LiteLLM format."""
        return [msg.to_dict() for msg in messages]

    def _parse_response(self, response: Any) -> ModelResponse:
        """Parse LiteLLM response."""
        choice = response.choices[0]
        message = choice.message

        tool_calls = None
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = [
                ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
                for tc in message.tool_calls
            ]

        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return ModelResponse(
            content=message.content or "",
            role=message.role,
            tool_calls=tool_calls,
            usage=usage,
            model=response.model,
            finish_reason=choice.finish_reason,
            raw_response=response,
        )

    def generate(
        self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs
    ) -> ModelResponse:
        litellm = self._get_litellm()

        params = {
            "model": self.model_id,
            "messages": self._prepare_messages(messages),
            "temperature": kwargs.get("temperature", self.config.temperature),
            **self.litellm_kwargs,
        }

        if self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens

        if self.api_key:
            params["api_key"] = self.api_key

        if self.api_base:
            params["api_base"] = self.api_base

        if tools:
            params["tools"] = tools

        response = litellm.completion(**params)
        return self._parse_response(response)

    async def generate_async(
        self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs
    ) -> ModelResponse:
        litellm = self._get_litellm()

        params = {
            "model": self.model_id,
            "messages": self._prepare_messages(messages),
            "temperature": kwargs.get("temperature", self.config.temperature),
            **self.litellm_kwargs,
        }

        if self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens

        if self.api_key:
            params["api_key"] = self.api_key

        if self.api_base:
            params["api_base"] = self.api_base

        if tools:
            params["tools"] = tools

        response = await litellm.acompletion(**params)
        return self._parse_response(response)

    def stream(self, messages: List[Message], tools: Optional[List[Dict]] = None, **kwargs):
        litellm = self._get_litellm()

        params = {
            "model": self.model_id,
            "messages": self._prepare_messages(messages),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True,
            **self.litellm_kwargs,
        }

        if self.api_key:
            params["api_key"] = self.api_key

        if self.api_base:
            params["api_base"] = self.api_base

        response = litellm.completion(**params)

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def stream_async(
        self, messages: List[Message], tools: Optional[List[Dict]] = None, **kwargs
    ) -> AsyncIterator[str]:
        litellm = self._get_litellm()

        params = {
            "model": self.model_id,
            "messages": self._prepare_messages(messages),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True,
            **self.litellm_kwargs,
        }

        if self.api_key:
            params["api_key"] = self.api_key

        if self.api_base:
            params["api_base"] = self.api_base

        response = await litellm.acompletion(**params)

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @staticmethod
    def list_providers() -> List[str]:
        """List supported LiteLLM providers."""
        return [
            "openai",
            "azure",
            "anthropic",
            "cohere",
            "replicate",
            "huggingface",
            "bedrock",
            "sagemaker",
            "ollama",
            "vllm",
            "together_ai",
            "ai21",
            "nlp_cloud",
            "aleph_alpha",
            "petals",
            "vertex_ai",
            "palm",
            "deepinfra",
            "perplexity",
            "groq",
            "mistral",
            "cloudflare",
            "voyage",
            "databricks",
        ]

    @staticmethod
    def model_cost(model_id: str) -> Dict[str, float]:
        """Get cost information for a model.

        Returns:
            Dict with 'input_cost_per_token' and 'output_cost_per_token'
        """
        try:
            import litellm

            return litellm.get_model_cost_map().get(model_id, {})
        except Exception:
            return {}
