# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Azure OpenAI Model Provider

First-class support for Azure OpenAI Service with Azure AD authentication.
"""

import os
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


class AzureOpenAIModel(BaseModel):
    """Azure OpenAI Service model provider.

    Features:
    - Azure AD / Managed Identity authentication
    - API key authentication
    - Full OpenAI API compatibility
    - Enterprise-grade security

    Example:
        # Using Azure AD (recommended)
        model = AzureOpenAIModel(
            deployment="gpt-4o",
            endpoint="https://my-resource.openai.azure.com"
        )

        # Using API key
        model = AzureOpenAIModel(
            deployment="gpt-4o",
            endpoint="https://my-resource.openai.azure.com",
            api_key="your-api-key"
        )
    """

    def __init__(
        self,
        deployment: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-08-01-preview",
        use_azure_ad: bool = True,
        config: Optional[ModelConfig] = None,
        **kwargs,
    ):
        """Initialize Azure OpenAI model.

        Args:
            deployment: Azure OpenAI deployment name
            endpoint: Azure OpenAI endpoint URL
            api_key: API key (optional if using Azure AD)
            api_version: API version to use
            use_azure_ad: Whether to use Azure AD authentication
            config: Model configuration
            **kwargs: Additional configuration
        """
        super().__init__(config, **kwargs)

        self.deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        self.endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.api_version = api_version
        self.use_azure_ad = use_azure_ad and not self.api_key

        self._client = None
        self._async_client = None

    @property
    def provider(self) -> str:
        return "azure"

    @property
    def capabilities(self) -> List[ModelCapability]:
        return [
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.JSON_MODE,
            ModelCapability.STRUCTURED_OUTPUT,
            ModelCapability.VISION,
        ]

    def _get_client(self):
        """Get or create the sync client."""
        if self._client is None:
            try:
                from openai import AzureOpenAI
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")

            if self.use_azure_ad:
                try:
                    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

                    credential = DefaultAzureCredential()
                    token_provider = get_bearer_token_provider(
                        credential, "https://cognitiveservices.azure.com/.default"
                    )
                    self._client = AzureOpenAI(
                        azure_endpoint=self.endpoint,
                        azure_ad_token_provider=token_provider,
                        api_version=self.api_version,
                    )
                except ImportError:
                    raise ImportError(
                        "azure-identity required for Azure AD auth. "
                        "Install with: pip install azure-identity"
                    )
            else:
                self._client = AzureOpenAI(
                    azure_endpoint=self.endpoint,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
        return self._client

    async def _get_async_client(self):
        """Get or create the async client."""
        if self._async_client is None:
            try:
                from openai import AsyncAzureOpenAI
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")

            if self.use_azure_ad:
                try:
                    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

                    credential = DefaultAzureCredential()
                    token_provider = get_bearer_token_provider(
                        credential, "https://cognitiveservices.azure.com/.default"
                    )
                    self._async_client = AsyncAzureOpenAI(
                        azure_endpoint=self.endpoint,
                        azure_ad_token_provider=token_provider,
                        api_version=self.api_version,
                    )
                except ImportError:
                    raise ImportError(
                        "azure-identity required for Azure AD auth. "
                        "Install with: pip install azure-identity"
                    )
            else:
                self._async_client = AsyncAzureOpenAI(
                    azure_endpoint=self.endpoint,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
        return self._async_client

    def _prepare_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert Message objects to API format."""
        return [msg.to_dict() for msg in messages]

    def _parse_response(self, response: Any) -> ModelResponse:
        """Parse API response to ModelResponse."""
        choice = response.choices[0]
        message = choice.message

        tool_calls = None
        if message.tool_calls:
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
        """Generate a response synchronously."""
        client = self._get_client()

        api_messages = self._prepare_messages(messages)

        params = {
            "model": self.deployment,
            "messages": api_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }

        if tools:
            params["tools"] = tools

        if self.config.stop:
            params["stop"] = self.config.stop

        response = client.chat.completions.create(**params)
        return self._parse_response(response)

    async def generate_async(
        self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs
    ) -> ModelResponse:
        """Generate a response asynchronously."""
        client = await self._get_async_client()

        api_messages = self._prepare_messages(messages)

        params = {
            "model": self.deployment,
            "messages": api_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }

        if tools:
            params["tools"] = tools

        if self.config.stop:
            params["stop"] = self.config.stop

        response = await client.chat.completions.create(**params)
        return self._parse_response(response)

    def stream(
        self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs
    ):
        """Stream a response synchronously."""
        client = self._get_client()

        api_messages = self._prepare_messages(messages)

        params = {
            "model": self.deployment,
            "messages": api_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True,
        }

        if tools:
            params["tools"] = tools

        response = client.chat.completions.create(**params)

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def stream_async(
        self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs
    ) -> AsyncIterator[str]:
        """Stream a response asynchronously."""
        client = await self._get_async_client()

        api_messages = self._prepare_messages(messages)

        params = {
            "model": self.deployment,
            "messages": api_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True,
        }

        if tools:
            params["tools"] = tools

        response = await client.chat.completions.create(**params)

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
