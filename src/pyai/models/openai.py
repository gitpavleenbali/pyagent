# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
OpenAI Model Provider

Direct OpenAI API integration.
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


class OpenAIModel(BaseModel):
    """OpenAI API model provider.

    Example:
        model = OpenAIModel(model_id="gpt-4o")
        response = model.generate([Message(role="user", content="Hello!")])
    """

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        **kwargs,
    ):
        config = config or ModelConfig(model_id=model_id)
        config.model_id = model_id
        super().__init__(config, **kwargs)

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.organization = organization or os.environ.get("OPENAI_ORG_ID")
        self.base_url = base_url

        self._client = None
        self._async_client = None

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def capabilities(self) -> List[ModelCapability]:
        caps = [
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.JSON_MODE,
            ModelCapability.STRUCTURED_OUTPUT,
        ]
        # Add vision for vision-capable models
        if "vision" in self.model_id or "4o" in self.model_id or "4-turbo" in self.model_id:
            caps.append(ModelCapability.VISION)
        return caps

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")

            self._client = OpenAI(
                api_key=self.api_key,
                organization=self.organization,
                base_url=self.base_url,
            )
        return self._client

    async def _get_async_client(self):
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")

            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization,
                base_url=self.base_url,
            )
        return self._async_client

    def _prepare_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        return [msg.to_dict() for msg in messages]

    def _parse_response(self, response: Any) -> ModelResponse:
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
        client = self._get_client()

        params = {
            "model": self.model_id,
            "messages": self._prepare_messages(messages),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        if tools:
            params["tools"] = tools

        response = client.chat.completions.create(**params)
        return self._parse_response(response)

    async def generate_async(
        self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs
    ) -> ModelResponse:
        client = await self._get_async_client()

        params = {
            "model": self.model_id,
            "messages": self._prepare_messages(messages),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        if tools:
            params["tools"] = tools

        response = await client.chat.completions.create(**params)
        return self._parse_response(response)

    def stream(self, messages: List[Message], tools: Optional[List[Dict]] = None, **kwargs):
        client = self._get_client()

        params = {
            "model": self.model_id,
            "messages": self._prepare_messages(messages),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True,
        }

        response = client.chat.completions.create(**params)

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def stream_async(
        self, messages: List[Message], tools: Optional[List[Dict]] = None, **kwargs
    ) -> AsyncIterator[str]:
        client = await self._get_async_client()

        params = {
            "model": self.model_id,
            "messages": self._prepare_messages(messages),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True,
        }

        response = await client.chat.completions.create(**params)

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
