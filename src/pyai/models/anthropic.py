# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Anthropic Claude Model Provider

Support for Claude models via Anthropic API.
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


class AnthropicModel(BaseModel):
    """Anthropic Claude model provider.

    Example:
        model = AnthropicModel(model_id="claude-sonnet-4-20250514")
        response = model.generate([Message(role="user", content="Hello!")])
    """

    MODELS = [
        "claude-sonnet-4-20250514",  # Latest Claude 4 Sonnet
        "claude-opus-4-20250514",  # Claude 4 Opus
        "claude-3-5-sonnet-20241022",  # Claude 3.5 Sonnet
        "claude-3-5-haiku-20241022",  # Claude 3.5 Haiku (fast)
    ]

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        **kwargs,
    ):
        config = config or ModelConfig(model_id=model_id)
        config.model_id = model_id
        super().__init__(config, **kwargs)

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None
        self._async_client = None

    @property
    def provider(self) -> str:
        return "anthropic"

    @property
    def capabilities(self) -> List[ModelCapability]:
        return [
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION,
            ModelCapability.STRUCTURED_OUTPUT,
        ]

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")
        return self._client

    async def _get_async_client(self):
        if self._async_client is None:
            try:
                import anthropic

                self._async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")
        return self._async_client

    def _prepare_messages(self, messages: List[Message]) -> tuple:
        """Convert messages to Anthropic format (separate system prompt)."""
        system_prompt = None
        api_messages = []

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                api_messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                    }
                )

        return system_prompt, api_messages

    def _parse_response(self, response: Any) -> ModelResponse:
        """Parse Anthropic response."""
        content = ""
        tool_calls = []

        for block in response.content:
            if hasattr(block, "text"):
                content += block.text
            elif hasattr(block, "type") and block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        usage = Usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        return ModelResponse(
            content=content,
            role="assistant",
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            model=response.model,
            finish_reason=response.stop_reason,
            raw_response=response,
        )

    def generate(
        self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs
    ) -> ModelResponse:
        client = self._get_client()

        system_prompt, api_messages = self._prepare_messages(messages)

        params = {
            "model": self.model_id,
            "messages": api_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens or 4096),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        if system_prompt:
            params["system"] = system_prompt

        if tools:
            # Convert OpenAI-style tools to Anthropic format
            params["tools"] = self._convert_tools(tools)

        response = client.messages.create(**params)
        return self._parse_response(response)

    async def generate_async(
        self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs
    ) -> ModelResponse:
        client = await self._get_async_client()

        system_prompt, api_messages = self._prepare_messages(messages)

        params = {
            "model": self.model_id,
            "messages": api_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens or 4096),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        if system_prompt:
            params["system"] = system_prompt

        if tools:
            params["tools"] = self._convert_tools(tools)

        response = await client.messages.create(**params)
        return self._parse_response(response)

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style tools to Anthropic format."""
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    }
                )
            else:
                anthropic_tools.append(tool)
        return anthropic_tools

    def stream(self, messages: List[Message], tools: Optional[List[Dict]] = None, **kwargs):
        client = self._get_client()

        system_prompt, api_messages = self._prepare_messages(messages)

        params = {
            "model": self.model_id,
            "messages": api_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens or 4096),
            "stream": True,
        }

        if system_prompt:
            params["system"] = system_prompt

        with client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                yield text

    async def stream_async(
        self, messages: List[Message], tools: Optional[List[Dict]] = None, **kwargs
    ) -> AsyncIterator[str]:
        client = await self._get_async_client()

        system_prompt, api_messages = self._prepare_messages(messages)

        params = {
            "model": self.model_id,
            "messages": api_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens or 4096),
        }

        if system_prompt:
            params["system"] = system_prompt

        async with client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text
