# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Google Gemini Model Provider

Support for Google's Gemini models.
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


class GeminiModel(BaseModel):
    """Google Gemini model provider.

    Example:
        model = GeminiModel(model_id="gemini-2.5-flash")
        response = model.generate([Message(role="user", content="Hello!")])
    """

    MODELS = [
        "gemini-2.5-flash",  # Latest and fastest
        "gemini-2.5-pro",  # Most capable
        "gemini-2.0-flash-exp",  # Experimental
        "gemini-1.5-flash",  # Fast and versatile
        "gemini-1.5-pro",  # Complex reasoning
    ]

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        **kwargs,
    ):
        config = config or ModelConfig(model_id=model_id)
        config.model_id = model_id
        super().__init__(config, **kwargs)

        self.api_key = (
            api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        )
        self._model = None

    @property
    def provider(self) -> str:
        return "gemini"

    @property
    def capabilities(self) -> List[ModelCapability]:
        return [
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION,
            ModelCapability.AUDIO,
            ModelCapability.STRUCTURED_OUTPUT,
        ]

    def _get_model(self):
        """Get or create the Gemini model."""
        if self._model is None:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "google-generativeai package required. "
                    "Install with: pip install google-generativeai"
                )

            genai.configure(api_key=self.api_key)

            generation_config = {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            }
            if self.config.max_tokens:
                generation_config["max_output_tokens"] = self.config.max_tokens

            self._model = genai.GenerativeModel(
                model_name=self.model_id,
                generation_config=generation_config,
            )
        return self._model

    def _prepare_messages(self, messages: List[Message]) -> tuple:
        """Convert messages to Gemini format."""
        system_instruction = None
        history = []
        current_content = None

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role == "user":
                current_content = msg.content
                if history or current_content != messages[-1].content:
                    history.append({"role": "user", "parts": [msg.content]})
            elif msg.role == "assistant":
                history.append({"role": "model", "parts": [msg.content]})

        # Last user message is the current input
        if messages and messages[-1].role == "user":
            current_content = messages[-1].content
            # Remove from history if it was added
            if history and history[-1]["role"] == "user":
                history = history[:-1]

        return system_instruction, history, current_content

    def _parse_response(self, response: Any) -> ModelResponse:
        """Parse Gemini response."""
        text = response.text if hasattr(response, "text") else str(response)

        usage = None
        if hasattr(response, "usage_metadata"):
            usage = Usage(
                prompt_tokens=getattr(response.usage_metadata, "prompt_token_count", 0),
                completion_tokens=getattr(response.usage_metadata, "candidates_token_count", 0),
                total_tokens=getattr(response.usage_metadata, "total_token_count", 0),
            )

        return ModelResponse(
            content=text,
            role="assistant",
            usage=usage,
            model=self.model_id,
            finish_reason="stop",
            raw_response=response,
        )

    def generate(
        self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs
    ) -> ModelResponse:
        model = self._get_model()

        system_instruction, history, current_content = self._prepare_messages(messages)

        # Start chat with history
        chat = model.start_chat(history=history)

        # Generate response
        response = chat.send_message(current_content or "")
        return self._parse_response(response)

    async def generate_async(
        self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs
    ) -> ModelResponse:
        model = self._get_model()

        system_instruction, history, current_content = self._prepare_messages(messages)

        chat = model.start_chat(history=history)
        response = await chat.send_message_async(current_content or "")
        return self._parse_response(response)

    def stream(self, messages: List[Message], tools: Optional[List[Dict]] = None, **kwargs):
        model = self._get_model()

        system_instruction, history, current_content = self._prepare_messages(messages)

        chat = model.start_chat(history=history)
        response = chat.send_message(current_content or "", stream=True)

        for chunk in response:
            if chunk.text:
                yield chunk.text

    async def stream_async(
        self, messages: List[Message], tools: Optional[List[Dict]] = None, **kwargs
    ) -> AsyncIterator[str]:
        model = self._get_model()

        system_instruction, history, current_content = self._prepare_messages(messages)

        chat = model.start_chat(history=history)
        response = await chat.send_message_async(current_content or "", stream=True)

        async for chunk in response:
            if chunk.text:
                yield chunk.text
