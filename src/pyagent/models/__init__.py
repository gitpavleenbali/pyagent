# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
PyAgent Models Module - Multi-Provider LLM Support

Inspired by Google ADK's models/ module, this provides a unified interface
for multiple LLM providers including:

- Azure OpenAI (default, enterprise-grade)
- OpenAI (direct API)
- Ollama (local models)
- Anthropic (Claude)
- Google Gemini
- LiteLLM (100+ models)
- Custom providers

Example:
    from pyagent.models import AzureOpenAIModel, OllamaModel, get_model
    
    # Use Azure OpenAI (default)
    model = AzureOpenAIModel(deployment="gpt-4o")
    
    # Use local Ollama
    model = OllamaModel(model_id="llama3.2")
    
    # Auto-detect from environment
    model = get_model()
"""

from .base import BaseModel, ModelConfig, ModelResponse
from .azure_openai import AzureOpenAIModel
from .openai import OpenAIModel
from .ollama import OllamaModel
from .anthropic import AnthropicModel
from .gemini import GeminiModel
from .litellm import LiteLLMModel
from .registry import ModelRegistry, get_model, register_model

__all__ = [
    # Base
    "BaseModel",
    "ModelConfig", 
    "ModelResponse",
    # Providers
    "AzureOpenAIModel",
    "OpenAIModel",
    "OllamaModel",
    "AnthropicModel",
    "GeminiModel",
    "LiteLLMModel",
    # Registry
    "ModelRegistry",
    "get_model",
    "register_model",
]
