# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Tests for the models module.

Tests multi-provider LLM support including:
- Model registry
- Azure OpenAI
- OpenAI
- Ollama (local models)
- Anthropic
- Gemini
- LiteLLM
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os


class TestModelRegistry:
    """Tests for ModelRegistry."""
    
    def test_registry_import(self):
        """Test that ModelRegistry can be imported."""
        from pyagent.models import ModelRegistry
        assert ModelRegistry is not None
    
    def test_registry_singleton(self):
        """Test that registry uses singleton pattern."""
        from pyagent.models.registry import ModelRegistry
        
        r1 = ModelRegistry()
        r2 = ModelRegistry()
        # They should share the same internal state
        assert r1._providers == r2._providers
    
    def test_registry_list_providers(self):
        """Test listing available providers."""
        from pyagent.models import ModelRegistry
        
        providers = ModelRegistry.list_providers()
        
        assert isinstance(providers, list)
        assert "azure_openai" in providers or "azure" in providers
        assert "openai" in providers
        assert "ollama" in providers
        assert "anthropic" in providers
        assert "gemini" in providers or "google" in providers
        assert "litellm" in providers
    
    def test_get_model_with_provider(self):
        """Test get_model with explicit provider."""
        from pyagent.models import get_model
        from pyagent.models.openai import OpenAIModel
        
        # Mock the API key
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            model = get_model(provider="openai", model_id="gpt-4o")
            assert isinstance(model, OpenAIModel)
    
    def test_get_model_from_string(self):
        """Test registry from_string method."""
        from pyagent.models.registry import ModelRegistry
        from pyagent.models.ollama import OllamaModel
        
        model = ModelRegistry.from_string("ollama/llama3.2")
        assert isinstance(model, OllamaModel)
    
    def test_get_model_mock(self):
        """Test mock model for testing."""
        from pyagent.models import get_model
        
        model = get_model(provider="mock")
        assert model is not None
        
        # Mock should work without API keys
        response = model.generate([{"role": "user", "content": "Test"}])
        assert response is not None


class TestBaseModel:
    """Tests for BaseModel and related classes."""
    
    def test_message_class(self):
        """Test Message dataclass."""
        from pyagent.models.base import Message
        
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
    
    def test_message_to_dict(self):
        """Test Message to_dict conversion."""
        from pyagent.models.base import Message
        
        msg = Message(role="assistant", content="Hi there")
        d = msg.to_dict()
        
        assert d["role"] == "assistant"
        assert d["content"] == "Hi there"
    
    def test_model_config(self):
        """Test ModelConfig dataclass."""
        from pyagent.models.base import ModelConfig
        
        config = ModelConfig(
            model_id="gpt-4o",
            temperature=0.7,
            max_tokens=1000
        )
        
        assert config.model_id == "gpt-4o"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
    
    def test_model_response(self):
        """Test ModelResponse dataclass."""
        from pyagent.models.base import ModelResponse, Usage
        
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        response = ModelResponse(
            content="Hello!",
            model="gpt-4o",
            usage=usage
        )
        
        assert response.content == "Hello!"
        assert response.model == "gpt-4o"
        assert response.usage.total_tokens == 30
    
    def test_tool_call(self):
        """Test ToolCall dataclass."""
        from pyagent.models.base import ToolCall
        
        tc = ToolCall(
            id="call-123",
            name="get_weather",
            arguments={"city": "NYC"}
        )
        
        assert tc.id == "call-123"
        assert tc.name == "get_weather"
        assert tc.arguments["city"] == "NYC"
    
    def test_model_capability_enum(self):
        """Test ModelCapability enum."""
        from pyagent.models.base import ModelCapability
        
        assert ModelCapability.CHAT is not None
        assert ModelCapability.VISION is not None
        assert ModelCapability.FUNCTION_CALLING is not None
        assert ModelCapability.STREAMING is not None


class TestAzureOpenAIModel:
    """Tests for Azure OpenAI model."""
    
    def test_import(self):
        """Test that AzureOpenAIModel can be imported."""
        from pyagent.models.azure_openai import AzureOpenAIModel
        assert AzureOpenAIModel is not None
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        from pyagent.models.azure_openai import AzureOpenAIModel
        
        model = AzureOpenAIModel(
            deployment="gpt-4o",
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            api_version="2024-02-15-preview"
        )
        
        assert model.deployment == "gpt-4o"
    
    def test_capabilities(self):
        """Test model capabilities."""
        from pyagent.models.azure_openai import AzureOpenAIModel
        from pyagent.models.base import ModelCapability
        
        model = AzureOpenAIModel(
            deployment="gpt-4o",
            endpoint="https://test.openai.azure.com",
            api_key="test-key"
        )
        
        caps = model.capabilities
        assert ModelCapability.CHAT in caps
        assert ModelCapability.FUNCTION_CALLING in caps


class TestOpenAIModel:
    """Tests for direct OpenAI model."""
    
    def test_import(self):
        """Test that OpenAIModel can be imported."""
        from pyagent.models.openai import OpenAIModel
        assert OpenAIModel is not None
    
    def test_init(self):
        """Test initialization."""
        from pyagent.models.openai import OpenAIModel
        from pyagent.models.base import ModelConfig
        
        config = ModelConfig(model_id="gpt-4o-mini")
        model = OpenAIModel(config=config, api_key="test-key")
        assert model.model_id == "gpt-4o-mini"


class TestOllamaModel:
    """Tests for Ollama local models."""
    
    def test_import(self):
        """Test that OllamaModel can be imported."""
        from pyagent.models.ollama import OllamaModel
        assert OllamaModel is not None
    
    def test_init_defaults(self):
        """Test initialization with defaults."""
        from pyagent.models.ollama import OllamaModel
        
        model = OllamaModel(model_id="llama3.2")
        
        assert model.model_id == "llama3.2"
        assert model.host == "http://localhost:11434"
    
    def test_init_custom_url(self):
        """Test initialization with custom host URL."""
        from pyagent.models.ollama import OllamaModel
        
        model = OllamaModel(
            model_id="codellama",
            host="http://my-server:11434"
        )
        
        assert model.host == "http://my-server:11434"
    
    def test_supported_models(self):
        """Test known Ollama models."""
        from pyagent.models.ollama import OllamaModel
        
        # Should not raise
        for model_name in ["llama3.2", "codellama", "mistral", "phi3"]:
            model = OllamaModel(model_id=model_name)
            assert model.model_id == model_name


class TestAnthropicModel:
    """Tests for Anthropic Claude models."""
    
    def test_import(self):
        """Test that AnthropicModel can be imported."""
        from pyagent.models.anthropic import AnthropicModel
        assert AnthropicModel is not None
    
    def test_init(self):
        """Test initialization."""
        from pyagent.models.anthropic import AnthropicModel
        
        model = AnthropicModel(
            model_id="claude-3-5-sonnet-20241022",
            api_key="test-key"
        )
        
        assert "claude" in model.model_id


class TestGeminiModel:
    """Tests for Google Gemini models."""
    
    def test_import(self):
        """Test that GeminiModel can be imported."""
        from pyagent.models.gemini import GeminiModel
        assert GeminiModel is not None
    
    def test_init(self):
        """Test initialization."""
        from pyagent.models.gemini import GeminiModel
        
        model = GeminiModel(
            model_id="gemini-2.5-flash",
            api_key="test-key"
        )
        
        assert "gemini" in model.model_id


class TestLiteLLMModel:
    """Tests for LiteLLM universal provider."""
    
    def test_import(self):
        """Test that LiteLLMModel can be imported."""
        from pyagent.models.litellm import LiteLLMModel
        assert LiteLLMModel is not None
    
    def test_init(self):
        """Test initialization."""
        from pyagent.models.litellm import LiteLLMModel
        
        model = LiteLLMModel(model_id="gpt-4o")
        assert model.model_id == "gpt-4o"
    
    def test_provider_prefix(self):
        """Test provider prefix handling."""
        from pyagent.models.litellm import LiteLLMModel
        
        # Various provider formats
        model1 = LiteLLMModel(model_id="openai/gpt-4o")
        model2 = LiteLLMModel(model_id="anthropic/claude-3-opus")
        model3 = LiteLLMModel(model_id="azure/gpt-4")
        
        assert model1 is not None
        assert model2 is not None
        assert model3 is not None


class TestMockModel:
    """Tests for mock model used in testing."""
    
    def test_mock_generate(self):
        """Test mock model generation."""
        from pyagent.models import get_model
        
        model = get_model(provider="mock")
        response = model.generate([{"role": "user", "content": "Hello"}])
        
        assert response is not None
        assert hasattr(response, "content")
        assert isinstance(response.content, str)
    
    def test_mock_with_messages(self):
        """Test mock model with message list."""
        from pyagent.models import get_model
        from pyagent.models.base import Message
        
        model = get_model(provider="mock")
        messages = [
            Message(role="user", content="Hello"),
        ]
        
        response = model.generate(messages)
        assert response is not None


class TestModelIntegration:
    """Integration tests for model module."""
    
    def test_get_model_function(self):
        """Test the get_model convenience function."""
        from pyagent.models import get_model
        
        # Should work with mock
        model = get_model(provider="mock")
        assert model is not None
    
    def test_module_exports(self):
        """Test that all expected exports are available."""
        from pyagent import models
        
        # Check main exports
        assert hasattr(models, "get_model")
        assert hasattr(models, "ModelRegistry")
        assert hasattr(models, "BaseModel")
        assert hasattr(models, "AzureOpenAIModel")
        assert hasattr(models, "OpenAIModel")
        assert hasattr(models, "OllamaModel")
        assert hasattr(models, "AnthropicModel")
        assert hasattr(models, "GeminiModel")
        assert hasattr(models, "LiteLLMModel")
    
    def test_main_init_exports(self):
        """Test that models is exported from main pyagent."""
        import pyagent
        
        # Should be able to access models module
        assert hasattr(pyagent, "models")
        assert hasattr(pyagent, "get_model")
