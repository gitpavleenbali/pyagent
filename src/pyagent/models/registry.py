# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Model Registry

Auto-detection and factory for model providers.
Similar to Google ADK's model registry pattern.
"""

import os
from typing import Any, Dict, Optional, Type

from .base import BaseModel, ModelConfig


class ModelRegistry:
    """Registry for model providers.
    
    Provides auto-detection of appropriate model provider based on
    environment variables and model ID patterns.
    
    Example:
        # Auto-detect based on environment
        model = ModelRegistry.get_model()
        
        # Get specific provider
        model = ModelRegistry.get_model(provider="ollama", model_id="llama3.2")
        
        # Register custom provider
        ModelRegistry.register("custom", MyCustomModel)
    """
    
    _providers: Dict[str, Type[BaseModel]] = {}
    _initialized: bool = False
    
    @classmethod
    def _ensure_initialized(cls):
        """Lazy initialization of built-in providers."""
        if cls._initialized:
            return
        
        # Import and register built-in providers
        from .azure_openai import AzureOpenAIModel
        from .openai import OpenAIModel
        from .ollama import OllamaModel
        from .anthropic import AnthropicModel
        from .gemini import GeminiModel
        from .litellm import LiteLLMModel
        
        cls._providers = {
            "azure": AzureOpenAIModel,
            "azure_openai": AzureOpenAIModel,
            "openai": OpenAIModel,
            "ollama": OllamaModel,
            "anthropic": AnthropicModel,
            "claude": AnthropicModel,
            "gemini": GeminiModel,
            "google": GeminiModel,
            "litellm": LiteLLMModel,
        }
        cls._initialized = True
    
    @classmethod
    def register(cls, name: str, provider_class: Type[BaseModel]) -> None:
        """Register a custom model provider.
        
        Args:
            name: Provider name (e.g., "custom")
            provider_class: BaseModel subclass
        """
        cls._ensure_initialized()
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def get_provider(cls, name: str) -> Optional[Type[BaseModel]]:
        """Get a registered provider class.
        
        Args:
            name: Provider name
            
        Returns:
            Provider class or None if not found
        """
        cls._ensure_initialized()
        return cls._providers.get(name.lower())
    
    @classmethod
    def list_providers(cls) -> list:
        """List all registered provider names."""
        cls._ensure_initialized()
        return list(set(cls._providers.keys()))
    
    @classmethod
    def detect_provider(cls) -> str:
        """Auto-detect the best available provider from environment.
        
        Detection order:
        1. PYAGENT_PROVIDER environment variable
        2. Azure OpenAI (if AZURE_OPENAI_ENDPOINT set)
        3. OpenAI (if OPENAI_API_KEY set)
        4. Anthropic (if ANTHROPIC_API_KEY set)
        5. Google (if GOOGLE_API_KEY set)
        6. Ollama (if running locally)
        7. Default to "mock" for testing
        
        Returns:
            Provider name string
        """
        # Check explicit configuration
        explicit = os.environ.get("PYAGENT_PROVIDER")
        if explicit:
            return explicit.lower()
        
        # Check for Azure OpenAI
        if os.environ.get("AZURE_OPENAI_ENDPOINT"):
            return "azure"
        
        # Check for OpenAI
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        
        # Check for Anthropic
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        
        # Check for Google
        if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            return "gemini"
        
        # Check for local Ollama
        try:
            from .ollama import OllamaModel
            if OllamaModel.is_available():
                return "ollama"
        except Exception:
            pass
        
        # Default: assume mock mode for testing
        return "mock"
    
    @classmethod
    def get_model(
        cls,
        provider: Optional[str] = None,
        model_id: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        **kwargs
    ) -> BaseModel:
        """Get a model instance.
        
        Args:
            provider: Provider name (auto-detected if not provided)
            model_id: Model identifier (uses provider default if not provided)
            config: Model configuration
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Configured model instance
            
        Example:
            # Auto-detect everything
            model = ModelRegistry.get_model()
            
            # Specify provider
            model = ModelRegistry.get_model(provider="ollama")
            
            # Full specification
            model = ModelRegistry.get_model(
                provider="azure",
                model_id="gpt-4o",
                deployment="my-deployment"
            )
        """
        cls._ensure_initialized()
        
        # Determine provider
        if provider is None:
            provider = cls.detect_provider()
        
        provider = provider.lower()
        
        # Handle mock mode
        if provider == "mock":
            return MockModel(model_id=model_id or "mock-model", config=config, **kwargs)
        
        # Get provider class
        provider_class = cls._providers.get(provider)
        if provider_class is None:
            available = ", ".join(cls.list_providers())
            raise ValueError(
                f"Unknown provider: '{provider}'. Available: {available}"
            )
        
        # Create model instance
        if model_id:
            if config:
                config.model_id = model_id
            else:
                config = ModelConfig(model_id=model_id)
        
        return provider_class(config=config, **kwargs)
    
    @classmethod
    def from_string(cls, model_string: str, **kwargs) -> BaseModel:
        """Create model from a string identifier.
        
        Supports formats:
        - "provider/model_id" (e.g., "ollama/llama3.2")
        - "provider:model_id" (e.g., "azure:gpt-4o")
        - "model_id" (provider auto-detected)
        
        Args:
            model_string: Model identifier string
            **kwargs: Additional arguments
            
        Returns:
            Model instance
        """
        # Parse provider/model format
        for sep in ["/", ":"]:
            if sep in model_string:
                parts = model_string.split(sep, 1)
                return cls.get_model(provider=parts[0], model_id=parts[1], **kwargs)
        
        # Auto-detect provider
        return cls.get_model(model_id=model_string, **kwargs)


class MockModel(BaseModel):
    """Mock model for testing without API calls."""
    
    def __init__(self, model_id: str = "mock-model", config: Optional[ModelConfig] = None, **kwargs):
        config = config or ModelConfig(model_id=model_id)
        config.model_id = model_id
        super().__init__(config, **kwargs)
        self.response_text = kwargs.get("response_text", "This is a mock response.")
    
    @property
    def provider(self) -> str:
        return "mock"
    
    def generate(self, messages, tools=None, **kwargs):
        from .base import ModelResponse, Usage
        return ModelResponse(
            content=self.response_text,
            role="assistant",
            usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            model=self.model_id,
            finish_reason="stop",
        )
    
    async def generate_async(self, messages, tools=None, **kwargs):
        return self.generate(messages, tools, **kwargs)


# Convenience functions
def get_model(
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
    **kwargs
) -> BaseModel:
    """Get a model instance (convenience function).
    
    Example:
        model = get_model()  # Auto-detect
        model = get_model("ollama", "llama3.2")  # Specific
    """
    return ModelRegistry.get_model(provider=provider, model_id=model_id, **kwargs)


def register_model(name: str, provider_class: Type[BaseModel]) -> None:
    """Register a custom model provider (convenience function)."""
    ModelRegistry.register(name, provider_class)
