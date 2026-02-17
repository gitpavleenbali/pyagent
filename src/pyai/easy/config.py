"""
pyai Configuration - Zero-config by default, customizable when needed

Works out of the box with environment variables:
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY
    - AZURE_OPENAI_API_KEY
    - AZURE_OPENAI_ENDPOINT

Or configure programmatically:
    >>> import pyai
    >>> pyai.configure(api_key="sk-...", model="gpt-4o")
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class pyaiConfig:
    """Global configuration for pyai."""

    # LLM settings
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    provider: str = "openai"

    # Azure-specific
    azure_endpoint: Optional[str] = None
    azure_api_version: str = "2024-02-01"

    # Behavior settings
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60

    # Caching
    enable_cache: bool = True
    cache_ttl: int = 3600  # 1 hour

    # Logging
    verbose: bool = False

    # Extra settings
    extra: Dict[str, Any] = field(default_factory=dict)

    def get_api_key(self) -> str:
        """Get API key from config or environment."""
        if self.api_key:
            return self.api_key

        # Try environment variables based on provider
        env_keys = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "azure": "AZURE_OPENAI_API_KEY",
        }

        env_key = env_keys.get(self.provider, "OPENAI_API_KEY")
        api_key = os.environ.get(env_key)

        if not api_key:
            # Fallback to generic key
            api_key = os.environ.get("pyai_API_KEY")

        return api_key

    def get_azure_endpoint(self) -> Optional[str]:
        """Get Azure endpoint from config or environment."""
        return self.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")


# Global configuration instance
_config = pyaiConfig()


def configure(
    api_key: str = None,
    model: str = None,
    provider: str = None,
    azure_endpoint: str = None,
    temperature: float = None,
    max_tokens: int = None,
    timeout: int = None,
    verbose: bool = None,
    **kwargs,
) -> None:
    """
    Configure pyai globally. All parameters are optional.

    Args:
        api_key: Your API key
        model: Default model (e.g., "gpt-4o", "claude-3-opus")
        provider: LLM provider ("openai", "anthropic", "azure")
        azure_endpoint: Azure OpenAI endpoint URL
        temperature: Default temperature (0.0 - 2.0)
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds
        verbose: Enable verbose logging
        **kwargs: Additional settings stored in config.extra

    Example:
        >>> import pyai
        >>> pyai.configure(
        ...     api_key="sk-...",
        ...     model="gpt-4o",
        ...     temperature=0.3
        ... )
    """
    global _config

    if api_key is not None:
        _config.api_key = api_key
    if model is not None:
        _config.model = model
    if provider is not None:
        _config.provider = provider
    if azure_endpoint is not None:
        _config.azure_endpoint = azure_endpoint
    if temperature is not None:
        _config.temperature = temperature
    if max_tokens is not None:
        _config.max_tokens = max_tokens
    if timeout is not None:
        _config.timeout = timeout
    if verbose is not None:
        _config.verbose = verbose

    _config.extra.update(kwargs)


def set_config(**kwargs) -> None:
    """Alias for configure()."""
    configure(**kwargs)


def get_config() -> pyaiConfig:
    """Get the current global configuration."""
    return _config


def reset_config() -> None:
    """Reset configuration to defaults."""
    global _config
    _config = pyaiConfig()


# =============================================================================
# Convenience Methods for Common Operations
# =============================================================================


def set_model(model: str) -> None:
    """Set the default model."""
    configure(model=model)


def set_api_key(api_key: str) -> None:
    """Set the API key."""
    configure(api_key=api_key)


def use_azure(
    endpoint: str, deployment: str = None, api_version: str = "2024-02-15-preview"
) -> None:
    """
    Configure to use Azure OpenAI.

    Args:
        endpoint: Azure OpenAI endpoint URL
        deployment: Deployment name (also used as model)
        api_version: API version to use
    """
    configure(provider="azure", azure_endpoint=endpoint, model=deployment)
    _config.extra["azure_api_version"] = api_version


def enable_mock(enabled: bool = True) -> None:
    """
    Enable or disable mock mode for testing.

    When mock mode is enabled, LLM calls return predefined responses
    instead of making actual API calls.
    """
    _config.extra["mock_mode"] = enabled


def is_mock_enabled() -> bool:
    """Check if mock mode is enabled."""
    return _config.extra.get("mock_mode", False)


# =============================================================================
# Config Object for Direct Access
# =============================================================================


class ConfigAccessor:
    """
    Config object that provides both attribute access and methods.

    Example:
        >>> from pyai.easy.config import config
        >>> config.set_model("gpt-4o")
        >>> config.use_azure("https://...")
        >>> config.enable_mock(True)
    """

    @property
    def api_key(self):
        return _config.api_key

    @property
    def model(self):
        return _config.model

    @property
    def provider(self):
        return _config.provider

    @property
    def azure_endpoint(self):
        return _config.azure_endpoint

    @property
    def temperature(self):
        return _config.temperature

    # Methods
    set_model = staticmethod(set_model)
    set_api_key = staticmethod(set_api_key)
    use_azure = staticmethod(use_azure)
    enable_mock = staticmethod(enable_mock)
    is_mock_enabled = staticmethod(is_mock_enabled)

    def configure(self, **kwargs):
        """Configure pyai settings."""
        configure(**kwargs)

    def reset(self):
        """Reset to defaults."""
        reset_config()


# Global config accessor instance
config = ConfigAccessor()


def with_config(**kwargs):
    """
    Context manager for temporary configuration changes.

    Example:
        >>> with pyai.with_config(temperature=0.0):
        ...     result = ask("What is 2+2?")  # Uses temperature=0.0
        >>> # Back to original config
    """
    from contextlib import contextmanager

    @contextmanager
    def _context():
        global _config
        old_config = pyaiConfig(
            api_key=_config.api_key,
            model=_config.model,
            provider=_config.provider,
            azure_endpoint=_config.azure_endpoint,
            temperature=_config.temperature,
            max_tokens=_config.max_tokens,
            timeout=_config.timeout,
            verbose=_config.verbose,
            extra=_config.extra.copy(),
        )
        try:
            configure(**kwargs)
            yield _config
        finally:
            _config = old_config

    return _context()
