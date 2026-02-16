# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
PyAgent Errors Module

Provides a structured error hierarchy for PyAgent.

Inspired by Google ADK's errors/ module.

Error Hierarchy:
    PyAgentError (base)
    ├── ConfigurationError
    ├── ModelError
    │   ├── ModelNotFoundError
    │   ├── APIError
    │   ├── RateLimitError
    │   ├── TokenLimitError
    │   └── AuthenticationError
    ├── SessionError
    │   ├── SessionNotFoundError
    │   └── SessionExpiredError
    ├── SkillError
    │   ├── SkillNotFoundError
    │   └── SkillExecutionError
    ├── ValidationError
    │   ├── InputValidationError
    │   └── OutputValidationError
    ├── ExecutionError
    │   ├── TimeoutError
    │   └── CodeExecutionError
    └── HandoffError

Example:
    from pyagent.errors import ModelError, RateLimitError
    
    try:
        agent.run("Hello")
    except RateLimitError as e:
        print(f"Rate limited, retry in {e.retry_after}s")
    except ModelError as e:
        print(f"Model error: {e}")
"""


class PyAgentError(Exception):
    """Base exception for all PyAgent errors.
    
    All PyAgent exceptions inherit from this class, making it easy
    to catch any PyAgent-related error.
    
    Example:
        try:
            agent.run("Hello")
        except PyAgentError as e:
            print(f"PyAgent error: {e}")
    """
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(PyAgentError):
    """Error in configuration or setup.
    
    Raised when configuration is invalid or missing required values.
    
    Example:
        raise ConfigurationError("API key not set", {"required": "OPENAI_API_KEY"})
    """
    pass


# =============================================================================
# Model Errors
# =============================================================================

class ModelError(PyAgentError):
    """Base class for model-related errors.
    
    Raised when there's an issue with the LLM model.
    """
    pass


class ModelNotFoundError(ModelError):
    """Model not found or not available.
    
    Example:
        raise ModelNotFoundError("gpt-5 not found", {"available": ["gpt-4", "gpt-3.5"]})
    """
    
    def __init__(self, model_name: str, available_models: list = None):
        super().__init__(
            f"Model not found: {model_name}",
            {"model": model_name, "available": available_models or []}
        )
        self.model_name = model_name
        self.available_models = available_models or []


class APIError(ModelError):
    """Error from the model API.
    
    Wraps API errors from OpenAI, Azure, Anthropic, etc.
    
    Example:
        raise APIError("API request failed", status_code=500, provider="openai")
    """
    
    def __init__(self, message: str, status_code: int = None, provider: str = None, response: dict = None):
        super().__init__(
            message,
            {"status_code": status_code, "provider": provider, "response": response}
        )
        self.status_code = status_code
        self.provider = provider
        self.response = response


class RateLimitError(APIError):
    """API rate limit exceeded.
    
    Includes retry information when available.
    
    Example:
        raise RateLimitError("Rate limit exceeded", retry_after=30)
    """
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: float = None, **kwargs):
        super().__init__(message, status_code=429, **kwargs)
        self.retry_after = retry_after
        self.details["retry_after"] = retry_after


class TokenLimitError(ModelError):
    """Token limit exceeded.
    
    Raised when input or output exceeds model's token limit.
    
    Example:
        raise TokenLimitError("Input too long", tokens_used=50000, max_tokens=32000)
    """
    
    def __init__(self, message: str, tokens_used: int = None, max_tokens: int = None):
        super().__init__(
            message,
            {"tokens_used": tokens_used, "max_tokens": max_tokens}
        )
        self.tokens_used = tokens_used
        self.max_tokens = max_tokens


class AuthenticationError(ModelError):
    """Authentication failed.
    
    Raised when API key is invalid or expired.
    
    Example:
        raise AuthenticationError("Invalid API key", provider="azure")
    """
    
    def __init__(self, message: str = "Authentication failed", provider: str = None):
        super().__init__(message, {"provider": provider})
        self.provider = provider


# =============================================================================
# Session Errors
# =============================================================================

class SessionError(PyAgentError):
    """Base class for session-related errors."""
    pass


class SessionNotFoundError(SessionError):
    """Session not found.
    
    Example:
        raise SessionNotFoundError("abc123")
    """
    
    def __init__(self, session_id: str):
        super().__init__(
            f"Session not found: {session_id}",
            {"session_id": session_id}
        )
        self.session_id = session_id


class SessionExpiredError(SessionError):
    """Session has expired.
    
    Example:
        raise SessionExpiredError("Session expired after 30 days", session_id="abc123")
    """
    
    def __init__(self, message: str = "Session expired", session_id: str = None):
        super().__init__(message, {"session_id": session_id})
        self.session_id = session_id


# =============================================================================
# Skill Errors
# =============================================================================

class SkillError(PyAgentError):
    """Base class for skill/tool-related errors."""
    pass


class SkillNotFoundError(SkillError):
    """Skill not found in registry.
    
    Example:
        raise SkillNotFoundError("calculate", available=["add", "subtract"])
    """
    
    def __init__(self, skill_name: str, available_skills: list = None):
        super().__init__(
            f"Skill not found: {skill_name}",
            {"skill": skill_name, "available": available_skills or []}
        )
        self.skill_name = skill_name
        self.available_skills = available_skills or []


class SkillExecutionError(SkillError):
    """Error during skill execution.
    
    Example:
        raise SkillExecutionError("Division by zero", skill_name="divide", args={"a": 1, "b": 0})
    """
    
    def __init__(self, message: str, skill_name: str = None, args: dict = None, original_error: Exception = None):
        super().__init__(
            message,
            {"skill": skill_name, "args": args, "original_error": str(original_error) if original_error else None}
        )
        self.skill_name = skill_name
        self.args = args
        self.original_error = original_error


# =============================================================================
# Validation Errors
# =============================================================================

class ValidationError(PyAgentError):
    """Base class for validation errors."""
    pass


class InputValidationError(ValidationError):
    """Input validation failed.
    
    Raised when user input violates guardrails or validation rules.
    
    Example:
        raise InputValidationError("Input contains PII", field="email", value="***")
    """
    
    def __init__(self, message: str, field: str = None, value: str = None, rule: str = None):
        super().__init__(
            message,
            {"field": field, "value": value, "rule": rule}
        )
        self.field = field
        self.value = value
        self.rule = rule


class OutputValidationError(ValidationError):
    """Output validation failed.
    
    Raised when model output violates constraints.
    
    Example:
        raise OutputValidationError("Output contains blocked content", content_type="violence")
    """
    
    def __init__(self, message: str, content_type: str = None, output_preview: str = None):
        super().__init__(
            message,
            {"content_type": content_type, "output_preview": output_preview}
        )
        self.content_type = content_type
        self.output_preview = output_preview


# =============================================================================
# Execution Errors
# =============================================================================

class ExecutionError(PyAgentError):
    """Base class for execution errors."""
    pass


class TimeoutError(ExecutionError):
    """Operation timed out.
    
    Example:
        raise TimeoutError("Agent execution timed out", timeout_seconds=30)
    """
    
    def __init__(self, message: str = "Operation timed out", timeout_seconds: float = None):
        super().__init__(message, {"timeout_seconds": timeout_seconds})
        self.timeout_seconds = timeout_seconds


class CodeExecutionError(ExecutionError):
    """Error during code execution.
    
    Example:
        raise CodeExecutionError("NameError: 'x' is not defined", code="print(x)")
    """
    
    def __init__(self, message: str, code: str = None, line_number: int = None):
        super().__init__(
            message,
            {"code_preview": code[:100] if code else None, "line_number": line_number}
        )
        self.code = code
        self.line_number = line_number


# =============================================================================
# Handoff Errors
# =============================================================================

class HandoffError(PyAgentError):
    """Error during agent handoff.
    
    Example:
        raise HandoffError("Target agent not found", source="router", target="specialist")
    """
    
    def __init__(self, message: str, source_agent: str = None, target_agent: str = None):
        super().__init__(
            message,
            {"source": source_agent, "target": target_agent}
        )
        self.source_agent = source_agent
        self.target_agent = target_agent


# =============================================================================
# Utility Functions
# =============================================================================

def wrap_error(original: Exception, context: str = None) -> PyAgentError:
    """Wrap an external exception in a PyAgentError.
    
    Useful for wrapping third-party library errors.
    
    Example:
        try:
            openai.chat(...)
        except openai.RateLimitError as e:
            raise wrap_error(e, "OpenAI API call failed")
    
    Args:
        original: The original exception
        context: Additional context message
        
    Returns:
        Appropriate PyAgentError subclass
    """
    error_name = type(original).__name__.lower()
    message = f"{context}: {original}" if context else str(original)
    
    # Map common errors
    if "ratelimit" in error_name:
        retry = getattr(original, "retry_after", None)
        return RateLimitError(message, retry_after=retry)
    
    if "authentication" in error_name or "auth" in error_name:
        return AuthenticationError(message)
    
    if "timeout" in error_name:
        return TimeoutError(message)
    
    if "notfound" in error_name or "404" in str(original):
        return ModelNotFoundError(message)
    
    if "validation" in error_name:
        return ValidationError(message)
    
    # Default to base error
    return PyAgentError(message, {"original_type": type(original).__name__})


def is_retriable(error: Exception) -> bool:
    """Check if an error is retriable.
    
    Useful for implementing retry logic.
    
    Example:
        from pyagent.errors import is_retriable
        
        for attempt in range(3):
            try:
                result = agent.run(input)
                break
            except PyAgentError as e:
                if is_retriable(e) and attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise
    
    Args:
        error: The exception to check
        
    Returns:
        True if the error might succeed on retry
    """
    retriable_types = (
        RateLimitError,
        TimeoutError,
        APIError,  # Most API errors are transient
    )
    
    if isinstance(error, retriable_types):
        # Don't retry auth errors
        if isinstance(error, AuthenticationError):
            return False
        return True
    
    # Check for common transient errors
    error_str = str(error).lower()
    transient_keywords = ["timeout", "connection", "temporary", "unavailable", "retry", "overloaded"]
    
    return any(keyword in error_str for keyword in transient_keywords)


__all__ = [
    # Base
    "PyAgentError",
    # Configuration
    "ConfigurationError",
    # Model errors
    "ModelError",
    "ModelNotFoundError",
    "APIError",
    "RateLimitError",
    "TokenLimitError",
    "AuthenticationError",
    # Session errors
    "SessionError",
    "SessionNotFoundError",
    "SessionExpiredError",
    # Skill errors
    "SkillError",
    "SkillNotFoundError",
    "SkillExecutionError",
    # Validation errors
    "ValidationError",
    "InputValidationError",
    "OutputValidationError",
    # Execution errors
    "ExecutionError",
    "TimeoutError",
    "CodeExecutionError",
    # Handoff errors
    "HandoffError",
    # Utilities
    "wrap_error",
    "is_retriable",
]
