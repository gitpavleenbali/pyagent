"""
guardrails - Input/output validation and safety rails

Protect your AI applications with input validation, output filtering,
and content safety checks. Inspired by OpenAI Agents' guardrails but simpler.

Examples:
    >>> from pyai import guardrails, ask

    # Simple content filter
    >>> safe_ask = guardrails.wrap(ask, block_pii=True)
    >>> safe_ask("What is the capital of France?")  # Works
    >>> safe_ask("My SSN is 123-45-6789")  # Blocked

    # Custom validators
    >>> @guardrails.input_validator
    ... def no_sql(text: str) -> bool:
    ...     return "SELECT" not in text.upper()

    # Output filters
    >>> @guardrails.output_filter
    ... def redact_emails(text: str) -> str:
    ...     return re.sub(r'\\S+@\\S+', '[REDACTED]', text)

    # Combine guardrails
    >>> protected = guardrails.protect(
    ...     ask,
    ...     validators=[no_sql, guardrails.no_pii],
    ...     filters=[redact_emails]
    ... )
"""

import re
from dataclasses import dataclass
from functools import wraps
from typing import Callable, List


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    passed: bool
    message: str = ""
    blocked_content: str = None
    rule_name: str = None

    def __bool__(self) -> bool:
        return self.passed


@dataclass
class GuardrailViolation(Exception):
    """Raised when a guardrail is violated."""

    rule: str
    message: str
    content: str = None

    def __str__(self) -> str:
        return f"Guardrail '{self.rule}' violated: {self.message}"


# =============================================================================
# Built-in Validators
# =============================================================================


def _check_pii(text: str) -> GuardrailResult:
    """Check for personally identifiable information."""
    patterns = {
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    }

    for pii_type, pattern in patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return GuardrailResult(
                passed=False,
                message=f"Detected potential {pii_type.replace('_', ' ')}",
                blocked_content=text,
                rule_name="no_pii",
            )

    return GuardrailResult(passed=True, rule_name="no_pii")


def _check_harmful(text: str) -> GuardrailResult:
    """Check for potentially harmful content."""
    harmful_patterns = [
        r"\b(hack|exploit|crack|malware|virus)\b",
        r"\b(password|credential)s?\s+(steal|dump|crack)",
        r"(sql|xss|injection)\s*attack",
        r"\b(weapon|bomb|explosive)\b.*\b(make|build|create)\b",
    ]

    text_lower = text.lower()
    for pattern in harmful_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return GuardrailResult(
                passed=False,
                message="Detected potentially harmful content",
                blocked_content=text,
                rule_name="no_harmful",
            )

    return GuardrailResult(passed=True, rule_name="no_harmful")


def _check_prompt_injection(text: str) -> GuardrailResult:
    """Check for common prompt injection attempts."""
    injection_patterns = [
        # Instruction override attempts
        r"ignore\s+.*\s*instructions",
        r"ignore\s+(previous|all|above|prior|earlier)",
        r"disregard\s+(your|the|all|previous)",
        r"forget\s+(your|the|all|previous)\s+(rules|instructions|guidelines)",
        # Role play / identity manipulation
        r"you\s+are\s+now\s+(a|an|the)",
        r"pretend\s+(to\s+be|you\s+are|you're)",
        r"act\s+as\s+(if|a|an)",
        r"roleplay\s+as",
        # Known jailbreak techniques
        r"jailbreak",
        r"DAN\s*mode",
        r"do\s+anything\s+now",
        # System prompt manipulation
        r"\[SYSTEM\]|\[INST\]|\[/INST\]",
        r"<\|system\|>|<\|user\|>|<\|assistant\|>",
        r"system\s*:\s*you\s+are",
        # Bypass attempts
        r"bypass\s+(the|your|safety|content)",
        r"override\s+(the|your|safety|restrictions)",
        r"unlock\s+(your|full|hidden)",
    ]

    for pattern in injection_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return GuardrailResult(
                passed=False,
                message="Detected potential prompt injection",
                blocked_content=text,
                rule_name="no_injection",
            )

    return GuardrailResult(passed=True, rule_name="no_injection")


def _check_length(max_length: int) -> Callable:
    """Create a length validator."""

    def validator(text: str) -> GuardrailResult:
        if len(text) > max_length:
            return GuardrailResult(
                passed=False,
                message=f"Input exceeds maximum length of {max_length}",
                rule_name="max_length",
            )
        return GuardrailResult(passed=True, rule_name="max_length")

    return validator


# =============================================================================
# Built-in Output Filters
# =============================================================================


def _redact_pii(text: str) -> str:
    """Redact PII from output."""
    replacements = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]"),
        (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[CARD REDACTED]"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
        (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]"),
    ]

    result = text
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result


def _limit_length(max_length: int) -> Callable:
    """Create an output length limiter."""

    def filter_fn(text: str) -> str:
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text

    return filter_fn


# =============================================================================
# Guardrail Decorators
# =============================================================================


def input_validator(func: Callable[[str], bool]) -> Callable[[str], GuardrailResult]:
    """
    Decorator to create an input validator from a simple function.

    Args:
        func: Function that returns True if input is valid

    Examples:
        >>> @guardrails.input_validator
        ... def no_sql(text: str) -> bool:
        ...     return "SELECT" not in text.upper()
    """

    @wraps(func)
    def validator(text: str) -> GuardrailResult:
        try:
            passed = func(text)
            return GuardrailResult(
                passed=passed,
                message="" if passed else f"Validation failed: {func.__name__}",
                rule_name=func.__name__,
            )
        except Exception as e:
            return GuardrailResult(
                passed=False, message=f"Validation error: {e}", rule_name=func.__name__
            )

    return validator


def output_filter(func: Callable[[str], str]) -> Callable[[str], str]:
    """
    Decorator to create an output filter.

    Args:
        func: Function that transforms output text

    Examples:
        >>> @guardrails.output_filter
        ... def uppercase(text: str) -> str:
        ...     return text.upper()
    """

    @wraps(func)
    def filter_fn(text: str) -> str:
        return func(text)

    filter_fn._is_output_filter = True
    return filter_fn


# =============================================================================
# Main Guardrail Functions
# =============================================================================


def validate(
    text: str,
    validators: List[Callable] = None,
    *,
    block_pii: bool = False,
    block_harmful: bool = False,
    block_injection: bool = False,
    max_length: int = None,
    raise_on_fail: bool = True,
) -> GuardrailResult:
    """
    Validate input text against guardrails.

    Args:
        text: Text to validate
        validators: Custom validator functions
        block_pii: Block personally identifiable information
        block_harmful: Block potentially harmful content
        block_injection: Block prompt injection attempts
        max_length: Maximum allowed length
        raise_on_fail: Raise exception on validation failure

    Returns:
        GuardrailResult with validation status

    Examples:
        >>> result = guardrails.validate("Hello world", block_pii=True)
        >>> if result.passed:
        ...     print("Input is safe")
    """
    all_validators = list(validators or [])

    if block_pii:
        all_validators.append(_check_pii)
    if block_harmful:
        all_validators.append(_check_harmful)
    if block_injection:
        all_validators.append(_check_prompt_injection)
    if max_length:
        all_validators.append(_check_length(max_length))

    for validator in all_validators:
        result = validator(text)
        if not result.passed:
            if raise_on_fail:
                raise GuardrailViolation(
                    rule=result.rule_name, message=result.message, content=result.blocked_content
                )
            return result

    return GuardrailResult(passed=True)


def filter_output(
    text: str, filters: List[Callable] = None, *, redact_pii: bool = False, max_length: int = None
) -> str:
    """
    Filter output text.

    Args:
        text: Text to filter
        filters: Custom filter functions
        redact_pii: Redact PII from output
        max_length: Maximum output length

    Returns:
        Filtered text

    Examples:
        >>> text = "Contact john@email.com for help"
        >>> clean = guardrails.filter_output(text, redact_pii=True)
        >>> print(clean)  # "Contact [EMAIL] for help"
    """
    all_filters = list(filters or [])

    if redact_pii:
        all_filters.append(_redact_pii)
    if max_length:
        all_filters.append(_limit_length(max_length))

    result = text
    for f in all_filters:
        result = f(result)

    return result


def wrap(
    func: Callable,
    *,
    validators: List[Callable] = None,
    filters: List[Callable] = None,
    block_pii: bool = False,
    block_harmful: bool = False,
    block_injection: bool = False,
    redact_pii: bool = False,
    max_input_length: int = None,
    max_output_length: int = None,
) -> Callable:
    """
    Wrap a function with guardrails.

    Args:
        func: Function to wrap (e.g., ask, agent)
        validators: Input validators
        filters: Output filters
        block_pii: Block PII in input
        block_harmful: Block harmful content in input
        block_injection: Block prompt injection in input
        redact_pii: Redact PII from output
        max_input_length: Maximum input length
        max_output_length: Maximum output length

    Returns:
        Wrapped function with guardrails

    Examples:
        >>> safe_ask = guardrails.wrap(ask, block_pii=True, redact_pii=True)
        >>> answer = safe_ask("What is Python?")  # Protected
    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        # Validate input
        if args:
            input_text = str(args[0])
            validate(
                input_text,
                validators=validators,
                block_pii=block_pii,
                block_harmful=block_harmful,
                block_injection=block_injection,
                max_length=max_input_length,
            )

        # Call function
        result = func(*args, **kwargs)

        # Filter output
        if isinstance(result, str):
            result = filter_output(
                result, filters=filters, redact_pii=redact_pii, max_length=max_output_length
            )

        return result

    return wrapped


def protect(
    func: Callable, *, validators: List[Callable] = None, filters: List[Callable] = None, **kwargs
) -> Callable:
    """
    Alias for wrap() with a more intuitive name.

    Examples:
        >>> protected_ask = guardrails.protect(ask, block_pii=True)
    """
    return wrap(func, validators=validators, filters=filters, **kwargs)


# =============================================================================
# Pre-built Guardrails
# =============================================================================

# Export validators as module-level functions
no_pii = _check_pii
no_harmful = _check_harmful
no_injection = _check_prompt_injection
max_length = _check_length

# Export filters
redact_pii = _redact_pii
limit_length = _limit_length


class GuardrailsModule:
    """Guardrails module with all functions attached."""

    # Functions
    validate = staticmethod(validate)
    filter_output = staticmethod(filter_output)
    wrap = staticmethod(wrap)
    protect = staticmethod(protect)

    # Decorators
    input_validator = staticmethod(input_validator)
    output_filter = staticmethod(output_filter)

    # Built-in validators
    no_pii = staticmethod(no_pii)
    no_harmful = staticmethod(no_harmful)
    no_injection = staticmethod(no_injection)
    max_length = staticmethod(max_length)

    # Built-in filters
    redact_pii = staticmethod(redact_pii)
    limit_length = staticmethod(limit_length)

    # Classes
    Result = GuardrailResult
    Violation = GuardrailViolation


# Module-level instance
guardrails = GuardrailsModule()
