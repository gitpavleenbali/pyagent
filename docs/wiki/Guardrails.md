# Guardrails

The guardrails module provides input/output validation and safety rails for AI applications.

## Overview

Guardrails protect your AI applications with:
- **Input Validation**: Block harmful or inappropriate inputs
- **Output Filtering**: Redact or sanitize model outputs
- **PII Detection**: Automatic detection of personal information
- **Content Safety**: Block harmful content patterns

## Quick Start

```python
from pyai import guardrails, ask

# Simple content filter
safe_ask = guardrails.wrap(ask, block_pii=True)
safe_ask("What is the capital of France?")  # Works
safe_ask("My SSN is 123-45-6789")  # Blocked
```

## Built-in Validators

### PII Detection

```python
from pyai import guardrails

# Check for PII
result = guardrails.check_pii("My email is test@example.com")
print(result.passed)  # False
print(result.message)  # "Detected potential email"

# Detects:
# - Social Security Numbers (SSN)
# - Credit card numbers
# - Email addresses
# - Phone numbers
# - IP addresses
```

### Harmful Content

```python
# Check for harmful patterns
result = guardrails.check_harmful("How to hack a system")
print(result.passed)  # False

# Detects:
# - Hacking/exploit mentions
# - Password stealing attempts
# - SQL injection patterns
# - Weapon/bomb instructions
```

### SQL Injection

```python
# Block SQL injection attempts
@guardrails.input_validator
def no_sql(text: str) -> bool:
    dangerous = ["SELECT", "DROP", "DELETE", "INSERT", "UPDATE"]
    return not any(kw in text.upper() for kw in dangerous)
```

## Custom Validators

### Input Validators

```python
from pyai import guardrails

@guardrails.input_validator
def no_profanity(text: str) -> bool:
    """Block profane content."""
    bad_words = ["badword1", "badword2"]  # Your list
    return not any(word in text.lower() for word in bad_words)

@guardrails.input_validator
def max_length(text: str) -> bool:
    """Enforce maximum length."""
    return len(text) <= 1000
```

### Output Filters

```python
import re
from pyai import guardrails

@guardrails.output_filter
def redact_emails(text: str) -> str:
    """Redact email addresses."""
    return re.sub(r'\S+@\S+', '[EMAIL REDACTED]', text)

@guardrails.output_filter
def redact_phone(text: str) -> str:
    """Redact phone numbers."""
    return re.sub(r'\d{3}[-.]?\d{3}[-.]?\d{4}', '[PHONE REDACTED]', text)
```

## Protection Wrapper

### Wrap Functions

```python
from pyai import guardrails, ask

# Wrap with validators
safe_ask = guardrails.protect(
    ask,
    validators=[
        guardrails.no_pii,
        guardrails.no_harmful,
        no_profanity
    ],
    filters=[
        redact_emails,
        redact_phone
    ]
)

# Use protected function
result = safe_ask("What is AI?")  # Works
```

### Configuration Options

```python
safe_ask = guardrails.protect(
    ask,
    block_pii=True,           # Block inputs with PII
    block_harmful=True,       # Block harmful content
    max_input_length=5000,    # Maximum input length
    max_output_length=10000,  # Maximum output length
    validators=[],            # Custom input validators
    filters=[],               # Custom output filters
    on_violation="raise"      # "raise", "return_none", "return_error"
)
```

## GuardrailResult

```python
from pyai.easy.guardrails import GuardrailResult

result = GuardrailResult(
    passed=False,
    message="Detected potential SSN",
    blocked_content="123-45-6789",
    rule_name="no_pii"
)

if not result:
    print(f"Blocked: {result.message}")
```

## GuardrailViolation

```python
from pyai.easy.guardrails import GuardrailViolation

try:
    safe_ask("My SSN is 123-45-6789")
except GuardrailViolation as e:
    print(f"Rule: {e.rule}")      # "no_pii"
    print(f"Message: {e.message}")  # Details
```

## Input Guardrails

### Wrap with Error Handling

```python
from pyai import guardrails, ask

# Returns error message instead of raising
safe_ask = guardrails.wrap(
    ask,
    block_pii=True,
    on_violation="return_error"
)

result = safe_ask("My SSN is 123-45-6789")
print(result)  # "Error: Input blocked - Detected potential ssn"
```

### Silent Blocking

```python
# Returns None when blocked
safe_ask = guardrails.wrap(
    ask,
    block_pii=True,
    on_violation="return_none"
)

result = safe_ask("My SSN is 123-45-6789")
print(result)  # None
```

## Output Guardrails

### Automatic Redaction

```python
from pyai import guardrails, ask

# Redact PII in outputs
sanitized_ask = guardrails.sanitize_output(
    ask,
    redact_pii=True
)

result = sanitized_ask("Generate a sample user profile")
# Output: "Name: John, Email: [EMAIL REDACTED], Phone: [PHONE REDACTED]"
```

### Custom Sanitizers

```python
@guardrails.output_filter
def remove_code(text: str) -> str:
    """Remove code blocks from output."""
    import re
    return re.sub(r'```[\s\S]*?```', '[CODE REMOVED]', text)
```

## Combining Guardrails

```python
from pyai import guardrails, ask, research

# Create reusable guardrail config
security_profile = guardrails.Profile(
    validators=[
        guardrails.no_pii,
        guardrails.no_harmful,
        guardrails.no_injection
    ],
    filters=[
        guardrails.redact_pii,
        guardrails.truncate(max_length=5000)
    ],
    on_violation="raise"
)

# Apply to multiple functions
safe_ask = security_profile.protect(ask)
safe_research = security_profile.protect(research)
```

## Integration with Agents

```python
from pyai import Agent, guardrails

agent = Agent(
    name="safe-agent",
    instructions="You are a helpful assistant.",
    guardrails=guardrails.Profile(
        block_pii=True,
        block_harmful=True
    )
)

# All agent interactions are protected
result = agent.run("What is the weather?")
```

## See Also

- [Azure-AD-Auth](Azure-AD-Auth) - Authentication
- [Tracing](Tracing) - Observability
- [Agent](Agent) - Agent class
