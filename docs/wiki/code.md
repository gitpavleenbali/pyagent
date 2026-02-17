# code

The `code` module provides AI-powered code operations.

## Import

```python
from pyai.easy import code
```

## Quick Start

```python
from pyai.easy import code

# Write code
result = code.write("Python function to calculate factorial")

# Review code
review = code.review(my_code)

# Debug code
fix = code.debug(broken_code, error_message)

# Explain code
explanation = code.explain(complex_code)
```

## Functions

### Write Code

```python
from pyai.easy import code

# Generate code from description
result = code.write("function to validate email addresses")
print(result)

# Specify language
result = code.write(
    "REST API endpoint for user authentication",
    language="python",
    framework="fastapi"
)
```

### Review Code

```python
# Code review
review = code.review("""
def calc(x):
    return x*x
""")

print(review)
# {
#   "issues": ["Function name not descriptive"],
#   "suggestions": ["Add type hints", "Add docstring"],
#   "security": [],
#   "overall": "Good logic, needs documentation"
# }
```

### Debug Code

```python
broken = """
def divide(a, b):
    return a / b
"""
error = "ZeroDivisionError: division by zero"

fix = code.debug(broken, error)
print(fix)
# Returns fixed code with explanation
```

### Explain Code

```python
complex_code = """
def fib(n):
    return n if n < 2 else fib(n-1) + fib(n-2)
"""

explanation = code.explain(complex_code)
print(explanation)
# Clear explanation of what the code does
```

### Refactor Code

```python
messy_code = """
def f(x):
    if x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        return f(x-1) + f(x-2)
"""

refactored = code.refactor(messy_code)
# Returns cleaner, optimized version
```

### Convert Code

```python
# Convert between languages
python_code = """
def greet(name):
    print(f"Hello, {name}!")
"""

javascript = code.convert(python_code, from_lang="python", to_lang="javascript")
print(javascript)
# function greet(name) {
#     console.log(`Hello, ${name}!`);
# }
```

### Add Tests

```python
# Generate tests for code
function_code = """
def add(a, b):
    return a + b
"""

tests = code.test(function_code)
print(tests)
# def test_add():
#     assert add(2, 3) == 5
#     assert add(-1, 1) == 0
#     assert add(0, 0) == 0
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `code` | str | required | Code to analyze |
| `language` | str | "python" | Programming language |
| `framework` | str | None | Framework (fastapi, django, etc.) |
| `style` | str | None | Code style preference |

## Examples

### Full Workflow

```python
from pyai.easy import code

# 1. Write code
func = code.write("function to parse CSV file")

# 2. Review it
review = code.review(func)
print("Issues:", review['issues'])

# 3. Add tests
tests = code.test(func)

# 4. Add documentation
documented = code.document(func)
```

### Async Usage

```python
import asyncio
from pyai.easy import code

async def main():
    result = await code.write_async("async HTTP client")
    print(result)

asyncio.run(main())
```

## See Also

- [[ask]] - Question answering
- [[generate]] - Content generation
- [[analyze]] - Data analysis
