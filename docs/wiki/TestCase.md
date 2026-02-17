# TestCase

A `TestCase` represents a single evaluation scenario for testing agent behavior.

## Import

```python
from pyagent.evaluation import TestCase
```

## Constructor

```python
TestCase(
    input: str,                      # Input prompt/question
    expected_output: str = None,     # Expected response (optional)
    criteria: list[str] = None,      # Criteria to evaluate
    metadata: dict = None,           # Additional metadata
    name: str = None,                # Test case name
    tags: list[str] = None,          # Tags for filtering
    timeout: float = 30.0            # Timeout in seconds
)
```

## Creating Test Cases

### Basic Test Case

```python
test = TestCase(
    input="What is the capital of France?",
    expected_output="Paris",
    criteria=["accuracy"]
)
```

### Open-Ended Test Case

```python
test = TestCase(
    input="Write a haiku about coding",
    expected_output=None,  # No exact expected output
    criteria=["creativity", "format"]
)
```

### With Metadata

```python
test = TestCase(
    input="Solve: 15 * 7 + 3",
    expected_output="108",
    name="math_test_001",
    tags=["math", "arithmetic"],
    metadata={
        "difficulty": "easy",
        "category": "multiplication"
    }
)
```

## Class Methods

### from_dict()

Create from dictionary:

```python
data = {
    "input": "What is 2+2?",
    "expected_output": "4",
    "criteria": ["accuracy"]
}
test = TestCase.from_dict(data)
```

### from_yaml()

Load from YAML file:

```yaml
# test_case.yaml
input: "Summarize this article"
expected_output: null
criteria:
  - relevance
  - conciseness
tags:
  - summarization
```

```python
test = TestCase.from_yaml("test_case.yaml")
```

## Batch Creation

```python
# Create multiple test cases
test_cases = [
    TestCase(input="2+2", expected_output="4"),
    TestCase(input="3*4", expected_output="12"),
    TestCase(input="10/2", expected_output="5"),
]

# Or from a list of dicts
test_cases = TestCase.batch_create([
    {"input": "Hello", "expected_output": "Hi"},
    {"input": "Goodbye", "expected_output": "Bye"},
])
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `input` | str | The input prompt |
| `expected_output` | str | Expected response |
| `criteria` | list | Evaluation criteria |
| `name` | str | Test case identifier |
| `tags` | list | Categorization tags |
| `metadata` | dict | Additional data |

## See Also

- [Evaluation-Module](Evaluation-Module) - Module overview
- [Evaluator](Evaluator) - Running evaluations
- [EvalSet](EvalSet) - Grouping test cases
