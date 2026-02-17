# Evaluation

The Evaluation module provides testing and benchmarking for AI agents.

## Overview

PYAI's evaluation framework enables:
- Automated agent testing
- Quality metrics
- Regression detection
- Benchmark comparisons

## Quick Start

```python
from pyai.evaluation import Evaluator, TestCase

# Create test cases
tests = [
    TestCase(
        input="What is 2 + 2?",
        expected="4"
    ),
    TestCase(
        input="Capital of France?",
        expected="Paris"
    )
]

# Create evaluator
evaluator = Evaluator(agent)

# Run evaluation
results = evaluator.run(tests)
print(f"Score: {results.score}%")
```

## Test Cases

### Basic Test Case

```python
from pyai.evaluation import TestCase

test = TestCase(
    input="What is Python?",
    expected="programming language",
    match_type="contains"  # Contains expected text
)
```

### Match Types

| Type | Description |
|------|-------------|
| `exact` | Exact string match |
| `contains` | Contains substring |
| `regex` | Regex pattern match |
| `semantic` | Semantic similarity |
| `function` | Custom function |

### Custom Matcher

```python
def math_matcher(output: str, expected: str) -> bool:
    """Check if math result is correct"""
    try:
        return abs(float(output) - float(expected)) < 0.01
    except:
        return False

test = TestCase(
    input="Calculate 25 * 4",
    expected="100",
    match_type="function",
    matcher=math_matcher
)
```

## Evaluator

```python
from pyai.evaluation import Evaluator

evaluator = Evaluator(
    agent,
    metrics=["accuracy", "latency", "tokens"]
)

results = evaluator.run(test_cases)
```

### Metrics

| Metric | Description |
|--------|-------------|
| `accuracy` | Correct responses % |
| `latency` | Response time |
| `tokens` | Token usage |
| `cost` | Estimated cost |
| `semantic_score` | Embedding similarity |

## EvalSet

```python
from pyai.evaluation import EvalSet

# Create evaluation set
eval_set = EvalSet(
    name="Math Tests",
    version="1.0"
)

eval_set.add(TestCase(input="2+2", expected="4"))
eval_set.add(TestCase(input="10/2", expected="5"))

# Save for reuse
eval_set.save("math_tests.json")

# Load
eval_set = EvalSet.load("math_tests.json")
```

## Full Example

```python
from pyai import Agent
from pyai.evaluation import Evaluator, EvalSet, TestCase

# Create agent
agent = Agent(
    name="Math Tutor",
    instructions="You are a math tutor. Answer math questions."
)

# Create test suite
math_tests = EvalSet(name="Math Evaluation")
math_tests.add(TestCase(input="What is 15 * 3?", expected="45"))
math_tests.add(TestCase(input="Square root of 144?", expected="12"))
math_tests.add(TestCase(input="What is 100 / 4?", expected="25"))

# Run evaluation
evaluator = Evaluator(agent)
results = evaluator.run(math_tests)

# Print results
print(f"Overall Score: {results.score}%")
print(f"Passed: {results.passed}/{results.total}")
print(f"Average Latency: {results.avg_latency}ms")

# Detailed results
for result in results.details:
    status = "✓" if result.passed else "✗"
    print(f"{status} {result.input} -> {result.output}")
```

## Comparison

```python
# Compare agents
evaluator = Evaluator()

results_v1 = evaluator.run(agent_v1, tests)
results_v2 = evaluator.run(agent_v2, tests)

comparison = evaluator.compare(results_v1, results_v2)
print(f"V2 is {comparison.improvement}% better")
```

## CI/CD Integration

```python
# In pytest
import pytest
from pyai.evaluation import Evaluator, EvalSet

@pytest.fixture
def agent():
    return Agent(name="Test Agent", instructions="...")

@pytest.fixture
def eval_set():
    return EvalSet.load("tests/eval_cases.json")

def test_agent_quality(agent, eval_set):
    evaluator = Evaluator(agent)
    results = evaluator.run(eval_set)
    
    assert results.score >= 90, f"Score {results.score}% below threshold"
    assert results.avg_latency < 5000, "Too slow"
```

## See Also

- [[Evaluator]] - Evaluator class
- [[TestCase]] - Test case class
- [[EvalSet]] - Evaluation set
- [[Agent]] - Agent class
