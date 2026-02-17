# EvalSet

An `EvalSet` is a collection of test cases grouped together for organized evaluation.

## Import

```python
from pyai.evaluation import EvalSet
```

## Constructor

```python
EvalSet(
    name: str,                       # Evaluation set name
    test_cases: list[TestCase],      # List of test cases
    description: str = None,         # Description
    version: str = "1.0",            # Version identifier
    metadata: dict = None            # Additional metadata
)
```

## Creating EvalSets

### Basic Creation

```python
from pyai.evaluation import EvalSet, TestCase

eval_set = EvalSet(
    name="Math Evaluation",
    test_cases=[
        TestCase(input="2+2", expected_output="4"),
        TestCase(input="5*5", expected_output="25"),
        TestCase(input="10/2", expected_output="5"),
    ],
    description="Basic arithmetic tests"
)
```

### From YAML File

```yaml
# eval_set.yaml
name: Customer Support Evaluation
description: Tests for customer support agent
version: "2.0"

test_cases:
  - input: "I need help with my order"
    criteria: [helpfulness, tone]
    tags: [orders]
    
  - input: "How do I return an item?"
    expected_output_contains: "return policy"
    criteria: [accuracy, clarity]
    tags: [returns]
    
  - input: "I'm very angry about this!"
    criteria: [empathy, de-escalation]
    tags: [complaints]
```

```python
eval_set = EvalSet.from_yaml("eval_set.yaml")
```

### From JSON

```python
eval_set = EvalSet.from_json("eval_set.json")
```

## Methods

### add_test_case()

Add a single test case:

```python
eval_set.add_test_case(
    TestCase(input="New test", expected_output="Expected")
)
```

### filter_by_tags()

Filter test cases by tags:

```python
# Get only math-related tests
math_tests = eval_set.filter_by_tags(["math"])

# Exclude certain tags
no_advanced = eval_set.filter_by_tags(exclude=["advanced"])
```

### split()

Split into training/validation sets:

```python
train_set, val_set = eval_set.split(ratio=0.8)
```

### sample()

Random sampling:

```python
# Get 10 random test cases
sample = eval_set.sample(n=10)
```

## Serialization

### Save to File

```python
eval_set.to_yaml("output.yaml")
eval_set.to_json("output.json")
```

### Export for Sharing

```python
# Export with all metadata
eval_set.export("benchmark_v1.zip")
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | str | Set name |
| `test_cases` | list | All test cases |
| `size` | int | Number of test cases |
| `tags` | set | All unique tags |
| `version` | str | Version string |

## Built-in Benchmarks

```python
from pyai.evaluation.benchmarks import (
    MMLU,
    HellaSwag,
    TruthfulQA,
    HumanEval
)

# Load standard benchmark
mmlu = MMLU.load(subset="computer_science")
results = evaluator.evaluate(mmlu, agent=my_agent)
```

## See Also

- [Evaluation-Module](Evaluation-Module) - Module overview
- [TestCase](TestCase) - Individual test cases
- [Evaluator](Evaluator) - Running evaluations
