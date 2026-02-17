# Evaluator

The `Evaluator` class is the main engine for running agent evaluations and benchmarks.

## Import

```python
from pyagent.evaluation import Evaluator
```

## Constructor

```python
Evaluator(
    model: str = None,           # Model for LLM-based evaluation
    criteria: list = None,        # Default criteria
    verbose: bool = False,        # Enable verbose logging
    parallel: bool = True,        # Run tests in parallel
    max_workers: int = 4          # Number of parallel workers
)
```

## Methods

### evaluate()

Run evaluation on a test set.

```python
def evaluate(
    self,
    eval_set: EvalSet | list[TestCase],
    agent: Agent | Callable,
    criteria: list = None
) -> EvalResult:
```

**Example:**
```python
evaluator = Evaluator()
results = evaluator.evaluate(
    eval_set=my_tests,
    agent=my_agent,
    criteria=["accuracy", "latency"]
)
```

### compare()

Compare multiple agents on the same test set.

```python
def compare(
    self,
    eval_set: EvalSet,
    agents: list[Agent],
    criteria: list = None
) -> ComparisonResult:
```

**Example:**
```python
comparison = evaluator.compare(
    eval_set=benchmark,
    agents=[gpt4_agent, claude_agent, local_agent]
)

# View comparison table
print(comparison.to_table())
```

### add_criteria()

Register custom evaluation criteria.

```python
evaluator.add_criteria("custom_metric", my_criteria_function)
```

## EvalResult

The evaluation result object contains:

```python
result.pass_rate      # Percentage of passed tests
result.avg_score      # Average score (0-1)
result.total_tests    # Number of tests run
result.passed_tests   # Number of tests passed
result.failed_tests   # Number of tests failed
result.details        # Per-test results
result.metrics        # Aggregated metrics
result.duration       # Total evaluation time
```

## Configuration

### YAML Configuration

```yaml
# eval_config.yaml
evaluator:
  model: gpt-4
  parallel: true
  max_workers: 8
  
criteria:
  - name: accuracy
    threshold: 0.9
  - name: latency
    max_ms: 2000
```

```python
evaluator = Evaluator.from_config("eval_config.yaml")
```

## Callbacks

Register callbacks for evaluation events:

```python
def on_test_complete(test_case, result):
    print(f"Test {test_case.name}: {result.score}")

evaluator.on_test_complete = on_test_complete
```

## See Also

- [Evaluation-Module](Evaluation-Module) - Module overview
- [TestCase](TestCase) - Test case definition
- [EvalSet](EvalSet) - Test set management
