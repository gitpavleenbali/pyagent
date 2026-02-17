# Evaluation Module

The Evaluation module provides comprehensive tools for testing, benchmarking, and evaluating AI agent performance.

## Overview

```python
from pyai.evaluation import Evaluator, TestCase, EvalSet, EvalCriteria
```

## Key Components

| Component | Description |
|-----------|-------------|
| [Evaluator](Evaluator) | Main evaluation engine |
| [TestCase](TestCase) | Individual test case definition |
| [EvalSet](EvalSet) | Collection of test cases |
| EvalCriteria | Evaluation criteria and metrics |
| EvalResult | Evaluation results container |

## Quick Start

```python
from pyai.evaluation import Evaluator, TestCase, EvalSet

# Create test cases
test_cases = [
    TestCase(
        input="What is 2+2?",
        expected_output="4",
        criteria=["accuracy", "conciseness"]
    ),
    TestCase(
        input="Explain quantum computing",
        expected_output=None,  # Open-ended
        criteria=["relevance", "clarity"]
    )
]

# Create evaluation set
eval_set = EvalSet(name="Math Tests", test_cases=test_cases)

# Run evaluation
evaluator = Evaluator()
results = evaluator.evaluate(eval_set, agent=my_agent)

# View results
print(f"Pass Rate: {results.pass_rate}%")
print(f"Average Score: {results.avg_score}")
```

## Built-in Criteria

### Accuracy Criteria
```python
from pyai.evaluation.criteria import AccuracyCriteria

criteria = AccuracyCriteria(
    threshold=0.8,  # 80% accuracy required
    comparison_method="exact"  # or "semantic", "fuzzy"
)
```

### Custom Criteria
```python
from pyai.evaluation import EvalCriteria

class ToneCriteria(EvalCriteria):
    def evaluate(self, output: str, expected: str) -> float:
        # Custom evaluation logic
        if "professional" in output.lower():
            return 1.0
        return 0.5
```

## Batch Evaluation

```python
# Evaluate multiple agents
agents = [agent1, agent2, agent3]
comparison = evaluator.compare(eval_set, agents=agents)

# Generate comparison report
comparison.to_markdown("comparison_report.md")
comparison.to_json("comparison_results.json")
```

## Metrics

The evaluation module tracks:

- **Pass Rate**: Percentage of tests passed
- **Average Score**: Mean score across all criteria
- **Latency**: Response time metrics
- **Token Usage**: Token consumption per test
- **Cost**: Estimated API cost

## Integration with CI/CD

```yaml
# .github/workflows/eval.yml
- name: Run Agent Evaluation
  run: |
    python -m pyai.evaluation run \
      --eval-set tests/eval_cases.yaml \
      --threshold 0.85 \
      --output results.json
```

## See Also

- [Evaluator](Evaluator) - Detailed evaluator configuration
- [TestCase](TestCase) - Test case specification
- [EvalSet](EvalSet) - Managing evaluation sets
