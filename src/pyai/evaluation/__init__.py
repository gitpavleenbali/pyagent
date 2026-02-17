# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
pyai Evaluation Module

Inspired by Google ADK's evaluation/ module, this provides:

- Test case definition and eval sets
- Agent evaluation with multiple criteria
- Metrics collection and comparison
- Support for exact match, semantic similarity, LLM judges

Example:
    from pyai.evaluation import EvalSet, TestCase, evaluate_agent

    # Define test cases
    eval_set = EvalSet([
        TestCase(
            input="What is 2+2?",
            expected="4",
        ),
        TestCase(
            input="Summarize this text",
            expected_contains=["key point", "summary"],
        ),
    ])

    # Evaluate an agent
    results = evaluate_agent(agent, eval_set)
    print(results.summary())

    # Compare multiple agents
    from pyai.evaluation import compare_agents
    comparison = compare_agents(
        {"gpt-4": agent1, "gpt-3.5": agent2},
        eval_set
    )
"""

from typing import TYPE_CHECKING

# Type stubs for static analysis (pyright/pylance)
if TYPE_CHECKING:
    from .base import EvalMetrics, EvalResult, EvalSet, EvalStatus, TestCase
    from .criteria import (
        CompositeCriteria,
        ContainsMatch,
        CriteriaResult,
        CustomCriteria,
        EvalCriteria,
        ExactMatch,
        JSONSchema,
        LengthCheck,
        LLMJudge,
        NotContainsMatch,
        RegexMatch,
        SemanticSimilarity,
        composite,
        contains,
        custom,
        exact_match,
        json_schema,
        length,
        llm_judge,
        not_contains,
        regex,
        semantic,
    )
    from .evaluator import (
        EvalConfig,
        Evaluator,
        compare_agents,
        create_eval_set,
        evaluate_agent,
        load_eval_set,
    )


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy load evaluation components."""
    _base_exports = {"TestCase", "EvalSet", "EvalResult", "EvalMetrics", "EvalStatus"}
    _evaluator_exports = {
        "Evaluator",
        "EvalConfig",
        "evaluate_agent",
        "compare_agents",
        "load_eval_set",
        "create_eval_set",
    }
    _criteria_exports = {
        "EvalCriteria",
        "CriteriaResult",
        "ExactMatch",
        "ContainsMatch",
        "NotContainsMatch",
        "RegexMatch",
        "JSONSchema",
        "LengthCheck",
        "SemanticSimilarity",
        "LLMJudge",
        "CustomCriteria",
        "CompositeCriteria",
        "exact_match",
        "contains",
        "not_contains",
        "regex",
        "json_schema",
        "length",
        "semantic",
        "llm_judge",
        "custom",
        "composite",
    }

    if name in _base_exports:
        from . import base

        return getattr(base, name)
    elif name in _evaluator_exports:
        from . import evaluator

        return getattr(evaluator, name)
    elif name in _criteria_exports:
        from . import criteria

        return getattr(criteria, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core classes
    "TestCase",
    "EvalSet",
    "EvalResult",
    "EvalMetrics",
    "EvalStatus",
    # Evaluator
    "Evaluator",
    "EvalConfig",
    "evaluate_agent",
    "compare_agents",
    "load_eval_set",
    "create_eval_set",
    # Criteria classes
    "EvalCriteria",
    "CriteriaResult",
    "ExactMatch",
    "ContainsMatch",
    "NotContainsMatch",
    "RegexMatch",
    "JSONSchema",
    "LengthCheck",
    "SemanticSimilarity",
    "LLMJudge",
    "CustomCriteria",
    "CompositeCriteria",
    # Convenience aliases
    "exact_match",
    "contains",
    "not_contains",
    "regex",
    "json_schema",
    "length",
    "semantic",
    "llm_judge",
    "custom",
    "composite",
]
