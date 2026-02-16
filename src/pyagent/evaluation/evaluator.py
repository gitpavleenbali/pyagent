# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Agent Evaluator

Run evaluations against agents and collect results.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import TestCase, EvalSet, EvalResult, EvalMetrics, EvalStatus
from .criteria import (
    EvalCriteria,
    ExactMatch,
    ContainsMatch,
    NotContainsMatch,
    JSONSchema,
    CompositeCriteria,
    CriteriaResult,
)


@dataclass
class EvalConfig:
    """Configuration for evaluation runs.
    
    Attributes:
        criteria: List of criteria to apply
        parallel: Run tests in parallel
        max_workers: Max parallel workers
        timeout_seconds: Per-test timeout
        fail_fast: Stop on first failure
        verbose: Print progress
    """
    criteria: Optional[List[EvalCriteria]] = None
    parallel: bool = True
    max_workers: int = 4
    timeout_seconds: float = 60.0
    fail_fast: bool = False
    verbose: bool = True


class Evaluator:
    """Run evaluations against agents.
    
    Inspired by Google ADK's evaluation framework.
    
    Example:
        from pyagent import Agent
        from pyagent.evaluation import Evaluator, EvalSet, TestCase
        
        # Create test cases
        eval_set = EvalSet([
            TestCase(input="What is 2+2?", expected="4"),
            TestCase(input="Hello!", expected_contains=["hello", "hi"]),
        ])
        
        # Create agent
        agent = Agent("math-helper")
        
        # Run evaluation
        evaluator = Evaluator(agent)
        metrics = evaluator.evaluate(eval_set)
        
        print(metrics.summary())
    """
    
    def __init__(
        self,
        agent: Any,
        config: Optional[EvalConfig] = None
    ):
        """Initialize evaluator.
        
        Args:
            agent: Agent to evaluate (any callable or Agent instance)
            config: Evaluation configuration
        """
        self.agent = agent
        self.config = config or EvalConfig()
        
        # Default criteria if none provided
        if not self.config.criteria:
            self.config.criteria = self._get_default_criteria()
    
    def _get_default_criteria(self) -> List[EvalCriteria]:
        """Get default criteria based on test case fields."""
        return [
            ExactMatch(ignore_case=True, strip=True),
            ContainsMatch(match_all=True, ignore_case=True),
            NotContainsMatch(ignore_case=True),
            JSONSchema(),
        ]
    
    def _run_agent(self, input_text: str, context: Optional[Dict] = None) -> tuple:
        """Run agent and measure time.
        
        Returns:
            Tuple of (output, latency_ms, tokens_used)
        """
        start = time.perf_counter()
        tokens = 0
        
        try:
            # Handle different agent types
            if hasattr(self.agent, "run"):
                # PyAgent Agent
                result = self.agent.run(input_text)
                if hasattr(result, "output"):
                    output = result.output
                    if hasattr(result, "usage"):
                        tokens = getattr(result.usage, "total_tokens", 0)
                else:
                    output = str(result)
            elif hasattr(self.agent, "invoke"):
                # LangChain-style
                result = self.agent.invoke({"input": input_text})
                output = result.get("output", str(result))
            elif callable(self.agent):
                # Simple callable
                output = self.agent(input_text)
                if not isinstance(output, str):
                    output = str(output)
            else:
                raise ValueError(f"Unsupported agent type: {type(self.agent)}")
            
            latency_ms = (time.perf_counter() - start) * 1000
            return output, latency_ms, tokens
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            raise RuntimeError(f"Agent execution failed: {e}") from e
    
    def _evaluate_test(self, test_case: TestCase) -> EvalResult:
        """Evaluate a single test case."""
        try:
            # Run agent
            output, latency_ms, tokens = self._run_agent(
                test_case.input,
                test_case.context
            )
            
            # Apply criteria
            all_passed = True
            total_score = 0.0
            criteria_count = 0
            details = {"criteria_results": []}
            
            for criterion in self.config.criteria:
                # Skip inapplicable criteria
                if isinstance(criterion, ExactMatch) and not test_case.expected:
                    continue
                if isinstance(criterion, ContainsMatch) and not test_case.expected_contains:
                    continue
                if isinstance(criterion, NotContainsMatch) and not test_case.expected_not_contains:
                    continue
                if isinstance(criterion, JSONSchema) and not test_case.expected_schema:
                    continue
                
                # Build context for criteria
                ctx = test_case.context or {}
                ctx.update({
                    "expected_contains": test_case.expected_contains,
                    "expected_not_contains": test_case.expected_not_contains,
                    "expected_schema": test_case.expected_schema,
                })
                
                result = criterion.evaluate(
                    actual=output,
                    expected=test_case.expected,
                    context=ctx
                )
                
                all_passed = all_passed and result.passed
                total_score += result.score
                criteria_count += 1
                
                details["criteria_results"].append({
                    "criteria": criterion.name,
                    "passed": result.passed,
                    "score": result.score,
                    "reason": result.reason,
                })
            
            # Calculate final score
            avg_score = total_score / criteria_count if criteria_count > 0 else 1.0
            
            return EvalResult(
                test_case=test_case,
                status=EvalStatus.PASSED if all_passed else EvalStatus.FAILED,
                actual_output=output,
                score=avg_score,
                details=details,
                latency_ms=latency_ms,
                tokens_used=tokens,
            )
            
        except Exception as e:
            return EvalResult(
                test_case=test_case,
                status=EvalStatus.ERROR,
                actual_output="",
                score=0.0,
                error=str(e),
                details={"error": str(e)},
            )
    
    def evaluate(
        self,
        eval_set: Union[EvalSet, List[TestCase]],
        tags: Optional[List[str]] = None
    ) -> EvalMetrics:
        """Run evaluation on all test cases.
        
        Args:
            eval_set: Test cases to run
            tags: Only run tests with these tags
            
        Returns:
            EvalMetrics with aggregated results
        """
        # Convert to EvalSet if needed
        if isinstance(eval_set, list):
            eval_set = EvalSet(eval_set)
        
        # Filter by tags
        if tags:
            eval_set = eval_set.filter(tags=tags)
        
        metrics = EvalMetrics()
        start_time = time.perf_counter()
        
        if self.config.verbose:
            print(f"🧪 Running {len(eval_set)} test cases...")
        
        if self.config.parallel and len(eval_set) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self._evaluate_test, tc): tc
                    for tc in eval_set
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    metrics.add_result(result)
                    
                    if self.config.verbose:
                        status_icon = "✅" if result.passed else "❌"
                        print(f"  {status_icon} {result.test_case.id}: {result.status.value}")
                    
                    if self.config.fail_fast and not result.passed:
                        break
        else:
            # Sequential execution
            for test_case in eval_set:
                result = self._evaluate_test(test_case)
                metrics.add_result(result)
                
                if self.config.verbose:
                    status_icon = "✅" if result.passed else "❌"
                    print(f"  {status_icon} {test_case.id}: {result.status.value}")
                
                if self.config.fail_fast and not result.passed:
                    break
        
        metrics.duration_seconds = time.perf_counter() - start_time
        
        if self.config.verbose:
            print(metrics.summary())
        
        return metrics
    
    async def evaluate_async(
        self,
        eval_set: Union[EvalSet, List[TestCase]],
        tags: Optional[List[str]] = None
    ) -> EvalMetrics:
        """Async evaluation for async agents.
        
        Args:
            eval_set: Test cases to run
            tags: Only run tests with these tags
            
        Returns:
            EvalMetrics with aggregated results
        """
        # For async agents
        if isinstance(eval_set, list):
            eval_set = EvalSet(eval_set)
        
        if tags:
            eval_set = eval_set.filter(tags=tags)
        
        metrics = EvalMetrics()
        start_time = time.perf_counter()
        
        # Run with asyncio
        tasks = [
            self._evaluate_test_async(tc)
            for tc in eval_set
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                # Create error result
                metrics.add_result(EvalResult(
                    test_case=TestCase(input="unknown"),
                    status=EvalStatus.ERROR,
                    actual_output="",
                    error=str(result),
                ))
            else:
                metrics.add_result(result)
        
        metrics.duration_seconds = time.perf_counter() - start_time
        return metrics
    
    async def _evaluate_test_async(self, test_case: TestCase) -> EvalResult:
        """Async test evaluation."""
        try:
            start = time.perf_counter()
            
            # Handle async agents
            if hasattr(self.agent, "arun"):
                result = await self.agent.arun(test_case.input)
                output = result.output if hasattr(result, "output") else str(result)
            elif asyncio.iscoroutinefunction(self.agent):
                output = await self.agent(test_case.input)
            else:
                # Fall back to sync in executor
                loop = asyncio.get_event_loop()
                output, latency_ms, tokens = await loop.run_in_executor(
                    None, self._run_agent, test_case.input, test_case.context
                )
                # Apply criteria (same as sync version)
                return self._evaluate_test(test_case)
            
            latency_ms = (time.perf_counter() - start) * 1000
            
            # Apply criteria
            all_passed = True
            total_score = 0.0
            criteria_count = 0
            
            for criterion in self.config.criteria:
                if isinstance(criterion, ExactMatch) and not test_case.expected:
                    continue
                if isinstance(criterion, ContainsMatch) and not test_case.expected_contains:
                    continue
                
                ctx = test_case.context or {}
                ctx.update({
                    "expected_contains": test_case.expected_contains,
                    "expected_not_contains": test_case.expected_not_contains,
                })
                
                result = criterion.evaluate(output, test_case.expected, ctx)
                all_passed = all_passed and result.passed
                total_score += result.score
                criteria_count += 1
            
            avg_score = total_score / criteria_count if criteria_count > 0 else 1.0
            
            return EvalResult(
                test_case=test_case,
                status=EvalStatus.PASSED if all_passed else EvalStatus.FAILED,
                actual_output=output,
                score=avg_score,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            return EvalResult(
                test_case=test_case,
                status=EvalStatus.ERROR,
                actual_output="",
                error=str(e),
            )


def evaluate_agent(
    agent: Any,
    test_cases: Union[EvalSet, List[TestCase], List[Dict]],
    criteria: Optional[List[EvalCriteria]] = None,
    verbose: bool = True
) -> EvalMetrics:
    """Convenience function to evaluate an agent.
    
    One-liner evaluation:
    
        from pyagent.evaluation import evaluate_agent, TestCase
        
        metrics = evaluate_agent(
            my_agent,
            [TestCase(input="Hi", expected_contains=["hello"])]
        )
    
    Args:
        agent: Agent to evaluate
        test_cases: Test cases (EvalSet, list of TestCase, or list of dicts)
        criteria: Evaluation criteria (uses defaults if None)
        verbose: Print progress
        
    Returns:
        EvalMetrics with results
    """
    # Convert dicts to TestCase
    if test_cases and isinstance(test_cases[0], dict):
        test_cases = [TestCase.from_dict(tc) for tc in test_cases]
    
    config = EvalConfig(criteria=criteria, verbose=verbose)
    evaluator = Evaluator(agent, config)
    
    return evaluator.evaluate(test_cases)


def compare_agents(
    agents: Dict[str, Any],
    test_cases: Union[EvalSet, List[TestCase]],
    criteria: Optional[List[EvalCriteria]] = None,
    verbose: bool = True
) -> Dict[str, EvalMetrics]:
    """Compare multiple agents on the same test cases.
    
    Useful for A/B testing or model comparison.
    
    Example:
        from pyagent.evaluation import compare_agents, TestCase
        
        results = compare_agents(
            {
                "gpt-4": agent_gpt4,
                "gpt-3.5": agent_gpt35,
                "local": agent_ollama,
            },
            [TestCase(input="What is AI?", expected_contains=["artificial", "intelligence"])]
        )
        
        for name, metrics in results.items():
            print(f"{name}: {metrics.accuracy:.1%} accuracy")
    
    Args:
        agents: Dict of agent_name -> agent
        test_cases: Test cases to run
        criteria: Evaluation criteria
        verbose: Print progress
        
    Returns:
        Dict of agent_name -> EvalMetrics
    """
    results = {}
    
    for name, agent in agents.items():
        if verbose:
            print(f"\n📊 Evaluating: {name}")
        
        config = EvalConfig(criteria=criteria, verbose=verbose)
        evaluator = Evaluator(agent, config)
        
        results[name] = evaluator.evaluate(test_cases)
    
    # Print comparison summary
    if verbose:
        print("\n" + "=" * 50)
        print("📊 Comparison Summary:")
        print("=" * 50)
        for name, metrics in sorted(results.items(), key=lambda x: x[1].accuracy, reverse=True):
            print(f"  {name}: {metrics.accuracy:.1%} accuracy, {metrics.avg_latency_ms:.0f}ms avg latency")
    
    return results


def load_eval_set(filepath: str) -> EvalSet:
    """Load evaluation set from file.
    
    Supports .json, .yaml, .evalset.json formats.
    
    Args:
        filepath: Path to eval set file
        
    Returns:
        EvalSet loaded from file
    """
    import json
    
    if filepath.endswith(".yaml") or filepath.endswith(".yml"):
        try:
            import yaml
            with open(filepath) as f:
                data = yaml.safe_load(f)
            return EvalSet.from_dict(data)
        except ImportError:
            raise ImportError("YAML support requires pyyaml. Install with: pip install pyyaml")
    else:
        return EvalSet.from_file(filepath)


def create_eval_set(
    inputs: List[str],
    expected: Optional[List[str]] = None,
    name: str = "generated"
) -> EvalSet:
    """Quickly create an eval set from input/expected pairs.
    
    Example:
        eval_set = create_eval_set(
            inputs=["What is 2+2?", "What is 3+3?"],
            expected=["4", "6"]
        )
    """
    test_cases = []
    for i, input_text in enumerate(inputs):
        exp = expected[i] if expected and i < len(expected) else None
        test_cases.append(TestCase(input=input_text, expected=exp))
    
    return EvalSet(test_cases, name=name)
