# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Base Evaluation Classes

Core abstractions for agent evaluation.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class EvalStatus(Enum):
    """Evaluation result status."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class TestCase:
    """A single test case for agent evaluation.

    Attributes:
        id: Unique test case identifier
        input: Input prompt/message for the agent
        expected: Expected exact output (optional)
        expected_contains: Strings that should appear in output
        expected_not_contains: Strings that should NOT appear
        expected_schema: JSON schema for structured output
        context: Additional context for evaluation
        tags: Tags for filtering test cases
        metadata: Additional metadata
    """

    input: str
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    expected: Optional[str] = None
    expected_contains: Optional[List[str]] = None
    expected_not_contains: Optional[List[str]] = None
    expected_schema: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Multi-turn conversation support
    conversation: Optional[List[Dict[str, str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "input": self.input,
            "expected": self.expected,
            "expected_contains": self.expected_contains,
            "expected_not_contains": self.expected_not_contains,
            "expected_schema": self.expected_schema,
            "context": self.context,
            "tags": self.tags,
            "metadata": self.metadata,
            "conversation": self.conversation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            input=data["input"],
            expected=data.get("expected"),
            expected_contains=data.get("expected_contains"),
            expected_not_contains=data.get("expected_not_contains"),
            expected_schema=data.get("expected_schema"),
            context=data.get("context"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            conversation=data.get("conversation"),
        )


@dataclass
class EvalResult:
    """Result of evaluating a single test case.

    Attributes:
        test_case: The test case that was evaluated
        status: Pass/fail/error status
        actual_output: The agent's actual output
        score: Numeric score (0.0 to 1.0)
        details: Detailed evaluation results
        latency_ms: Response latency in milliseconds
        tokens_used: Tokens consumed
        error: Error message if status is ERROR
        timestamp: When evaluation was run
    """

    test_case: TestCase
    status: EvalStatus
    actual_output: str
    score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    tokens_used: int = 0
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def passed(self) -> bool:
        """Check if test passed."""
        return self.status == EvalStatus.PASSED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_case_id": self.test_case.id,
            "status": self.status.value,
            "actual_output": self.actual_output,
            "score": self.score,
            "details": self.details,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EvalMetrics:
    """Aggregated metrics from an evaluation run.

    Attributes:
        total: Total number of test cases
        passed: Number of passed tests
        failed: Number of failed tests
        errors: Number of tests with errors
        skipped: Number of skipped tests
        accuracy: Pass rate (0.0 to 1.0)
        avg_score: Average score across all tests
        avg_latency_ms: Average latency
        total_tokens: Total tokens used
        results: Individual test results
    """

    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    accuracy: float = 0.0
    avg_score: float = 0.0
    avg_latency_ms: float = 0.0
    total_tokens: int = 0
    results: List[EvalResult] = field(default_factory=list)
    duration_seconds: float = 0.0

    def add_result(self, result: EvalResult) -> None:
        """Add a result and update metrics."""
        self.results.append(result)
        self.total += 1

        if result.status == EvalStatus.PASSED:
            self.passed += 1
        elif result.status == EvalStatus.FAILED:
            self.failed += 1
        elif result.status == EvalStatus.ERROR:
            self.errors += 1
        else:
            self.skipped += 1

        self.total_tokens += result.tokens_used

        # Recalculate averages
        self._recalculate()

    def _recalculate(self) -> None:
        """Recalculate aggregate metrics."""
        if self.total > 0:
            self.accuracy = self.passed / self.total
            self.avg_score = sum(r.score for r in self.results) / self.total
            self.avg_latency_ms = sum(r.latency_ms for r in self.results) / self.total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "accuracy": self.accuracy,
            "avg_score": self.avg_score,
            "avg_latency_ms": self.avg_latency_ms,
            "total_tokens": self.total_tokens,
            "duration_seconds": self.duration_seconds,
            "results": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:
        """Generate a summary string."""
        return (
            f"📊 Evaluation Results:\n"
            f"   Total: {self.total}\n"
            f"   ✅ Passed: {self.passed}\n"
            f"   ❌ Failed: {self.failed}\n"
            f"   ⚠️  Errors: {self.errors}\n"
            f"   ⏭️  Skipped: {self.skipped}\n"
            f"   📈 Accuracy: {self.accuracy:.1%}\n"
            f"   ⭐ Avg Score: {self.avg_score:.2f}\n"
            f"   ⏱️  Avg Latency: {self.avg_latency_ms:.0f}ms\n"
            f"   🎟️  Total Tokens: {self.total_tokens:,}"
        )


class EvalSet:
    """A collection of test cases for evaluation.

    Like Google ADK's .evalset.json format.

    Example:
        # Create from list
        eval_set = EvalSet([
            TestCase(input="What is 2+2?", expected="4"),
            TestCase(input="Hello", expected_contains=["hi", "hello"]),
        ])

        # Load from file
        eval_set = EvalSet.from_file("tests.evalset.json")

        # Filter by tag
        math_tests = eval_set.filter(tags=["math"])
    """

    def __init__(
        self,
        test_cases: Optional[List[TestCase]] = None,
        name: str = "default",
        description: str = "",
    ):
        """Initialize evaluation set.

        Args:
            test_cases: List of test cases
            name: Name of the eval set
            description: Description of what this evaluates
        """
        self.test_cases = test_cases or []
        self.name = name
        self.description = description

    def add(self, test_case: TestCase) -> None:
        """Add a test case."""
        self.test_cases.append(test_case)

    def filter(
        self, tags: Optional[List[str]] = None, ids: Optional[List[str]] = None
    ) -> "EvalSet":
        """Filter test cases.

        Args:
            tags: Only include cases with these tags
            ids: Only include cases with these IDs

        Returns:
            New EvalSet with filtered cases
        """
        filtered = self.test_cases

        if tags:
            filtered = [tc for tc in filtered if any(t in tc.tags for t in tags)]
        if ids:
            filtered = [tc for tc in filtered if tc.id in ids]

        return EvalSet(filtered, name=f"{self.name}_filtered")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalSet":
        """Create from dictionary."""
        test_cases = [TestCase.from_dict(tc) for tc in data.get("test_cases", [])]
        return cls(
            test_cases=test_cases,
            name=data.get("name", "default"),
            description=data.get("description", ""),
        )

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "EvalSet":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, filepath: str) -> None:
        """Save to file.

        Args:
            filepath: Path to save (typically .evalset.json)
        """
        with open(filepath, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_file(cls, filepath: str) -> "EvalSet":
        """Load from file.

        Args:
            filepath: Path to .evalset.json file
        """
        with open(filepath, "r") as f:
            return cls.from_json(f.read())

    def __len__(self) -> int:
        return len(self.test_cases)

    def __iter__(self):
        return iter(self.test_cases)
