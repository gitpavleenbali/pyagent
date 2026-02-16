# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Evaluation Criteria

Different criteria for evaluating agent outputs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
import json
import re


@dataclass
class CriteriaResult:
    """Result from applying evaluation criteria.
    
    Attributes:
        passed: Whether the criteria was met
        score: Numeric score (0.0 to 1.0)
        reason: Explanation of the result
        details: Additional details
    """
    passed: bool
    score: float
    reason: str
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class EvalCriteria(ABC):
    """Base class for evaluation criteria.
    
    Implement custom criteria by subclassing:
    
        class MyCustomCriteria(EvalCriteria):
            def evaluate(self, actual: str, expected: str, context: dict) -> CriteriaResult:
                # Your logic here
                return CriteriaResult(passed=True, score=1.0, reason="...")
    """
    
    name: str = "base_criteria"
    
    @abstractmethod
    def evaluate(
        self,
        actual: str,
        expected: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> CriteriaResult:
        """Evaluate the actual output against expected.
        
        Args:
            actual: The agent's actual output
            expected: Expected output (optional)
            context: Additional context from TestCase
            
        Returns:
            CriteriaResult with pass/fail and score
        """
        pass


class ExactMatch(EvalCriteria):
    """Exact string match criteria.
    
    Example:
        criteria = ExactMatch(ignore_case=True, ignore_whitespace=True)
        result = criteria.evaluate("Hello World!", "hello world!")
        # result.passed = True (case ignored)
    """
    
    name = "exact_match"
    
    def __init__(
        self,
        ignore_case: bool = False,
        ignore_whitespace: bool = False,
        strip: bool = True
    ):
        self.ignore_case = ignore_case
        self.ignore_whitespace = ignore_whitespace
        self.strip = strip
    
    def evaluate(
        self,
        actual: str,
        expected: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> CriteriaResult:
        if expected is None:
            return CriteriaResult(
                passed=False,
                score=0.0,
                reason="No expected value provided for exact match"
            )
        
        a = actual
        e = expected
        
        if self.strip:
            a = a.strip()
            e = e.strip()
        
        if self.ignore_case:
            a = a.lower()
            e = e.lower()
        
        if self.ignore_whitespace:
            a = " ".join(a.split())
            e = " ".join(e.split())
        
        passed = a == e
        
        return CriteriaResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=f"Exact match {'succeeded' if passed else 'failed'}",
            details={"actual_normalized": a, "expected_normalized": e}
        )


class ContainsMatch(EvalCriteria):
    """Check if output contains expected substrings.
    
    Example:
        criteria = ContainsMatch(substrings=["hello", "world"], match_all=True)
        result = criteria.evaluate("Hello world!")
        # result.passed = True
    """
    
    name = "contains_match"
    
    def __init__(
        self,
        substrings: Optional[List[str]] = None,
        match_all: bool = True,
        ignore_case: bool = True
    ):
        self.substrings = substrings or []
        self.match_all = match_all
        self.ignore_case = ignore_case
    
    def evaluate(
        self,
        actual: str,
        expected: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> CriteriaResult:
        # Get substrings from context if not provided
        substrs = self.substrings
        if not substrs and context:
            substrs = context.get("expected_contains", [])
        if not substrs and expected:
            substrs = [expected]
        
        if not substrs:
            return CriteriaResult(
                passed=True,
                score=1.0,
                reason="No substrings to check"
            )
        
        check_actual = actual.lower() if self.ignore_case else actual
        
        found = []
        missing = []
        
        for substr in substrs:
            check_substr = substr.lower() if self.ignore_case else substr
            if check_substr in check_actual:
                found.append(substr)
            else:
                missing.append(substr)
        
        if self.match_all:
            passed = len(missing) == 0
        else:
            passed = len(found) > 0
        
        score = len(found) / len(substrs) if substrs else 1.0
        
        return CriteriaResult(
            passed=passed,
            score=score,
            reason=f"Found {len(found)}/{len(substrs)} expected substrings",
            details={"found": found, "missing": missing}
        )


class NotContainsMatch(EvalCriteria):
    """Check that output does NOT contain forbidden strings.
    
    Example:
        criteria = NotContainsMatch(forbidden=["password", "secret"])
        result = criteria.evaluate("Hello user!")
        # result.passed = True (no forbidden strings found)
    """
    
    name = "not_contains_match"
    
    def __init__(
        self,
        forbidden: Optional[List[str]] = None,
        ignore_case: bool = True
    ):
        self.forbidden = forbidden or []
        self.ignore_case = ignore_case
    
    def evaluate(
        self,
        actual: str,
        expected: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> CriteriaResult:
        # Get forbidden from context if not provided
        forbidden = self.forbidden
        if not forbidden and context:
            forbidden = context.get("expected_not_contains", [])
        
        if not forbidden:
            return CriteriaResult(
                passed=True,
                score=1.0,
                reason="No forbidden strings to check"
            )
        
        check_actual = actual.lower() if self.ignore_case else actual
        
        found_forbidden = []
        
        for word in forbidden:
            check_word = word.lower() if self.ignore_case else word
            if check_word in check_actual:
                found_forbidden.append(word)
        
        passed = len(found_forbidden) == 0
        score = 1.0 - (len(found_forbidden) / len(forbidden))
        
        return CriteriaResult(
            passed=passed,
            score=max(0.0, score),
            reason=f"Found {len(found_forbidden)} forbidden strings" if found_forbidden 
                   else "No forbidden strings found",
            details={"found_forbidden": found_forbidden}
        )


class RegexMatch(EvalCriteria):
    """Check if output matches a regex pattern.
    
    Example:
        criteria = RegexMatch(pattern=r"\\d{4}-\\d{2}-\\d{2}")
        result = criteria.evaluate("Date: 2024-01-15")
        # result.passed = True
    """
    
    name = "regex_match"
    
    def __init__(
        self,
        pattern: Optional[str] = None,
        flags: int = 0
    ):
        self.pattern = pattern
        self.flags = flags
    
    def evaluate(
        self,
        actual: str,
        expected: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> CriteriaResult:
        pattern = self.pattern or expected
        
        if not pattern:
            return CriteriaResult(
                passed=False,
                score=0.0,
                reason="No regex pattern provided"
            )
        
        try:
            match = re.search(pattern, actual, self.flags)
            passed = match is not None
            
            return CriteriaResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                reason=f"Regex {'matched' if passed else 'did not match'}",
                details={
                    "pattern": pattern,
                    "match": match.group() if match else None
                }
            )
        except re.error as e:
            return CriteriaResult(
                passed=False,
                score=0.0,
                reason=f"Invalid regex: {e}",
                details={"error": str(e)}
            )


class JSONSchema(EvalCriteria):
    """Validate output against JSON schema.
    
    Example:
        criteria = JSONSchema(schema={"type": "object", "required": ["name"]})
        result = criteria.evaluate('{"name": "John"}')
        # result.passed = True
    """
    
    name = "json_schema"
    
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        self.schema = schema
    
    def evaluate(
        self,
        actual: str,
        expected: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> CriteriaResult:
        schema = self.schema
        if not schema and context:
            schema = context.get("expected_schema")
        
        if not schema:
            return CriteriaResult(
                passed=False,
                score=0.0,
                reason="No JSON schema provided"
            )
        
        # Try to parse JSON
        try:
            # Extract JSON from markdown code blocks if needed
            json_str = actual
            if "```json" in actual:
                match = re.search(r"```json\s*(.*?)\s*```", actual, re.DOTALL)
                if match:
                    json_str = match.group(1)
            elif "```" in actual:
                match = re.search(r"```\s*(.*?)\s*```", actual, re.DOTALL)
                if match:
                    json_str = match.group(1)
            
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return CriteriaResult(
                passed=False,
                score=0.0,
                reason=f"Invalid JSON: {e}",
                details={"error": str(e)}
            )
        
        # Basic schema validation (without jsonschema dependency)
        # For full validation, use: pip install jsonschema
        try:
            passed, errors = self._validate_schema(data, schema)
            
            return CriteriaResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                reason="JSON schema validation " + ("passed" if passed else "failed"),
                details={"errors": errors, "parsed_data": data}
            )
        except Exception as e:
            return CriteriaResult(
                passed=False,
                score=0.0,
                reason=f"Schema validation error: {e}",
                details={"error": str(e)}
            )
    
    def _validate_schema(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str = ""
    ) -> tuple:
        """Basic JSON schema validation."""
        errors = []
        
        # Type check
        if "type" in schema:
            expected_type = schema["type"]
            type_map = {
                "object": dict,
                "array": list,
                "string": str,
                "number": (int, float),
                "integer": int,
                "boolean": bool,
                "null": type(None),
            }
            if expected_type in type_map:
                if not isinstance(data, type_map[expected_type]):
                    errors.append(f"{path or 'root'}: expected {expected_type}")
        
        # Required fields
        if "required" in schema and isinstance(data, dict):
            for field in schema["required"]:
                if field not in data:
                    errors.append(f"{path or 'root'}: missing required field '{field}'")
        
        # Properties
        if "properties" in schema and isinstance(data, dict):
            for key, prop_schema in schema["properties"].items():
                if key in data:
                    _, prop_errors = self._validate_schema(
                        data[key], prop_schema, f"{path}.{key}" if path else key
                    )
                    errors.extend(prop_errors)
        
        return len(errors) == 0, errors


class LengthCheck(EvalCriteria):
    """Check output length constraints.
    
    Example:
        criteria = LengthCheck(min_length=10, max_length=100)
        result = criteria.evaluate("This is a medium length response.")
    """
    
    name = "length_check"
    
    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        unit: str = "chars"  # "chars", "words", "lines"
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.unit = unit
    
    def evaluate(
        self,
        actual: str,
        expected: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> CriteriaResult:
        if self.unit == "words":
            length = len(actual.split())
        elif self.unit == "lines":
            length = len(actual.strip().split("\n"))
        else:
            length = len(actual)
        
        min_ok = self.min_length is None or length >= self.min_length
        max_ok = self.max_length is None or length <= self.max_length
        passed = min_ok and max_ok
        
        # Calculate score
        if passed:
            score = 1.0
        elif self.min_length and length < self.min_length:
            score = length / self.min_length
        elif self.max_length and length > self.max_length:
            score = self.max_length / length
        else:
            score = 0.0
        
        constraints = []
        if self.min_length:
            constraints.append(f"min={self.min_length}")
        if self.max_length:
            constraints.append(f"max={self.max_length}")
        
        return CriteriaResult(
            passed=passed,
            score=min(1.0, max(0.0, score)),
            reason=f"Length {length} {self.unit} ({'within' if passed else 'outside'} {', '.join(constraints)})",
            details={"length": length, "unit": self.unit}
        )


class SemanticSimilarity(EvalCriteria):
    """Check semantic similarity using embeddings.
    
    Requires sentence-transformers or Azure OpenAI embeddings.
    
    Example:
        criteria = SemanticSimilarity(threshold=0.8)
        result = criteria.evaluate("Hello!", "Hi there!")
        # result.passed = True (semantically similar)
    """
    
    name = "semantic_similarity"
    
    def __init__(
        self,
        threshold: float = 0.8,
        model: str = "all-MiniLM-L6-v2"
    ):
        self.threshold = threshold
        self.model_name = model
        self._model = None
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "SemanticSimilarity requires sentence-transformers. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def evaluate(
        self,
        actual: str,
        expected: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> CriteriaResult:
        if not expected:
            return CriteriaResult(
                passed=False,
                score=0.0,
                reason="No expected value for semantic comparison"
            )
        
        try:
            model = self._load_model()
            
            embeddings = model.encode([actual, expected])
            
            # Cosine similarity
            from numpy import dot
            from numpy.linalg import norm
            
            similarity = dot(embeddings[0], embeddings[1]) / (
                norm(embeddings[0]) * norm(embeddings[1])
            )
            
            passed = similarity >= self.threshold
            
            return CriteriaResult(
                passed=passed,
                score=float(similarity),
                reason=f"Semantic similarity: {similarity:.2f} (threshold: {self.threshold})",
                details={"similarity": float(similarity), "threshold": self.threshold}
            )
            
        except ImportError as e:
            return CriteriaResult(
                passed=False,
                score=0.0,
                reason=str(e),
                details={"error": str(e)}
            )
        except Exception as e:
            return CriteriaResult(
                passed=False,
                score=0.0,
                reason=f"Similarity check failed: {e}",
                details={"error": str(e)}
            )


class LLMJudge(EvalCriteria):
    """Use an LLM to judge output quality.
    
    Like Google ADK's LLM-based evaluation.
    
    Example:
        criteria = LLMJudge(
            prompt="Is this response helpful and accurate?",
            model="gpt-4"
        )
        result = criteria.evaluate("Paris is the capital of France.")
    """
    
    name = "llm_judge"
    
    def __init__(
        self,
        prompt: Optional[str] = None,
        model: str = "gpt-4o-mini",
        rubric: Optional[str] = None
    ):
        self.prompt = prompt or (
            "You are an expert evaluator. Assess if the response is correct, "
            "helpful, and appropriate. Respond with JSON: "
            '{"passed": true/false, "score": 0.0-1.0, "reason": "..."}'
        )
        self.model = model
        self.rubric = rubric
    
    def evaluate(
        self,
        actual: str,
        expected: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> CriteriaResult:
        try:
            # Try to import and use our model registry
            from ..models import get_model
            
            judge_prompt = f"""
{self.prompt}

{"Rubric: " + self.rubric if self.rubric else ""}

{"Expected Output: " + expected if expected else ""}

Actual Output:
{actual}

Respond with valid JSON only: {{"passed": true/false, "score": 0.0-1.0, "reason": "..."}}
"""
            
            model = get_model(self.model)
            response = model.complete(judge_prompt)
            
            # Parse response
            try:
                # Extract JSON from response
                json_match = re.search(r'\{[^}]+\}', response.content)
                if json_match:
                    result = json.loads(json_match.group())
                    return CriteriaResult(
                        passed=result.get("passed", False),
                        score=float(result.get("score", 0.0)),
                        reason=result.get("reason", "No reason provided"),
                        details={"llm_response": response.content}
                    )
            except json.JSONDecodeError:
                pass
            
            # Fallback: simple keyword detection
            passed = any(word in response.content.lower() for word in ["yes", "correct", "good", "passed", "true"])
            
            return CriteriaResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                reason=response.content[:200],
                details={"llm_response": response.content}
            )
            
        except Exception as e:
            return CriteriaResult(
                passed=False,
                score=0.0,
                reason=f"LLM judge failed: {e}",
                details={"error": str(e)}
            )


class CustomCriteria(EvalCriteria):
    """Create custom criteria from a function.
    
    Example:
        def check_polite(actual: str, expected: str, context: dict) -> tuple:
            is_polite = "please" in actual.lower() or "thank" in actual.lower()
            return is_polite, 1.0 if is_polite else 0.0, "Politeness check"
        
        criteria = CustomCriteria(check_polite, name="politeness")
    """
    
    def __init__(
        self,
        func: Callable[[str, Optional[str], Optional[Dict]], tuple],
        name: str = "custom"
    ):
        self.func = func
        self.name = name
    
    def evaluate(
        self,
        actual: str,
        expected: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> CriteriaResult:
        try:
            result = self.func(actual, expected, context)
            
            if isinstance(result, CriteriaResult):
                return result
            elif isinstance(result, tuple):
                passed, score, reason = result[0], result[1], result[2] if len(result) > 2 else ""
                return CriteriaResult(passed=passed, score=score, reason=reason)
            elif isinstance(result, bool):
                return CriteriaResult(passed=result, score=1.0 if result else 0.0, reason="")
            else:
                return CriteriaResult(
                    passed=False,
                    score=0.0,
                    reason=f"Invalid return type from custom criteria: {type(result)}"
                )
        except Exception as e:
            return CriteriaResult(
                passed=False,
                score=0.0,
                reason=f"Custom criteria error: {e}",
                details={"error": str(e)}
            )


class CompositeCriteria(EvalCriteria):
    """Combine multiple criteria with AND/OR logic.
    
    Example:
        criteria = CompositeCriteria(
            [ContainsMatch(["hello"]), LengthCheck(max_length=100)],
            mode="all"  # All must pass
        )
    """
    
    name = "composite"
    
    def __init__(
        self,
        criteria: List[EvalCriteria],
        mode: str = "all",  # "all" or "any"
        weights: Optional[List[float]] = None
    ):
        self.criteria = criteria
        self.mode = mode
        self.weights = weights or [1.0] * len(criteria)
    
    def evaluate(
        self,
        actual: str,
        expected: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> CriteriaResult:
        results = []
        for criterion in self.criteria:
            results.append(criterion.evaluate(actual, expected, context))
        
        if self.mode == "all":
            passed = all(r.passed for r in results)
        else:
            passed = any(r.passed for r in results)
        
        # Weighted average score
        total_weight = sum(self.weights)
        score = sum(r.score * w for r, w in zip(results, self.weights)) / total_weight
        
        return CriteriaResult(
            passed=passed,
            score=score,
            reason=f"Composite ({self.mode}): {sum(1 for r in results if r.passed)}/{len(results)} passed",
            details={
                "individual_results": [
                    {"criteria": c.name, "passed": r.passed, "score": r.score}
                    for c, r in zip(self.criteria, results)
                ]
            }
        )


# Convenience aliases
exact_match = ExactMatch
contains = ContainsMatch
not_contains = NotContainsMatch
regex = RegexMatch
json_schema = JSONSchema
length = LengthCheck
semantic = SemanticSimilarity
llm_judge = LLMJudge
custom = CustomCriteria
composite = CompositeCriteria
