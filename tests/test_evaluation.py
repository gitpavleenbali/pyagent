# pyright: reportMissingImports=false, reportUnusedVariable=false, reportGeneralTypeIssues=false
# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Tests for the evaluation module.

Tests agent evaluation including:
- TestCase creation
- EvalSet management
- Evaluation criteria
- Evaluator
- Compare agents
"""

import pytest
import json
import tempfile
import os


class TestTestCase:
    """Tests for TestCase class."""
    
    def test_testcase_import(self):
        """Test that TestCase can be imported."""
        from pyai.evaluation import TestCase
        assert TestCase is not None
    
    def test_basic_testcase(self):
        """Test creating a basic test case."""
        from pyai.evaluation import TestCase
        
        tc = TestCase(
            input="What is 2+2?",
            expected="4"
        )
        
        assert tc.input == "What is 2+2?"
        assert tc.expected == "4"
        assert tc.id is not None
    
    def test_testcase_with_contains(self):
        """Test test case with expected_contains."""
        from pyai.evaluation import TestCase
        
        tc = TestCase(
            input="Tell me about Python",
            expected_contains=["programming", "language"]
        )
        
        assert tc.expected_contains == ["programming", "language"]
    
    def test_testcase_with_not_contains(self):
        """Test test case with expected_not_contains."""
        from pyai.evaluation import TestCase
        
        tc = TestCase(
            input="What is the capital of France?",
            expected_not_contains=["London", "Berlin"]
        )
        
        assert tc.expected_not_contains == ["London", "Berlin"]
    
    def test_testcase_with_tags(self):
        """Test test case with tags."""
        from pyai.evaluation import TestCase
        
        tc = TestCase(
            input="Calculate 5*5",
            expected="25",
            tags=["math", "multiplication"]
        )
        
        assert "math" in tc.tags
        assert "multiplication" in tc.tags
    
    def test_testcase_to_dict(self):
        """Test test case serialization."""
        from pyai.evaluation import TestCase
        
        tc = TestCase(
            input="Hello",
            expected="Hi",
            tags=["greeting"]
        )
        
        d = tc.to_dict()
        
        assert d["input"] == "Hello"
        assert d["expected"] == "Hi"
        assert "greeting" in d["tags"]
    
    def test_testcase_from_dict(self):
        """Test test case deserialization."""
        from pyai.evaluation import TestCase
        
        data = {
            "input": "Test input",
            "expected": "Test output",
            "tags": ["test"]
        }
        
        tc = TestCase.from_dict(data)
        
        assert tc.input == "Test input"
        assert tc.expected == "Test output"


class TestEvalSet:
    """Tests for EvalSet class."""
    
    def test_evalset_import(self):
        """Test that EvalSet can be imported."""
        from pyai.evaluation import EvalSet
        assert EvalSet is not None
    
    def test_create_evalset(self):
        """Test creating an eval set."""
        from pyai.evaluation import EvalSet, TestCase
        
        eval_set = EvalSet([
            TestCase(input="Q1", expected="A1"),
            TestCase(input="Q2", expected="A2"),
        ])
        
        assert len(eval_set) == 2
    
    def test_evalset_add(self):
        """Test adding test cases."""
        from pyai.evaluation import EvalSet, TestCase
        
        eval_set = EvalSet()
        eval_set.add(TestCase(input="Q1", expected="A1"))
        
        assert len(eval_set) == 1
    
    def test_evalset_filter_by_tags(self):
        """Test filtering by tags."""
        from pyai.evaluation import EvalSet, TestCase
        
        eval_set = EvalSet([
            TestCase(input="Math Q", expected="Math A", tags=["math"]),
            TestCase(input="Science Q", expected="Science A", tags=["science"]),
            TestCase(input="Both Q", expected="Both A", tags=["math", "science"]),
        ])
        
        math_only = eval_set.filter(tags=["math"])
        
        assert len(math_only) == 2  # "Math Q" and "Both Q"
    
    def test_evalset_to_json(self):
        """Test JSON serialization."""
        from pyai.evaluation import EvalSet, TestCase
        
        eval_set = EvalSet([
            TestCase(input="Q1", expected="A1"),
        ], name="test-set")
        
        json_str = eval_set.to_json()
        data = json.loads(json_str)
        
        assert data["name"] == "test-set"
        assert len(data["test_cases"]) == 1
    
    def test_evalset_from_json(self):
        """Test JSON deserialization."""
        from pyai.evaluation import EvalSet
        
        json_str = json.dumps({
            "name": "loaded-set",
            "test_cases": [
                {"input": "Q1", "expected": "A1"}
            ]
        })
        
        eval_set = EvalSet.from_json(json_str)
        
        assert eval_set.name == "loaded-set"
        assert len(eval_set) == 1
    
    def test_evalset_save_and_load(self):
        """Test saving and loading from file."""
        from pyai.evaluation import EvalSet, TestCase
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "tests.evalset.json")
            
            # Save
            eval_set = EvalSet([
                TestCase(input="Q1", expected="A1"),
            ], name="file-test")
            eval_set.save(filepath)
            
            # Load
            loaded = EvalSet.from_file(filepath)
            
            assert loaded.name == "file-test"
            assert len(loaded) == 1


class TestEvalCriteria:
    """Tests for evaluation criteria."""
    
    def test_exact_match(self):
        """Test ExactMatch criteria."""
        from pyai.evaluation import ExactMatch
        
        criteria = ExactMatch()
        
        result = criteria.evaluate("hello", "hello")
        assert result.passed is True
        assert result.score == 1.0
        
        result = criteria.evaluate("hello", "world")
        assert result.passed is False
        assert result.score == 0.0
    
    def test_exact_match_ignore_case(self):
        """Test ExactMatch with ignore_case."""
        from pyai.evaluation import ExactMatch
        
        criteria = ExactMatch(ignore_case=True)
        
        result = criteria.evaluate("HELLO", "hello")
        assert result.passed is True
    
    def test_contains_match(self):
        """Test ContainsMatch criteria."""
        from pyai.evaluation import ContainsMatch
        
        criteria = ContainsMatch(substrings=["hello", "world"])
        
        result = criteria.evaluate("hello world!", expected=None)
        assert result.passed is True
        
        result = criteria.evaluate("just hello", expected=None)
        assert result.passed is False  # Missing "world"
    
    def test_contains_match_any(self):
        """Test ContainsMatch with match_all=False."""
        from pyai.evaluation import ContainsMatch
        
        criteria = ContainsMatch(substrings=["hello", "world"], match_all=False)
        
        result = criteria.evaluate("just hello", expected=None)
        assert result.passed is True  # Has at least one
    
    def test_not_contains_match(self):
        """Test NotContainsMatch criteria."""
        from pyai.evaluation import NotContainsMatch
        
        criteria = NotContainsMatch(forbidden=["password", "secret"])
        
        result = criteria.evaluate("This is safe text")
        assert result.passed is True
        
        result = criteria.evaluate("Here is the password")
        assert result.passed is False
    
    def test_regex_match(self):
        """Test RegexMatch criteria."""
        from pyai.evaluation import RegexMatch
        
        criteria = RegexMatch(pattern=r"\d{4}-\d{2}-\d{2}")
        
        result = criteria.evaluate("Date: 2024-01-15")
        assert result.passed is True
        
        result = criteria.evaluate("No date here")
        assert result.passed is False
    
    def test_json_schema_criteria(self):
        """Test JSONSchema criteria."""
        from pyai.evaluation import JSONSchema
        
        criteria = JSONSchema(schema={
            "type": "object",
            "required": ["name"]
        })
        
        result = criteria.evaluate('{"name": "John"}')
        assert result.passed is True
        
        result = criteria.evaluate('{"age": 30}')
        assert result.passed is False  # Missing "name"
    
    def test_length_check(self):
        """Test LengthCheck criteria."""
        from pyai.evaluation import LengthCheck
        
        criteria = LengthCheck(min_length=5, max_length=20)
        
        result = criteria.evaluate("Hello World")
        assert result.passed is True
        
        result = criteria.evaluate("Hi")
        assert result.passed is False  # Too short
        
        result = criteria.evaluate("This is a very long text that exceeds the limit")
        assert result.passed is False  # Too long
    
    def test_custom_criteria(self):
        """Test CustomCriteria."""
        from pyai.evaluation import CustomCriteria
        
        def check_uppercase(actual, expected, context):
            is_upper = actual.isupper()
            return is_upper, 1.0 if is_upper else 0.0, "Uppercase check"
        
        criteria = CustomCriteria(check_uppercase)
        
        result = criteria.evaluate("HELLO WORLD")
        assert result.passed is True
        
        result = criteria.evaluate("Hello World")
        assert result.passed is False
    
    def test_composite_criteria(self):
        """Test CompositeCriteria."""
        from pyai.evaluation import CompositeCriteria, LengthCheck, ContainsMatch
        
        criteria = CompositeCriteria([
            LengthCheck(min_length=5),
            ContainsMatch(substrings=["hello"]),
        ], mode="all")
        
        result = criteria.evaluate("hello world")
        assert result.passed is True
        
        result = criteria.evaluate("hi")
        assert result.passed is False


class TestEvaluator:
    """Tests for Evaluator class."""
    
    def test_evaluator_import(self):
        """Test that Evaluator can be imported."""
        from pyai.evaluation import Evaluator
        assert Evaluator is not None
    
    def test_evaluator_with_callable(self):
        """Test evaluator with a simple callable."""
        from pyai.evaluation import Evaluator, EvalSet, TestCase
        
        # Simple agent that echoes
        def echo_agent(input_text):
            return input_text.upper()
        
        eval_set = EvalSet([
            TestCase(input="hello", expected="HELLO"),
            TestCase(input="world", expected="WORLD"),
        ])
        
        evaluator = Evaluator(echo_agent)
        metrics = evaluator.evaluate(eval_set)
        
        assert metrics.total == 2
        assert metrics.passed == 2
        assert metrics.accuracy == 1.0
    
    def test_evaluator_with_failures(self):
        """Test evaluator with failing tests."""
        from pyai.evaluation import Evaluator, EvalSet, TestCase
        
        def wrong_agent(input_text):
            return "always wrong"
        
        eval_set = EvalSet([
            TestCase(input="hello", expected="hello"),
            TestCase(input="world", expected="world"),
        ])
        
        evaluator = Evaluator(wrong_agent)
        metrics = evaluator.evaluate(eval_set)
        
        assert metrics.total == 2
        assert metrics.failed == 2
        assert metrics.accuracy == 0.0
    
    def test_evaluator_metrics(self):
        """Test metrics calculation."""
        from pyai.evaluation import Evaluator, EvalSet, TestCase
        
        def partial_agent(input_text):
            if "pass" in input_text:
                return "correct"
            return "wrong"
        
        eval_set = EvalSet([
            TestCase(input="pass1", expected="correct"),
            TestCase(input="pass2", expected="correct"),
            TestCase(input="fail1", expected="correct"),
        ])
        
        evaluator = Evaluator(partial_agent)
        metrics = evaluator.evaluate(eval_set)
        
        assert metrics.total == 3
        assert metrics.passed == 2
        assert metrics.failed == 1
        assert abs(metrics.accuracy - 0.667) < 0.01
    
    def test_evaluator_summary(self):
        """Test metrics summary generation."""
        from pyai.evaluation import Evaluator, EvalSet, TestCase
        
        def echo_agent(x):
            return x
        
        eval_set = EvalSet([
            TestCase(input="test", expected="test"),
        ])
        
        evaluator = Evaluator(echo_agent)
        metrics = evaluator.evaluate(eval_set)
        
        summary = metrics.summary()
        
        assert "Total" in summary
        assert "Passed" in summary
        assert "Accuracy" in summary


class TestEvaluateAgent:
    """Tests for evaluate_agent convenience function."""
    
    def test_evaluate_agent_function(self):
        """Test evaluate_agent function."""
        from pyai.evaluation import evaluate_agent, TestCase
        
        def simple_agent(x):
            return x.upper()
        
        metrics = evaluate_agent(
            simple_agent,
            [TestCase(input="hello", expected="HELLO")]
        )
        
        assert metrics.passed == 1
    
    def test_evaluate_agent_with_dicts(self):
        """Test evaluate_agent with dict test cases."""
        from pyai.evaluation import evaluate_agent
        
        def simple_agent(x):
            return x.upper()
        
        metrics = evaluate_agent(
            simple_agent,
            [{"input": "hello", "expected": "HELLO"}],
            verbose=False
        )
        
        assert metrics.passed == 1


class TestCompareAgents:
    """Tests for compare_agents function."""
    
    def test_compare_agents_function(self):
        """Test compare_agents function."""
        from pyai.evaluation import compare_agents, TestCase
        
        def agent1(x):
            return "CORRECT ANSWER"
        
        def agent2(x):
            return "wrong answer"
        
        eval_set = [
            TestCase(input="Hello", expected="CORRECT ANSWER"),
        ]
        
        results = compare_agents(
            {"correct": agent1, "wrong": agent2},
            eval_set,
            verbose=False
        )
        
        assert "correct" in results
        assert "wrong" in results
        assert results["correct"].passed == 1
        assert results["wrong"].passed == 0


class TestEvalStatus:
    """Tests for EvalStatus enum."""
    
    def test_eval_status_values(self):
        """Test EvalStatus enum values."""
        from pyai.evaluation import EvalStatus
        
        assert EvalStatus.PASSED.value == "passed"
        assert EvalStatus.FAILED.value == "failed"
        assert EvalStatus.ERROR.value == "error"
        assert EvalStatus.SKIPPED.value == "skipped"


class TestEvalIntegration:
    """Integration tests for evaluation module."""
    
    def test_module_exports(self):
        """Test that all expected exports are available."""
        from pyai import evaluation
        
        assert hasattr(evaluation, "TestCase")
        assert hasattr(evaluation, "EvalSet")
        assert hasattr(evaluation, "Evaluator")
        assert hasattr(evaluation, "evaluate_agent")
        assert hasattr(evaluation, "compare_agents")
        assert hasattr(evaluation, "ExactMatch")
        assert hasattr(evaluation, "ContainsMatch")
    
    def test_main_init_exports(self):
        """Test that evaluation is exported from main pyai."""
        import pyai
        
        assert hasattr(pyai, "evaluation")
        assert hasattr(pyai, "evaluate_agent")
        assert hasattr(pyai, "EvalSet")
        assert hasattr(pyai, "TestCase")
    
    def test_full_evaluation_workflow(self):
        """Test complete evaluation workflow."""
        from pyai.evaluation import (
            EvalSet, TestCase, Evaluator, EvalConfig,
            ExactMatch, ContainsMatch
        )
        
        # Create test set
        eval_set = EvalSet([
            TestCase(
                input="What is the capital of France?",
                expected_contains=["Paris"]
            ),
            TestCase(
                input="What is 2+2?",
                expected="4"
            ),
        ], name="geography-math")
        
        # Mock agent
        def smart_agent(question):
            if "France" in question:
                return "The capital of France is Paris."
            if "2+2" in question:
                return "4"
            return "I don't know"
        
        # Evaluate
        config = EvalConfig(verbose=False)
        evaluator = Evaluator(smart_agent, config)
        metrics = evaluator.evaluate(eval_set)
        
        # Verify
        assert metrics.total == 2
        assert metrics.passed == 2
        assert metrics.accuracy == 1.0
