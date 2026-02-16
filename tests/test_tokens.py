"""
Tests for PyAgent Token Counting Module.

Tests token counting and cost calculation utilities.
"""

import pytest
from typing import List, Dict


# =============================================================================
# Token Counter Tests
# =============================================================================

class TestTokenCounter:
    """Tests for TokenCounter class."""
    
    def test_counter_initialization(self):
        """Test counter initialization."""
        from pyagent.tokens.counter import TokenCounter
        
        counter = TokenCounter("gpt-4")
        assert counter.model == "gpt-4"
    
    def test_counter_with_different_models(self):
        """Test counter with different models."""
        from pyagent.tokens.counter import TokenCounter
        
        # OpenAI
        counter = TokenCounter("gpt-4")
        assert counter.model == "gpt-4"
        
        # Claude
        counter = TokenCounter("claude-3-opus")
        assert counter.model == "claude-3-opus"
    
    def test_count_string(self):
        """Test counting tokens in string."""
        from pyagent.tokens.counter import TokenCounter
        
        counter = TokenCounter("gpt-4")
        result = counter.count("Hello, how are you?")
        
        assert result.input_tokens > 0
        assert result.output_tokens == 0
        assert result.total_tokens == result.input_tokens
    
    def test_count_output(self):
        """Test counting output tokens."""
        from pyagent.tokens.counter import TokenCounter
        
        counter = TokenCounter("gpt-4")
        result = counter.count("This is the response.", is_output=True)
        
        assert result.input_tokens == 0
        assert result.output_tokens > 0
        assert result.total_tokens == result.output_tokens
    
    def test_count_messages(self):
        """Test counting tokens in message list."""
        from pyagent.tokens.counter import TokenCounter
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        
        counter = TokenCounter("gpt-4")
        result = counter.count_messages(messages)
        
        assert result.input_tokens > 0
    
    def test_count_messages_with_completion(self):
        """Test counting with completion."""
        from pyagent.tokens.counter import TokenCounter
        
        messages = [
            {"role": "user", "content": "Hello!"},
        ]
        completion = "Hi there! How can I help you?"
        
        counter = TokenCounter("gpt-4")
        result = counter.count_messages(messages, completion=completion)
        
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        assert result.total_tokens == result.input_tokens + result.output_tokens
    
    def test_token_count_dataclass(self):
        """Test TokenCount dataclass."""
        from pyagent.tokens.counter import TokenCount
        
        count = TokenCount(
            input_tokens=100,
            output_tokens=50,
            model="gpt-4",
            method="tiktoken"
        )
        
        assert count.input_tokens == 100
        assert count.output_tokens == 50
        assert count.total_tokens == 150
        assert count.model == "gpt-4"
    
    def test_estimate_tokens(self):
        """Test token estimation."""
        from pyagent.tokens.counter import TokenCounter
        
        # Use a model that forces estimation
        counter = TokenCounter("unknown-model")
        result = counter.count("Hello, world! This is a test message.")
        
        assert result.input_tokens > 0
        assert result.method == "char_estimate"


class TestCountTokensFunction:
    """Tests for count_tokens utility function."""
    
    def test_count_tokens_string(self):
        """Test counting tokens in string."""
        from pyagent.tokens import count_tokens
        
        tokens = count_tokens("Hello, world!")
        assert tokens > 0
    
    def test_count_tokens_with_model(self):
        """Test counting with specific model."""
        from pyagent.tokens import count_tokens
        
        tokens = count_tokens("Test message", model="gpt-4")
        assert tokens > 0
    
    def test_count_tokens_empty(self):
        """Test counting empty string."""
        from pyagent.tokens import count_tokens
        
        tokens = count_tokens("")
        # Should be at least 1 (minimum)
        assert tokens >= 0


class TestEstimateTokensFunction:
    """Tests for estimate_tokens utility function."""
    
    def test_estimate_tokens(self):
        """Test basic estimation."""
        from pyagent.tokens.counter import estimate_tokens
        
        # 40 chars / 4 chars per token = 10 tokens
        tokens = estimate_tokens("1234567890" * 4)
        assert tokens == 10
    
    def test_estimate_tokens_custom_ratio(self):
        """Test estimation with custom ratio."""
        from pyagent.tokens.counter import estimate_tokens
        
        # 40 chars / 8 chars per token = 5 tokens
        tokens = estimate_tokens("1234567890" * 4, chars_per_token=8)
        assert tokens == 5


# =============================================================================
# Cost Calculator Tests
# =============================================================================

class TestTokenCost:
    """Tests for TokenCost class."""
    
    def test_token_cost_dataclass(self):
        """Test TokenCost initialization."""
        from pyagent.tokens.cost import TokenCost
        
        cost = TokenCost(
            input_cost=0.03,
            output_cost=0.06,
            model="gpt-4",
            input_tokens=1000,
            output_tokens=500
        )
        
        assert cost.input_cost == 0.03
        assert cost.output_cost == 0.06
        assert cost.total_cost == 0.09
    
    def test_token_cost_to_dict(self):
        """Test TokenCost serialization."""
        from pyagent.tokens.cost import TokenCost
        
        cost = TokenCost(
            input_cost=0.01,
            output_cost=0.02,
            model="gpt-4"
        )
        
        result = cost.to_dict()
        assert "input_cost" in result
        assert "output_cost" in result
        assert "total_cost" in result
    
    def test_token_cost_str(self):
        """Test TokenCost string representation."""
        from pyagent.tokens.cost import TokenCost
        
        cost = TokenCost(
            input_tokens=100,
            output_tokens=50,
            total_cost=0.005,
            model="gpt-4"
        )
        
        s = str(cost)
        assert "gpt-4" in s
        assert "100" in s


class TestCalculateCost:
    """Tests for calculate_cost function."""
    
    def test_calculate_cost_basic(self):
        """Test basic cost calculation."""
        from pyagent.tokens.cost import calculate_cost
        
        cost = calculate_cost(1000, 500, model="gpt-4")
        
        assert cost.input_tokens == 1000
        assert cost.output_tokens == 500
        assert cost.total_cost > 0
    
    def test_calculate_cost_gpt4(self):
        """Test GPT-4 pricing."""
        from pyagent.tokens.cost import calculate_cost
        
        # GPT-4: $0.03/1K input, $0.06/1K output
        cost = calculate_cost(1000, 1000, model="gpt-4")
        
        # 1000 input tokens * $0.03/1K = $0.03
        # 1000 output tokens * $0.06/1K = $0.06
        # Total = $0.09
        assert abs(cost.input_cost - 0.03) < 0.001
        assert abs(cost.output_cost - 0.06) < 0.001
        assert abs(cost.total_cost - 0.09) < 0.001
    
    def test_calculate_cost_gpt4o_mini(self):
        """Test GPT-4o-mini pricing (cheaper model)."""
        from pyagent.tokens.cost import calculate_cost
        
        cost = calculate_cost(1000, 1000, model="gpt-4o-mini")
        
        # Much cheaper than GPT-4
        assert cost.total_cost < 0.01
    
    def test_calculate_cost_claude(self):
        """Test Claude pricing."""
        from pyagent.tokens.cost import calculate_cost
        
        cost = calculate_cost(1000, 1000, model="claude-3-opus")
        
        assert cost.total_cost > 0
        assert cost.model == "claude-3-opus"
    
    def test_calculate_cost_with_token_count(self):
        """Test cost calculation with TokenCount input."""
        from pyagent.tokens.counter import TokenCount
        from pyagent.tokens.cost import calculate_cost
        
        token_count = TokenCount(
            input_tokens=1000,
            output_tokens=500,
            model="gpt-4"
        )
        
        cost = calculate_cost(token_count)
        
        assert cost.input_tokens == 1000
        assert cost.output_tokens == 500


class TestModelPricing:
    """Tests for model pricing lookup."""
    
    def test_model_pricing_exists(self):
        """Test MODEL_PRICING dictionary exists."""
        from pyagent.tokens.cost import MODEL_PRICING
        
        assert "gpt-4" in MODEL_PRICING
        assert "claude-3-opus" in MODEL_PRICING
    
    def test_get_model_pricing(self):
        """Test get_model_pricing function."""
        from pyagent.tokens.cost import get_model_pricing
        
        input_rate, output_rate = get_model_pricing("gpt-4")
        
        assert input_rate > 0
        assert output_rate > 0
    
    def test_get_model_pricing_prefix_match(self):
        """Test prefix matching for model variants."""
        from pyagent.tokens.cost import get_model_pricing
        
        # Should match gpt-4-turbo prefix
        input_rate, output_rate = get_model_pricing("gpt-4-turbo-2024")
        
        assert input_rate > 0


class TestCostTracker:
    """Tests for CostTracker class."""
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        from pyagent.tokens.cost import CostTracker
        
        tracker = CostTracker(model="gpt-4")
        
        assert tracker.model == "gpt-4"
        assert tracker.request_count == 0
        assert tracker.total_cost == 0.0
    
    def test_tracker_add(self):
        """Test adding tokens to tracker."""
        from pyagent.tokens.cost import CostTracker
        
        tracker = CostTracker(model="gpt-4")
        
        cost = tracker.add(1000, 500)
        
        assert cost.input_tokens == 1000
        assert cost.output_tokens == 500
        assert tracker.request_count == 1
    
    def test_tracker_multiple_adds(self):
        """Test multiple adds."""
        from pyagent.tokens.cost import CostTracker
        
        tracker = CostTracker(model="gpt-4")
        
        tracker.add(1000, 500)
        tracker.add(800, 300)
        tracker.add(1200, 600)
        
        assert tracker.request_count == 3
        assert tracker.total_input_tokens == 3000
        assert tracker.total_output_tokens == 1400
    
    def test_tracker_total_cost(self):
        """Test total cost calculation."""
        from pyagent.tokens.cost import CostTracker
        
        tracker = CostTracker(model="gpt-4")
        
        tracker.add(1000, 1000)  # $0.09
        tracker.add(1000, 1000)  # $0.09
        
        # Total should be ~$0.18
        assert abs(tracker.total_cost - 0.18) < 0.01
    
    def test_tracker_average_cost(self):
        """Test average cost per request."""
        from pyagent.tokens.cost import CostTracker
        
        tracker = CostTracker(model="gpt-4")
        
        tracker.add(1000, 1000)
        tracker.add(1000, 1000)
        
        avg = tracker.average_cost
        assert avg > 0
        assert avg == tracker.total_cost / 2
    
    def test_tracker_summary(self):
        """Test tracker summary."""
        from pyagent.tokens.cost import CostTracker
        
        tracker = CostTracker(model="gpt-4")
        tracker.add(1000, 500)
        
        summary = tracker.summary()
        
        assert "model" in summary
        assert "request_count" in summary
        assert "total_cost" in summary
        assert summary["request_count"] == 1
    
    def test_tracker_reset(self):
        """Test tracker reset."""
        from pyagent.tokens.cost import CostTracker
        
        tracker = CostTracker(model="gpt-4")
        tracker.add(1000, 500)
        tracker.add(800, 300)
        
        tracker.reset()
        
        assert tracker.request_count == 0
        assert tracker.total_cost == 0.0
        assert tracker.total_input_tokens == 0


class TestEstimateMonthly:
    """Tests for monthly cost estimation."""
    
    def test_estimate_monthly_cost(self):
        """Test monthly cost estimation."""
        from pyagent.tokens.cost import estimate_monthly_cost
        
        estimate = estimate_monthly_cost(
            requests_per_day=100,
            avg_input_tokens=500,
            avg_output_tokens=200,
            model="gpt-4"
        )
        
        assert "daily_cost" in estimate
        assert "monthly_cost" in estimate
        assert estimate["requests_per_day"] == 100
        assert estimate["monthly_cost"] > estimate["daily_cost"]


# =============================================================================
# Integration Tests
# =============================================================================

class TestTokensIntegration:
    """Integration tests for tokens module."""
    
    def test_count_and_calculate(self):
        """Test combined counting and cost calculation."""
        from pyagent.tokens.counter import TokenCounter
        from pyagent.tokens.cost import calculate_cost
        
        # Count tokens
        counter = TokenCounter("gpt-4")
        messages = [
            {"role": "user", "content": "Write a haiku about coding."},
        ]
        completion = "Code flows like water\nBugs appear then disappear\nStack Overflow helps"
        
        count = counter.count_messages(messages, completion=completion)
        
        # Calculate cost
        cost = calculate_cost(count)
        
        assert cost.total_cost > 0
    
    def test_import_from_main_package(self):
        """Test imports from main pyagent package."""
        from pyagent import TokenCounter, count_tokens, calculate_cost, CostTracker
        
        assert TokenCounter is not None
        assert count_tokens is not None
        assert calculate_cost is not None
        assert CostTracker is not None
    
    def test_import_tokens_module(self):
        """Test importing tokens module."""
        from pyagent.tokens import (
            TokenCounter,
            count_tokens,
            estimate_tokens,
            TokenCost,
            calculate_cost,
            MODEL_PRICING,
        )
        
        assert TokenCounter is not None
        assert count_tokens is not None
        assert calculate_cost is not None


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_string(self):
        """Test counting empty string."""
        from pyagent.tokens.counter import TokenCounter
        
        counter = TokenCounter("gpt-4")
        result = counter.count("")
        
        # Should handle gracefully
        assert result.input_tokens >= 0
    
    def test_very_long_text(self):
        """Test counting very long text."""
        from pyagent.tokens.counter import TokenCounter
        
        long_text = "word " * 10000  # 50000 characters
        
        counter = TokenCounter("gpt-4")
        result = counter.count(long_text)
        
        assert result.input_tokens > 1000
    
    def test_unknown_model_pricing(self):
        """Test pricing for unknown model."""
        from pyagent.tokens.cost import calculate_cost
        
        cost = calculate_cost(1000, 500, model="unknown-model-xyz")
        
        # Should use default pricing
        assert cost.total_cost > 0
    
    def test_zero_tokens(self):
        """Test cost with zero tokens."""
        from pyagent.tokens.cost import calculate_cost
        
        cost = calculate_cost(0, 0, model="gpt-4")
        
        assert cost.total_cost == 0.0
    
    def test_messages_with_list_content(self):
        """Test messages with list-based content (vision)."""
        from pyagent.tokens.counter import TokenCounter
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": "..."}
                ]
            }
        ]
        
        counter = TokenCounter("gpt-4")
        result = counter.count_messages(messages)
        
        # Should extract text content
        assert result.input_tokens > 0
