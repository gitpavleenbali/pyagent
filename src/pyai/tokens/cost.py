# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Token Cost Calculator

Calculate costs for API usage based on token counts.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

from .counter import TokenCount

# Model pricing per 1000 tokens (in USD)
# Format: (input_cost_per_1k, output_cost_per_1k)
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # OpenAI GPT-4 models
    "gpt-4": (0.03, 0.06),
    "gpt-4-32k": (0.06, 0.12),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4-turbo-preview": (0.01, 0.03),
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    # OpenAI GPT-3.5 models
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "gpt-3.5-turbo-16k": (0.003, 0.004),
    "gpt-3.5-turbo-instruct": (0.0015, 0.002),
    # Anthropic Claude models
    "claude-3-opus": (0.015, 0.075),
    "claude-3-sonnet": (0.003, 0.015),
    "claude-3-haiku": (0.00025, 0.00125),
    "claude-3.5-sonnet": (0.003, 0.015),
    "claude-2": (0.008, 0.024),
    "claude-2.1": (0.008, 0.024),
    # Google Gemini models
    "gemini-pro": (0.00025, 0.0005),
    "gemini-ultra": (0.00075, 0.0015),
    "gemini-1.5-pro": (0.00125, 0.005),
    "gemini-1.5-flash": (0.000075, 0.0003),
    # OpenAI Embedding models
    "text-embedding-ada-002": (0.0001, 0.0),
    "text-embedding-3-small": (0.00002, 0.0),
    "text-embedding-3-large": (0.00013, 0.0),
}


@dataclass
class TokenCost:
    """Cost calculation result.

    Attributes:
        input_cost: Cost for input tokens in USD
        output_cost: Cost for output tokens in USD
        total_cost: Total cost in USD
        model: Model used for pricing
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
    """

    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    model: str = "unknown"
    input_tokens: int = 0
    output_tokens: int = 0
    input_rate: float = 0.0
    output_rate: float = 0.0

    def __post_init__(self):
        if self.total_cost == 0:
            self.total_cost = self.input_cost + self.output_cost

    def __str__(self) -> str:
        return (
            f"TokenCost(model={self.model}, "
            f"tokens={self.input_tokens}+{self.output_tokens}, "
            f"cost=${self.total_cost:.6f})"
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
            "input_rate_per_1k": self.input_rate,
            "output_rate_per_1k": self.output_rate,
        }


def calculate_cost(
    input_tokens: Union[int, TokenCount], output_tokens: int = 0, model: str = "gpt-4"
) -> TokenCost:
    """Calculate cost for token usage.

    Args:
        input_tokens: Number of input tokens or TokenCount object
        output_tokens: Number of output tokens (if input_tokens is int)
        model: Model name for pricing lookup

    Returns:
        TokenCost with cost breakdown

    Example:
        cost = calculate_cost(1000, 500, model="gpt-4")
        print(f"Total: ${cost.total_cost:.4f}")

        # Or with TokenCount
        count = counter.count_messages(messages, completion)
        cost = calculate_cost(count, model="gpt-4")
    """
    # Handle TokenCount input
    if isinstance(input_tokens, TokenCount):
        token_count = input_tokens
        input_tokens = token_count.input_tokens
        output_tokens = token_count.output_tokens
        if model == "gpt-4" and token_count.model != "unknown":
            model = token_count.model

    # Get pricing
    pricing = get_model_pricing(model)
    input_rate, output_rate = pricing

    # Calculate costs (pricing is per 1000 tokens)
    input_cost = (input_tokens / 1000) * input_rate
    output_cost = (output_tokens / 1000) * output_rate

    return TokenCost(
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_rate=input_rate,
        output_rate=output_rate,
    )


def get_model_pricing(model: str) -> Tuple[float, float]:
    """Get pricing for a model.

    Args:
        model: Model name

    Returns:
        Tuple of (input_cost_per_1k, output_cost_per_1k)
    """
    # Check exact match
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]

    # Check prefix match
    for model_prefix, pricing in MODEL_PRICING.items():
        if model.startswith(model_prefix):
            return pricing

    # Default pricing (GPT-4-like)
    return (0.03, 0.06)


def estimate_monthly_cost(
    requests_per_day: int, avg_input_tokens: int, avg_output_tokens: int, model: str = "gpt-4"
) -> Dict[str, float]:
    """Estimate monthly cost.

    Args:
        requests_per_day: Average requests per day
        avg_input_tokens: Average input tokens per request
        avg_output_tokens: Average output tokens per request
        model: Model name

    Returns:
        Dict with daily and monthly cost estimates
    """
    # Calculate cost per request
    cost_per_request = calculate_cost(avg_input_tokens, avg_output_tokens, model)

    daily_cost = cost_per_request.total_cost * requests_per_day
    monthly_cost = daily_cost * 30

    return {
        "model": model,
        "requests_per_day": requests_per_day,
        "cost_per_request": cost_per_request.total_cost,
        "daily_cost": daily_cost,
        "monthly_cost": monthly_cost,
        "yearly_cost": daily_cost * 365,
    }


class CostTracker:
    """Track cumulative costs.

    Example:
        tracker = CostTracker(model="gpt-4")

        # Track each request
        tracker.add(1000, 500)
        tracker.add(800, 300)

        print(f"Total cost: ${tracker.total_cost:.4f}")
    """

    def __init__(self, model: str = "gpt-4"):
        """Initialize tracker.

        Args:
            model: Default model for pricing
        """
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0
        self._costs: list = []

    def add(
        self,
        input_tokens: Union[int, TokenCount],
        output_tokens: int = 0,
        model: Optional[str] = None,
    ) -> TokenCost:
        """Add a request's tokens.

        Args:
            input_tokens: Input tokens or TokenCount
            output_tokens: Output tokens
            model: Override model

        Returns:
            Cost for this request
        """
        cost = calculate_cost(input_tokens, output_tokens, model or self.model)

        self.total_input_tokens += cost.input_tokens
        self.total_output_tokens += cost.output_tokens
        self.request_count += 1
        self._costs.append(cost)

        return cost

    @property
    def total_cost(self) -> float:
        """Total cost in USD."""
        return sum(c.total_cost for c in self._costs)

    @property
    def average_cost(self) -> float:
        """Average cost per request."""
        if self.request_count == 0:
            return 0.0
        return self.total_cost / self.request_count

    def summary(self) -> Dict:
        """Get tracking summary."""
        return {
            "model": self.model,
            "request_count": self.request_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "average_cost_per_request": self.average_cost,
        }

    def reset(self):
        """Reset tracking."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0
        self._costs.clear()
