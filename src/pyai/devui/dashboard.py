# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Agent Dashboard

Monitoring and metrics visualization for agents.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class AgentMetrics:
    """Metrics collected from an agent."""

    total_runs: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    average_latency_ms: float = 0.0
    success_rate: float = 1.0
    errors: List[str] = field(default_factory=list)
    last_run: Optional[datetime] = None


@dataclass
class RunRecord:
    """Record of a single agent run."""

    id: str
    input: str
    output: str
    started_at: datetime
    ended_at: datetime
    tokens_used: int = 0
    cost: float = 0.0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Get run duration in milliseconds."""
        return (self.ended_at - self.started_at).total_seconds() * 1000


class AgentDashboard:
    """Dashboard for monitoring agent performance.

    Tracks metrics, run history, and provides visualizations.

    Example:
        from pyai.devui import AgentDashboard

        dashboard = AgentDashboard()

        # Wrap agent
        wrapped_agent = dashboard.wrap(agent)

        # Use wrapped agent (metrics are collected)
        result = wrapped_agent.run("Hello")

        # Get metrics
        metrics = dashboard.get_metrics()
        print(f"Total runs: {metrics.total_runs}")
    """

    def __init__(self, max_history: int = 1000):
        """Initialize dashboard.

        Args:
            max_history: Maximum run records to keep
        """
        self.max_history = max_history
        self._runs: List[RunRecord] = []
        self._metrics = AgentMetrics()

    def record_run(
        self,
        input: str,
        output: str,
        started_at: datetime,
        ended_at: datetime,
        tokens_used: int = 0,
        cost: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
        **metadata,
    ) -> RunRecord:
        """Record a run.

        Args:
            input: Run input
            output: Run output
            started_at: Start time
            ended_at: End time
            tokens_used: Tokens consumed
            cost: Cost in USD
            success: Whether run succeeded
            error: Error message if failed
            **metadata: Additional metadata

        Returns:
            Run record
        """
        import uuid

        record = RunRecord(
            id=str(uuid.uuid4()),
            input=input,
            output=output,
            started_at=started_at,
            ended_at=ended_at,
            tokens_used=tokens_used,
            cost=cost,
            success=success,
            error=error,
            metadata=metadata,
        )

        self._runs.append(record)

        # Trim history
        if len(self._runs) > self.max_history:
            self._runs = self._runs[-self.max_history :]

        # Update metrics
        self._update_metrics(record)

        return record

    def _update_metrics(self, record: RunRecord):
        """Update metrics from a run record."""
        self._metrics.total_runs += 1
        self._metrics.total_tokens += record.tokens_used
        self._metrics.total_cost += record.cost
        self._metrics.last_run = record.ended_at

        if not record.success and record.error:
            self._metrics.errors.append(record.error)

        # Update rolling averages
        n = self._metrics.total_runs
        old_avg = self._metrics.average_latency_ms
        self._metrics.average_latency_ms = (old_avg * (n - 1) + record.duration_ms) / n

        success_count = sum(1 for r in self._runs if r.success)
        self._metrics.success_rate = success_count / len(self._runs)

    def get_metrics(self) -> AgentMetrics:
        """Get current metrics."""
        return self._metrics

    def get_runs(
        self,
        limit: int = 100,
        success_only: bool = False,
    ) -> List[RunRecord]:
        """Get run history.

        Args:
            limit: Maximum records to return
            success_only: Only return successful runs

        Returns:
            List of run records
        """
        runs = self._runs
        if success_only:
            runs = [r for r in runs if r.success]
        return runs[-limit:]

    def wrap(self, agent: Any) -> "TrackedAgent":
        """Wrap an agent for automatic tracking.

        Args:
            agent: Agent to wrap

        Returns:
            Tracked agent wrapper
        """
        return TrackedAgent(agent, self)

    def clear(self):
        """Clear all history and reset metrics."""
        self._runs.clear()
        self._metrics = AgentMetrics()

    def export_runs(self, format: str = "json") -> str:
        """Export run history.

        Args:
            format: Export format ("json" or "csv")

        Returns:
            Exported data string
        """
        import json

        if format == "json":
            return json.dumps(
                [
                    {
                        "id": r.id,
                        "input": r.input,
                        "output": r.output,
                        "started_at": r.started_at.isoformat(),
                        "ended_at": r.ended_at.isoformat(),
                        "duration_ms": r.duration_ms,
                        "tokens_used": r.tokens_used,
                        "cost": r.cost,
                        "success": r.success,
                        "error": r.error,
                    }
                    for r in self._runs
                ],
                indent=2,
            )

        elif format == "csv":
            lines = ["id,input,output,started_at,duration_ms,tokens,cost,success"]
            for r in self._runs:
                lines.append(
                    f'"{r.id}","{r.input[:50]}","{r.output[:50]}",'
                    f"{r.started_at.isoformat()},{r.duration_ms:.0f},"
                    f"{r.tokens_used},{r.cost:.4f},{r.success}"
                )
            return "\n".join(lines)

        raise ValueError(f"Unknown format: {format}")


class TrackedAgent:
    """Wrapper that tracks agent runs."""

    def __init__(self, agent: Any, dashboard: AgentDashboard):
        self._agent = agent
        self._dashboard = dashboard

    def run(self, input: str, **kwargs) -> Any:
        """Run the agent with tracking."""
        started_at = datetime.utcnow()

        try:
            result = self._agent.run(input, **kwargs)
            ended_at = datetime.utcnow()

            # Extract output
            if hasattr(result, "output"):
                output = result.output
            elif isinstance(result, str):
                output = result
            else:
                output = str(result)

            # Extract tokens if available
            tokens = 0
            if hasattr(result, "tokens_used"):
                tokens = result.tokens_used

            self._dashboard.record_run(
                input=input,
                output=output,
                started_at=started_at,
                ended_at=ended_at,
                tokens_used=tokens,
                success=True,
            )

            return result

        except Exception as e:
            ended_at = datetime.utcnow()

            self._dashboard.record_run(
                input=input,
                output="",
                started_at=started_at,
                ended_at=ended_at,
                success=False,
                error=str(e),
            )

            raise

    def __getattr__(self, name):
        """Forward attribute access to wrapped agent."""
        return getattr(self._agent, name)
