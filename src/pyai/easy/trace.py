"""
trace - Tracing and observability for AI operations

Debug, monitor, and analyze your AI operations with built-in tracing.
Inspired by OpenAI Agents' tracing but with pyai's simplicity.

Examples:
    >>> from pyai import trace, ask

    # Enable tracing
    >>> trace.enable()
    >>> answer = ask("What is Python?")
    >>> trace.show()  # Show trace of last operation

    # Trace context manager
    >>> with trace.span("research_task") as span:
    ...     result = research("AI trends")
    ...     span.log("Found results")

    # Export traces
    >>> trace.export("traces.json")

    # Custom handlers
    >>> @trace.handler
    ... def my_logger(event):
    ...     print(f"[{event.type}] {event.message}")
"""

import functools
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List


@dataclass
class TraceEvent:
    """A single trace event."""

    type: str  # "start", "end", "log", "error", "llm_call", "tool_call"
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    span_id: str = None
    parent_span_id: str = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
        }

    def __repr__(self) -> str:
        return f"TraceEvent({self.type}: {self.message[:50]}...)"


@dataclass
class Span:
    """A trace span representing a unit of work."""

    name: str
    span_id: str = field(default_factory=lambda: f"span_{int(time.time() * 1000)}")
    parent_span_id: str = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = None
    events: List[TraceEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "running"  # "running", "completed", "error"

    def log(self, message: str, **metadata) -> None:
        """Log an event within this span."""
        self.events.append(
            TraceEvent(type="log", message=message, span_id=self.span_id, metadata=metadata)
        )

    def error(self, message: str, exception: Exception = None) -> None:
        """Log an error event."""
        self.events.append(
            TraceEvent(
                type="error",
                message=message,
                span_id=self.span_id,
                metadata={"exception": str(exception) if exception else None},
            )
        )
        self.status = "error"

    def end(self) -> None:
        """End this span."""
        self.end_time = datetime.now()
        if self.status == "running":
            self.status = "completed"

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "events": [e.to_dict() for e in self.events],
            "metadata": self.metadata,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.error(str(exc_val), exc_val)
        self.end()


class Tracer:
    """Main tracer class for tracking AI operations."""

    def __init__(self):
        self._enabled = False
        self._spans: List[Span] = []
        self._current_span: Span = None
        self._handlers: List[Callable] = []
        self._lock = threading.Lock()

    def enable(self) -> None:
        """Enable tracing."""
        self._enabled = True

    def disable(self) -> None:
        """Disable tracing."""
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def span(self, name: str, **metadata) -> Span:
        """
        Create a new trace span.

        Args:
            name: Span name
            **metadata: Additional metadata

        Returns:
            Span context manager

        Examples:
            >>> with trace.span("my_operation") as span:
            ...     span.log("Starting work")
            ...     result = do_work()
        """
        parent_id = self._current_span.span_id if self._current_span else None
        span = Span(name=name, parent_span_id=parent_id, metadata=metadata)

        with self._lock:
            self._spans.append(span)
            self._current_span = span

        self._emit(
            TraceEvent(
                type="start",
                message=f"Started span: {name}",
                span_id=span.span_id,
                parent_span_id=parent_id,
            )
        )

        return span

    def log(self, message: str, **metadata) -> None:
        """Log an event."""
        if not self._enabled:
            return

        event = TraceEvent(
            type="log",
            message=message,
            span_id=self._current_span.span_id if self._current_span else None,
            metadata=metadata,
        )

        if self._current_span:
            self._current_span.events.append(event)

        self._emit(event)

    def llm_call(
        self,
        provider: str,
        model: str,
        prompt: str,
        response: str = None,
        tokens: int = None,
        duration_ms: float = None,
    ) -> None:
        """Log an LLM API call."""
        if not self._enabled:
            return

        event = TraceEvent(
            type="llm_call",
            message=f"LLM call to {provider}/{model}",
            duration_ms=duration_ms,
            span_id=self._current_span.span_id if self._current_span else None,
            metadata={
                "provider": provider,
                "model": model,
                "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "response_preview": (
                    response[:100] + "..." if response and len(response) > 100 else response
                ),
                "tokens": tokens,
            },
        )

        if self._current_span:
            self._current_span.events.append(event)

        self._emit(event)

    def tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any] = None,
        result: Any = None,
        duration_ms: float = None,
    ) -> None:
        """Log a tool call."""
        if not self._enabled:
            return

        event = TraceEvent(
            type="tool_call",
            message=f"Tool call: {tool_name}",
            duration_ms=duration_ms,
            span_id=self._current_span.span_id if self._current_span else None,
            metadata={
                "tool": tool_name,
                "args": args,
                "result_preview": str(result)[:100] if result else None,
            },
        )

        if self._current_span:
            self._current_span.events.append(event)

        self._emit(event)

    def handler(self, func: Callable) -> Callable:
        """
        Register a trace event handler.

        Args:
            func: Function to call for each trace event

        Examples:
            >>> @trace.handler
            ... def log_events(event):
            ...     print(f"[{event.type}] {event.message}")
        """
        self._handlers.append(func)
        return func

    def _emit(self, event: TraceEvent) -> None:
        """Emit an event to all handlers."""
        for handler in self._handlers:
            try:
                handler(event)
            except Exception:
                pass  # Don't let handler errors break tracing

    def show(self, last_n: int = 10) -> None:
        """
        Display recent traces.

        Args:
            last_n: Number of recent spans to show
        """
        print("\n" + "=" * 60)
        print("TRACE OUTPUT")
        print("=" * 60)

        for span in self._spans[-last_n:]:
            status_icon = (
                "✅" if span.status == "completed" else "❌" if span.status == "error" else "🔄"
            )
            print(f"\n{status_icon} {span.name} ({span.duration_ms:.1f}ms)")
            print(f"   ID: {span.span_id}")

            for event in span.events:
                icon = {"log": "📝", "error": "❌", "llm_call": "🤖", "tool_call": "🔧"}.get(
                    event.type, "•"
                )
                print(f"   {icon} [{event.type}] {event.message[:50]}")

        print("\n" + "=" * 60)

    def export(self, filepath: str) -> None:
        """
        Export traces to a JSON file.

        Args:
            filepath: Path to output file
        """
        data = {
            "exported_at": datetime.now().isoformat(),
            "spans": [s.to_dict() for s in self._spans],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Exported {len(self._spans)} spans to {filepath}")

    def clear(self) -> None:
        """Clear all traces."""
        with self._lock:
            self._spans.clear()
            self._current_span = None

    def get_spans(self) -> List[Span]:
        """Get all recorded spans."""
        return list(self._spans)

    def summary(self) -> Dict[str, Any]:
        """Get trace summary statistics."""
        total_spans = len(self._spans)
        completed = sum(1 for s in self._spans if s.status == "completed")
        errors = sum(1 for s in self._spans if s.status == "error")

        llm_calls = sum(1 for s in self._spans for e in s.events if e.type == "llm_call")

        total_duration = sum(s.duration_ms for s in self._spans)

        return {
            "total_spans": total_spans,
            "completed": completed,
            "errors": errors,
            "llm_calls": llm_calls,
            "total_duration_ms": total_duration,
            "avg_duration_ms": total_duration / total_spans if total_spans else 0,
        }


def traced(name: str = None):
    """
    Decorator to trace a function.

    Args:
        name: Span name (defaults to function name)

    Examples:
        >>> @trace.traced("my_function")
        ... def do_work():
        ...     return "result"
    """

    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _tracer.enabled:
                return func(*args, **kwargs)

            with _tracer.span(span_name) as span:
                try:
                    result = func(*args, **kwargs)
                    span.log("Completed successfully")
                    return result
                except Exception as e:
                    span.error(str(e), e)
                    raise

        return wrapper

    # Handle @traced vs @traced("name")
    if callable(name):
        func = name
        name = None
        return decorator(func)

    return decorator


# Global tracer instance
_tracer = Tracer()


class TraceModule:
    """Trace module with all functions attached."""

    # Core functions
    enable = _tracer.enable
    disable = _tracer.disable
    span = _tracer.span
    log = _tracer.log
    llm_call = _tracer.llm_call
    tool_call = _tracer.tool_call
    handler = _tracer.handler
    show = _tracer.show
    export = _tracer.export
    clear = _tracer.clear
    get_spans = _tracer.get_spans
    summary = _tracer.summary

    # Decorator
    traced = staticmethod(traced)

    # Check if enabled
    @property
    def enabled(self) -> bool:
        return _tracer.enabled

    # Classes
    Event = TraceEvent
    Span = Span
    Tracer = Tracer


# Module-level instance
trace = TraceModule()
