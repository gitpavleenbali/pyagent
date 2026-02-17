# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Kernel Context

Execution context for kernel operations.
Tracks state, variables, and invocation chains.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class InvocationContext:
    """Context for a single function invocation.

    Attributes:
        invocation_id: Unique ID for this invocation
        function_name: Name of the function
        plugin_name: Name of the plugin
        arguments: Function arguments
        start_time: When invocation started
        end_time: When invocation ended
        result: Function result
        error: Any error that occurred
        metadata: Additional metadata
    """

    invocation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    function_name: Optional[str] = None
    plugin_name: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Any = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[float]:
        """Get invocation duration in milliseconds."""
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() * 1000
        return None

    @property
    def success(self) -> bool:
        """Check if invocation succeeded."""
        return self.error is None

    def start(self) -> "InvocationContext":
        """Mark invocation as started."""
        self.start_time = datetime.now()
        return self

    def complete(self, result: Any = None) -> "InvocationContext":
        """Mark invocation as complete with result."""
        self.end_time = datetime.now()
        self.result = result
        return self

    def fail(self, error: Exception) -> "InvocationContext":
        """Mark invocation as failed with error."""
        self.end_time = datetime.now()
        self.error = error
        return self


@dataclass
class KernelContext:
    """Context for kernel execution.

    Maintains state across a kernel execution session including:
    - Variables (key-value pairs)
    - Invocation history
    - Parent context (for nested executions)

    Example:
        ctx = KernelContext()
        ctx.variables["user_input"] = "Hello"
        ctx.variables["city"] = "NYC"

        # Track invocation
        inv = ctx.create_invocation("weather", "get_forecast")
        inv.start()
        result = func(**inv.arguments)
        inv.complete(result)
    """

    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    variables: Dict[str, Any] = field(default_factory=dict)
    invocations: List[InvocationContext] = field(default_factory=list)
    parent: Optional["KernelContext"] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def create_invocation(
        self, plugin_name: str, function_name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> InvocationContext:
        """Create and track a new invocation.

        Args:
            plugin_name: Name of the plugin
            function_name: Name of the function
            arguments: Function arguments

        Returns:
            New invocation context
        """
        inv = InvocationContext(
            plugin_name=plugin_name, function_name=function_name, arguments=arguments or {}
        )
        self.invocations.append(inv)
        return inv

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a variable value.

        Checks this context first, then parent contexts.

        Args:
            name: Variable name
            default: Default if not found

        Returns:
            Variable value or default
        """
        if name in self.variables:
            return self.variables[name]
        if self.parent:
            return self.parent.get_variable(name, default)
        return default

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable value.

        Args:
            name: Variable name
            value: Variable value
        """
        self.variables[name] = value

    def create_child(self) -> "KernelContext":
        """Create a child context.

        Child contexts inherit variables from parent but can
        override them without affecting the parent.

        Returns:
            New child context
        """
        return KernelContext(parent=self)

    @property
    def total_invocations(self) -> int:
        """Get total number of invocations."""
        return len(self.invocations)

    @property
    def total_duration_ms(self) -> float:
        """Get total duration of all invocations."""
        return sum(inv.duration_ms or 0 for inv in self.invocations)

    @property
    def failed_invocations(self) -> List[InvocationContext]:
        """Get all failed invocations."""
        return [inv for inv in self.invocations if not inv.success]

    def clear_invocations(self) -> None:
        """Clear invocation history."""
        self.invocations.clear()

    def clear_variables(self) -> None:
        """Clear all variables."""
        self.variables.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            "context_id": self.context_id,
            "variables": self.variables,
            "invocations": [
                {
                    "id": inv.invocation_id,
                    "plugin": inv.plugin_name,
                    "function": inv.function_name,
                    "arguments": inv.arguments,
                    "duration_ms": inv.duration_ms,
                    "success": inv.success,
                    "error": str(inv.error) if inv.error else None,
                }
                for inv in self.invocations
            ],
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
