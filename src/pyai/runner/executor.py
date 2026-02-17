# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Runner Executor

The Runner manages agent execution with structured control flow.
Similar to OpenAI Agents SDK's Runner pattern.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class RunStatus(Enum):
    """Status of a run."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RunConfig:
    """Configuration for agent execution.

    Example:
        config = RunConfig(
            max_turns=10,
            max_time=60.0,
            stop_on_tool_error=True
        )
    """

    max_turns: int = 10
    max_time: float = 300.0  # 5 minutes default
    timeout_per_turn: float = 60.0
    stop_on_tool_error: bool = False
    verbose: bool = False
    trace_enabled: bool = True
    run_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunContext:
    """Runtime context passed to agents during execution.

    Provides access to run state and utilities.
    """

    run_id: str
    turn_count: int = 0
    start_time: float = field(default_factory=time.time)
    variables: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time

    def set_variable(self, key: str, value: Any) -> None:
        """Set a context variable."""
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self.variables.get(key, default)


@dataclass
class RunResult:
    """Result of an agent run.

    Contains the final output and execution metadata.
    """

    run_id: str
    status: RunStatus
    output: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0
    elapsed_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if run was successful."""
        return self.status == RunStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "output": self.output,
            "messages": self.messages,
            "tool_calls": self.tool_calls,
            "turn_count": self.turn_count,
            "elapsed_time": self.elapsed_time,
            "error": self.error,
            "metadata": self.metadata,
        }


class Runner:
    """Structured runner for AI agent execution.

    Provides controlled execution with:
    - Turn limits
    - Time limits
    - Error handling
    - Execution tracing

    Example:
        # Simple usage
        result = Runner.run(agent, "What is the weather?")
        print(result.output)

        # With configuration
        runner = Runner(config=RunConfig(max_turns=5, verbose=True))
        result = runner.execute(agent, "Complex task")

        # Async execution
        result = await Runner.run_async(agent, "Async task")
    """

    def __init__(self, config: Optional[RunConfig] = None):
        """Initialize runner with configuration.

        Args:
            config: Run configuration
        """
        self.config = config or RunConfig()

    def execute(
        self, agent: Any, input: str, context: Optional[RunContext] = None, **kwargs
    ) -> RunResult:
        """Execute an agent with structured control.

        Args:
            agent: The agent to run (must have run/chat method)
            input: User input to process
            context: Optional execution context
            **kwargs: Additional arguments passed to agent

        Returns:
            RunResult with execution details
        """
        run_id = self.config.run_id or f"run-{uuid.uuid4().hex[:12]}"
        context = context or RunContext(run_id=run_id)
        start_time = time.time()

        try:
            # Execute agent
            messages = [{"role": "user", "content": input}]
            tool_calls = []
            output = None

            for turn in range(self.config.max_turns):
                context.turn_count = turn + 1

                # Check time limit
                if context.elapsed_time() > self.config.max_time:
                    return RunResult(
                        run_id=run_id,
                        status=RunStatus.FAILED,
                        error=f"Time limit exceeded ({self.config.max_time}s)",
                        messages=messages,
                        turn_count=context.turn_count,
                        elapsed_time=time.time() - start_time,
                    )

                # Run agent turn
                response = self._run_agent_turn(agent, messages, context, **kwargs)

                if response is None:
                    break

                # Handle response
                if isinstance(response, str):
                    output = response
                    messages.append({"role": "assistant", "content": output})
                    break
                elif isinstance(response, dict):
                    if "tool_calls" in response:
                        tool_calls.extend(response["tool_calls"])
                        messages.append(response)
                        # Continue if tool calls need processing
                        continue
                    elif "content" in response:
                        output = response["content"]
                        messages.append(response)
                        break
                else:
                    output = str(response)
                    messages.append({"role": "assistant", "content": output})
                    break

            return RunResult(
                run_id=run_id,
                status=RunStatus.COMPLETED,
                output=output,
                messages=messages,
                tool_calls=tool_calls,
                turn_count=context.turn_count,
                elapsed_time=time.time() - start_time,
                metadata=kwargs.get("metadata", {}),
            )

        except Exception as e:
            return RunResult(
                run_id=run_id,
                status=RunStatus.FAILED,
                error=str(e),
                messages=[{"role": "user", "content": input}],
                turn_count=context.turn_count if context else 0,
                elapsed_time=time.time() - start_time,
            )

    def _run_agent_turn(
        self, agent: Any, messages: List[Dict], context: RunContext, **kwargs
    ) -> Optional[Union[str, Dict]]:
        """Execute a single agent turn.

        Args:
            agent: The agent
            messages: Conversation history
            context: Execution context
            **kwargs: Additional arguments

        Returns:
            Agent response or None
        """
        # Try different agent interfaces
        if hasattr(agent, "run"):
            return agent.run(messages[-1]["content"], **kwargs)
        elif hasattr(agent, "chat"):
            return agent.chat(messages, **kwargs)
        elif hasattr(agent, "__call__"):
            return agent(messages[-1]["content"], **kwargs)
        elif callable(agent):
            return agent(messages[-1]["content"], **kwargs)
        else:
            raise TypeError(f"Agent must have run/chat method or be callable: {type(agent)}")

    async def execute_async(
        self, agent: Any, input: str, context: Optional[RunContext] = None, **kwargs
    ) -> RunResult:
        """Execute an agent asynchronously.

        Args:
            agent: The agent to run
            input: User input
            context: Optional execution context
            **kwargs: Additional arguments

        Returns:
            RunResult with execution details
        """
        run_id = self.config.run_id or f"run-{uuid.uuid4().hex[:12]}"
        context = context or RunContext(run_id=run_id)
        start_time = time.time()

        try:
            messages = [{"role": "user", "content": input}]
            tool_calls = []
            output = None

            for turn in range(self.config.max_turns):
                context.turn_count = turn + 1

                if context.elapsed_time() > self.config.max_time:
                    return RunResult(
                        run_id=run_id,
                        status=RunStatus.FAILED,
                        error=f"Time limit exceeded ({self.config.max_time}s)",
                        messages=messages,
                        turn_count=context.turn_count,
                        elapsed_time=time.time() - start_time,
                    )

                # Run agent turn asynchronously
                response = await self._run_agent_turn_async(agent, messages, context, **kwargs)

                if response is None:
                    break

                if isinstance(response, str):
                    output = response
                    messages.append({"role": "assistant", "content": output})
                    break
                elif isinstance(response, dict):
                    if "tool_calls" in response:
                        tool_calls.extend(response["tool_calls"])
                        messages.append(response)
                        continue
                    elif "content" in response:
                        output = response["content"]
                        messages.append(response)
                        break
                else:
                    output = str(response)
                    messages.append({"role": "assistant", "content": output})
                    break

            return RunResult(
                run_id=run_id,
                status=RunStatus.COMPLETED,
                output=output,
                messages=messages,
                tool_calls=tool_calls,
                turn_count=context.turn_count,
                elapsed_time=time.time() - start_time,
            )

        except Exception as e:
            return RunResult(
                run_id=run_id,
                status=RunStatus.FAILED,
                error=str(e),
                messages=[{"role": "user", "content": input}],
                turn_count=context.turn_count if context else 0,
                elapsed_time=time.time() - start_time,
            )

    async def _run_agent_turn_async(
        self, agent: Any, messages: List[Dict], context: RunContext, **kwargs
    ) -> Optional[Union[str, Dict]]:
        """Execute a single agent turn asynchronously."""
        if hasattr(agent, "run_async"):
            return await agent.run_async(messages[-1]["content"], **kwargs)
        elif hasattr(agent, "arun"):
            return await agent.arun(messages[-1]["content"], **kwargs)
        elif hasattr(agent, "chat_async"):
            return await agent.chat_async(messages, **kwargs)
        elif asyncio.iscoroutinefunction(agent):
            return await agent(messages[-1]["content"], **kwargs)
        else:
            # Fall back to sync execution in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: self._run_agent_turn(agent, messages, context, **kwargs)
            )

    @classmethod
    def run(cls, agent: Any, input: str, config: Optional[RunConfig] = None, **kwargs) -> RunResult:
        """Convenience class method to run an agent.

        Example:
            result = Runner.run(agent, "What is 2+2?")
            print(result.output)
        """
        runner = cls(config=config)
        return runner.execute(agent, input, **kwargs)

    @classmethod
    async def run_async(
        cls, agent: Any, input: str, config: Optional[RunConfig] = None, **kwargs
    ) -> RunResult:
        """Convenience class method for async execution.

        Example:
            result = await Runner.run_async(agent, "What is 2+2?")
            print(result.output)
        """
        runner = cls(config=config)
        return await runner.execute_async(agent, input, **kwargs)
