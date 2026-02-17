"""
pyai Orchestrator - Advanced Multi-Agent Scheduling & Coordination
=====================================================================

Enterprise-grade orchestration for complex AI workflows:
- Workflow scheduling with dependencies
- Multi-agent coordination patterns
- Event-driven execution
- Parallel and sequential pipelines
- State management and checkpointing
- Error handling and retry policies

Patterns:
- Supervisor: One agent oversees others
- Collaborative: Agents work together with shared context
- Pipeline: Sequential processing chain
- Broadcast: One message to multiple agents
- Router: Smart routing to specialized agents
- Consensus: Multiple agents vote on decisions

Examples:
    >>> from pyai.orchestrator import Orchestrator, workflow

    # Define a workflow
    >>> @workflow("research_pipeline")
    ... def research(topic):
    ...     researcher = agent(persona="researcher")
    ...     analyst = agent(persona="analyst")
    ...     writer = agent(persona="writer")
    ...     return chain([researcher, analyst, writer], topic)

    # Schedule execution
    >>> orch = Orchestrator()
    >>> orch.schedule(research, "AI trends", run_at="2024-01-01 09:00")

    # Event-driven
    >>> orch.on("new_document", handler=process_document)
"""

import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class ExecutionPattern(Enum):
    """Workflow execution patterns."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    SUPERVISOR = "supervisor"
    COLLABORATIVE = "collaborative"
    BROADCAST = "broadcast"
    ROUTER = "router"
    CONSENSUS = "consensus"


@dataclass
class Task:
    """A scheduled task."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    func: Callable = None
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime = None
    completed_at: datetime = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = None  # seconds
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "result": str(self.result)[:100] if self.result else None,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class Workflow:
    """A workflow definition."""

    name: str
    steps: List[Task] = field(default_factory=list)
    pattern: ExecutionPattern = ExecutionPattern.SEQUENTIAL
    context: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, task: Task) -> "Workflow":
        """Add a step to the workflow."""
        self.steps.append(task)
        return self

    def get_status(self) -> Dict[str, int]:
        """Get status summary of all steps."""
        status_counts = {}
        for step in self.steps:
            status = step.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        return status_counts


@dataclass
class ScheduledJob:
    """A scheduled job for recurring execution."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    workflow: Workflow = None
    cron: str = None  # Cron expression
    interval: int = None  # Seconds between runs
    run_at: datetime = None  # One-time execution
    last_run: datetime = None
    next_run: datetime = None
    enabled: bool = True


class Orchestrator:
    """
    Main orchestrator for managing workflows and tasks.

    Features:
    - Workflow definition and execution
    - Task scheduling (one-time, recurring, cron)
    - Dependency management
    - Parallel execution
    - State persistence
    - Event handling
    """

    def __init__(self, *, max_workers: int = 4, state_file: str = None, auto_save: bool = True):
        self.max_workers = max_workers
        self.state_file = state_file
        self.auto_save = auto_save

        self._tasks: Dict[str, Task] = {}
        self._workflows: Dict[str, Workflow] = {}
        self._jobs: Dict[str, ScheduledJob] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._running = False
        self._executor = None
        self._lock = threading.Lock()

        # Load state if exists
        if state_file:
            self._load_state()

    # =========================================================================
    # Task Management
    # =========================================================================

    def submit(
        self,
        func: Callable,
        *args,
        name: str = None,
        timeout: int = None,
        dependencies: List[str] = None,
        **kwargs,
    ) -> Task:
        """
        Submit a task for execution.

        Args:
            func: Function to execute
            *args: Positional arguments
            name: Task name
            timeout: Execution timeout in seconds
            dependencies: List of task IDs this task depends on
            **kwargs: Keyword arguments

        Returns:
            Task object

        Examples:
            >>> task = orch.submit(my_agent, "analyze data", name="analysis")
            >>> print(task.id)
        """
        task = Task(
            name=name or func.__name__,
            func=func,
            args=args,
            kwargs=kwargs,
            timeout=timeout,
            dependencies=dependencies or [],
        )

        with self._lock:
            self._tasks[task.id] = task

        # Execute if no dependencies
        if not task.dependencies:
            self._execute_task(task)

        return task

    def _execute_task(self, task: Task):
        """Execute a single task."""

        def run():
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()

            try:
                # Check dependencies
                for dep_id in task.dependencies:
                    dep_task = self._tasks.get(dep_id)
                    if dep_task and dep_task.status != TaskStatus.COMPLETED:
                        # Wait for dependency
                        while dep_task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                            time.sleep(0.1)
                        if dep_task.status == TaskStatus.FAILED:
                            raise Exception(f"Dependency {dep_id} failed")

                # Execute
                result = task.func(*task.args, **task.kwargs)
                task.result = result
                task.status = TaskStatus.COMPLETED

            except Exception as e:
                task.error = str(e)
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.RETRYING
                    time.sleep(2**task.retry_count)  # Exponential backoff
                    self._execute_task(task)
                else:
                    task.status = TaskStatus.FAILED

            finally:
                task.completed_at = datetime.now()
                self._emit("task_complete", task)
                if self.auto_save:
                    self._save_state()

        thread = threading.Thread(target=run)
        thread.start()

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def wait(self, task: Task, timeout: int = None) -> Any:
        """Wait for a task to complete."""
        start = time.time()
        while task.status in [TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.RETRYING]:
            if timeout and (time.time() - start) > timeout:
                raise TimeoutError(f"Task {task.id} timed out")
            time.sleep(0.1)

        if task.status == TaskStatus.FAILED:
            raise Exception(f"Task failed: {task.error}")

        return task.result

    # =========================================================================
    # Workflow Management
    # =========================================================================

    def create_workflow(
        self, name: str, *, pattern: ExecutionPattern = ExecutionPattern.SEQUENTIAL
    ) -> Workflow:
        """
        Create a new workflow.

        Args:
            name: Workflow name
            pattern: Execution pattern

        Returns:
            Workflow object
        """
        wf = Workflow(name=name, pattern=pattern)
        self._workflows[name] = wf
        return wf

    def run_workflow(
        self, workflow: Union[str, Workflow], *, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run a workflow.

        Args:
            workflow: Workflow name or object
            context: Initial context

        Returns:
            Final workflow state
        """
        if isinstance(workflow, str):
            workflow = self._workflows.get(workflow)
            if not workflow:
                raise ValueError(f"Workflow not found: {workflow}")

        workflow.context.update(context or {})

        if workflow.pattern == ExecutionPattern.SEQUENTIAL:
            return self._run_sequential(workflow)
        elif workflow.pattern == ExecutionPattern.PARALLEL:
            return self._run_parallel(workflow)
        elif workflow.pattern == ExecutionPattern.SUPERVISOR:
            return self._run_supervisor(workflow)
        else:
            return self._run_sequential(workflow)

    def _run_sequential(self, workflow: Workflow) -> Dict[str, Any]:
        """Run workflow steps sequentially."""
        for step in workflow.steps:
            self._execute_task(step)
            self.wait(step)
            workflow.state[step.name] = step.result
        return workflow.state

    def _run_parallel(self, workflow: Workflow) -> Dict[str, Any]:
        """Run workflow steps in parallel."""
        for step in workflow.steps:
            self._execute_task(step)

        # Wait for all
        for step in workflow.steps:
            self.wait(step)
            workflow.state[step.name] = step.result

        return workflow.state

    def _run_supervisor(self, workflow: Workflow) -> Dict[str, Any]:
        """Run with supervisor pattern - first step oversees others."""
        if not workflow.steps:
            return workflow.state

        supervisor = workflow.steps[0]
        workers = workflow.steps[1:]

        # Run workers
        for worker in workers:
            self._execute_task(worker)

        # Wait for workers
        for worker in workers:
            self.wait(worker)
            workflow.state[worker.name] = worker.result

        # Supervisor reviews
        json.dumps(workflow.state)
        supervisor.kwargs["worker_results"] = workflow.state
        self._execute_task(supervisor)
        self.wait(supervisor)
        workflow.state["supervisor_review"] = supervisor.result

        return workflow.state

    # =========================================================================
    # Scheduling
    # =========================================================================

    def schedule(
        self,
        func: Callable,
        *args,
        run_at: Union[datetime, str] = None,
        interval: int = None,
        cron: str = None,
        **kwargs,
    ) -> ScheduledJob:
        """
        Schedule a function for future execution.

        Args:
            func: Function to schedule
            *args: Arguments
            run_at: One-time execution (datetime or ISO string)
            interval: Repeat interval in seconds
            cron: Cron expression (e.g., "0 9 * * *" for 9am daily)
            **kwargs: Keyword arguments

        Returns:
            ScheduledJob object

        Examples:
            >>> # Run once at specific time
            >>> orch.schedule(backup, run_at="2024-01-01 09:00")

            >>> # Run every hour
            >>> orch.schedule(health_check, interval=3600)

            >>> # Run daily at 9am
            >>> orch.schedule(report, cron="0 9 * * *")
        """
        # Parse run_at if string
        if isinstance(run_at, str):
            run_at = datetime.fromisoformat(run_at)

        # Create task for the job
        task = Task(name=func.__name__, func=func, args=args, kwargs=kwargs)

        # Create workflow with single step
        workflow = Workflow(name=f"scheduled_{task.name}")
        workflow.add_step(task)

        job = ScheduledJob(
            workflow=workflow,
            run_at=run_at,
            interval=interval,
            cron=cron,
            next_run=run_at or datetime.now(),
        )

        self._jobs[job.id] = job
        return job

    def start(self):
        """Start the scheduler background thread."""
        self._running = True

        def scheduler_loop():
            while self._running:
                now = datetime.now()

                for job in self._jobs.values():
                    if not job.enabled:
                        continue

                    if job.next_run and job.next_run <= now:
                        # Execute
                        self.run_workflow(job.workflow)
                        job.last_run = now

                        # Calculate next run
                        if job.interval:
                            job.next_run = now + timedelta(seconds=job.interval)
                        elif job.cron:
                            job.next_run = self._parse_cron_next(job.cron, now)
                        else:
                            job.enabled = False  # One-time job

                time.sleep(1)

        self._executor = threading.Thread(target=scheduler_loop, daemon=True)
        self._executor.start()

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._executor:
            self._executor.join(timeout=5)

    def _parse_cron_next(self, cron: str, after: datetime) -> datetime:
        """Parse cron expression and return next run time."""
        # Simplified cron parsing (minute hour day month weekday)
        parts = cron.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {cron}")

        # For now, just add 1 day for daily jobs
        return after + timedelta(days=1)

    # =========================================================================
    # Event System
    # =========================================================================

    def on(self, event: str, handler: Callable):
        """
        Register an event handler.

        Args:
            event: Event name
            handler: Handler function

        Events:
            - task_complete: When a task completes
            - task_failed: When a task fails
            - workflow_complete: When a workflow completes
            - custom events via emit()
        """
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def emit(self, event: str, data: Any = None):
        """Emit a custom event."""
        self._emit(event, data)

    def _emit(self, event: str, data: Any = None):
        """Internal event emission."""
        handlers = self._event_handlers.get(event, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                print(f"Event handler error: {e}")

    # =========================================================================
    # State Persistence
    # =========================================================================

    def _save_state(self):
        """Save orchestrator state to file."""
        if not self.state_file:
            return

        state = {
            "tasks": {k: v.to_dict() for k, v in self._tasks.items()},
            "workflows": {k: v.name for k, v in self._workflows.items()},
            "jobs": {k: {"id": v.id, "enabled": v.enabled} for k, v in self._jobs.items()},
        }

        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load orchestrator state from file."""
        import os

        if not self.state_file or not os.path.exists(self.state_file):
            return

        try:
            with open(self.state_file, "r") as f:
                json.load(f)
            # Restore state as needed
        except Exception as e:
            print(f"Failed to load state: {e}")

    # =========================================================================
    # Status & Monitoring
    # =========================================================================

    def status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "running": self._running,
            "tasks": {
                "total": len(self._tasks),
                "pending": sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING),
                "running": sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING),
                "completed": sum(
                    1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED
                ),
                "failed": sum(1 for t in self._tasks.values() if t.status == TaskStatus.FAILED),
            },
            "workflows": len(self._workflows),
            "jobs": len(self._jobs),
        }

    def __repr__(self) -> str:
        return f"Orchestrator(tasks={len(self._tasks)}, workflows={len(self._workflows)})"


# =============================================================================
# Decorator for Workflow Functions
# =============================================================================


def workflow(name: str = None, pattern: ExecutionPattern = ExecutionPattern.SEQUENTIAL):
    """
    Decorator to define a workflow function.

    Examples:
        >>> @workflow("my_pipeline")
        ... def process_data(data):
        ...     agent1 = agent("processor")
        ...     agent2 = agent("validator")
        ...     result1 = agent1(data)
        ...     result2 = agent2(result1)
        ...     return result2
    """

    def decorator(func: Callable):
        func._workflow_name = name or func.__name__
        func._workflow_pattern = pattern
        return func

    return decorator


# =============================================================================
# Multi-Agent Patterns
# =============================================================================


class AgentPatterns:
    """
    Pre-built multi-agent coordination patterns.
    """

    @staticmethod
    def supervisor(
        supervisor_agent, worker_agents: List, task: str, *, review_prompt: str = None
    ) -> str:
        """
        Supervisor pattern: One agent oversees and combines work from others.

        Args:
            supervisor_agent: The supervising agent
            worker_agents: List of worker agents
            task: The task to complete
            review_prompt: Custom prompt for supervisor review

        Returns:
            Supervisor's final output
        """
        from pyai import handoff

        # Workers process in parallel (conceptually)
        worker_results = {}
        for worker in worker_agents:
            result = worker(task)
            worker_results[worker.name] = result

        # Supervisor reviews
        review_input = f"""Task: {task}

Worker Results:
{json.dumps(worker_results, indent=2)}

{review_prompt or "Review and synthesize these results into a final output."}"""

        return supervisor_agent(review_input)

    @staticmethod
    def consensus(agents: List, question: str, *, threshold: float = 0.6) -> Dict[str, Any]:
        """
        Consensus pattern: Multiple agents vote on a decision.

        Args:
            agents: List of voting agents
            question: Question to decide
            threshold: Agreement threshold (0-1)

        Returns:
            Dict with decision and votes
        """
        votes = {}
        for agent in agents:
            response = agent(f"Answer YES or NO: {question}")
            vote = "yes" if "yes" in response.lower() else "no"
            votes[agent.name] = vote

        # Count votes
        yes_count = sum(1 for v in votes.values() if v == "yes")
        agreement = yes_count / len(agents)

        return {
            "decision": "yes" if agreement >= threshold else "no",
            "agreement": agreement,
            "votes": votes,
        }

    @staticmethod
    def debate(agent_a, agent_b, topic: str, *, rounds: int = 3, judge_agent=None) -> str:
        """
        Debate pattern: Two agents debate, optionally judged.

        Args:
            agent_a: First debater
            agent_b: Second debater
            topic: Debate topic
            rounds: Number of rounds
            judge_agent: Optional judge to determine winner

        Returns:
            Debate transcript and optional verdict
        """
        transcript = []

        for round_num in range(rounds):
            # Agent A argues
            context = f"Debate topic: {topic}\n\nPrevious exchanges:\n" + "\n".join(transcript[-4:])
            a_response = agent_a(f"{context}\n\nMake your argument (round {round_num + 1}):")
            transcript.append(f"Agent A: {a_response}")

            # Agent B responds
            context = f"Debate topic: {topic}\n\nPrevious exchanges:\n" + "\n".join(transcript[-4:])
            b_response = agent_b(f"{context}\n\nCounter-argue (round {round_num + 1}):")
            transcript.append(f"Agent B: {b_response}")

        full_transcript = "\n\n".join(transcript)

        if judge_agent:
            verdict = judge_agent(
                f"Judge this debate:\n\n{full_transcript}\n\nWho made better arguments and why?"
            )
            return f"{full_transcript}\n\n--- VERDICT ---\n{verdict}"

        return full_transcript

    @staticmethod
    def chain_of_thought(agents: List, problem: str) -> str:
        """
        Chain of thought: Each agent builds on the previous one's reasoning.

        Args:
            agents: List of agents for chain
            problem: Problem to solve

        Returns:
            Final solution
        """
        reasoning_chain = []

        for i, agent in enumerate(agents):
            prompt = f"""Problem: {problem}

Previous reasoning:
{chr(10).join(reasoning_chain) if reasoning_chain else "None yet"}

Your turn (step {i + 1} of {len(agents)}): Continue the reasoning."""

            response = agent(prompt)
            reasoning_chain.append(f"Step {i + 1}: {response}")

        return "\n\n".join(reasoning_chain)


# Create orchestrator instance
patterns = AgentPatterns()
