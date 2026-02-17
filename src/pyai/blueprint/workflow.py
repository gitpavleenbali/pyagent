"""
Workflow - Multi-step agent processes
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class StepType(Enum):
    """Types of workflow steps"""

    AGENT = "agent"  # Run an agent
    SKILL = "skill"  # Execute a skill
    FUNCTION = "function"  # Run a custom function
    CONDITION = "condition"  # Conditional branching
    PARALLEL = "parallel"  # Parallel execution
    LOOP = "loop"  # Loop over items


class StepStatus(Enum):
    """Status of a workflow step"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowContext:
    """
    Context passed through a workflow.

    Accumulates data from each step, allowing steps to share information.
    """

    data: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    current_step: Optional[str] = None

    def set(self, key: str, value: Any) -> None:
        """Set a context value"""
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a context value"""
        return self.data.get(key, default)

    def update(self, values: Dict[str, Any]) -> None:
        """Update multiple values"""
        self.data.update(values)

    def add_to_history(self, step: str, result: Any) -> None:
        """Add a step result to history"""
        self.history.append(
            {
                "step": step,
                "result": result,
            }
        )

    def clone(self) -> "WorkflowContext":
        """Create a copy of the context"""
        return WorkflowContext(
            data=self.data.copy(),
            history=self.history.copy(),
        )


@dataclass
class Step:
    """
    Step - A single step in a workflow.

    Steps can be:
    - Agent invocations
    - Skill executions
    - Custom functions
    - Conditional branches

    Example:
        >>> step = Step(
        ...     name="research",
        ...     step_type=StepType.AGENT,
        ...     agent=research_agent,
        ...     input_mapping={"query": "research_topic"},
        ... )
    """

    name: str
    step_type: StepType = StepType.FUNCTION

    # For agent/skill steps
    agent: Optional[Any] = None
    skill: Optional[Any] = None

    # For function steps
    handler: Optional[Callable] = None

    # For conditional steps
    condition: Optional[Callable[[WorkflowContext], bool]] = None
    if_true: Optional["Step"] = None
    if_false: Optional["Step"] = None

    # For parallel steps
    parallel_steps: List["Step"] = field(default_factory=list)

    # Data mapping
    input_mapping: Dict[str, str] = field(default_factory=dict)  # step_input -> context_key
    output_mapping: Dict[str, str] = field(default_factory=dict)  # result_key -> context_key

    # Flow control
    next_step: Optional[str] = None
    on_error: Optional[str] = None  # Step to run on error
    retry_count: int = 0

    # Status
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute this step"""
        self.status = StepStatus.RUNNING
        context.current_step = self.name

        try:
            # Get input from context
            inputs = self._get_inputs(context)

            # Execute based on step type
            if self.step_type == StepType.AGENT and self.agent:
                result = await self.agent.run(inputs.get("message", ""), **inputs)
                self.result = result.content if hasattr(result, "content") else result

            elif self.step_type == StepType.SKILL and self.skill:
                result = await self.skill.execute(**inputs)
                self.result = result.data if hasattr(result, "data") else result

            elif self.step_type == StepType.FUNCTION and self.handler:
                result = self.handler(context, **inputs)
                if asyncio.iscoroutine(result):
                    result = await result
                self.result = result

            elif self.step_type == StepType.CONDITION and self.condition:
                should_continue = self.condition(context)
                if should_continue and self.if_true:
                    self.result = await self.if_true.execute(context)
                elif not should_continue and self.if_false:
                    self.result = await self.if_false.execute(context)

            elif self.step_type == StepType.PARALLEL:
                # Execute parallel steps concurrently
                tasks = [step.execute(context.clone()) for step in self.parallel_steps]
                results = await asyncio.gather(*tasks)
                self.result = results

            # Map outputs to context
            self._set_outputs(context, self.result)

            self.status = StepStatus.COMPLETED
            context.add_to_history(self.name, self.result)

            return self.result

        except Exception as e:
            self.status = StepStatus.FAILED
            self.error = str(e)
            raise

    def _get_inputs(self, context: WorkflowContext) -> Dict[str, Any]:
        """Get inputs from context using mapping"""
        inputs = {}
        for step_input, context_key in self.input_mapping.items():
            inputs[step_input] = context.get(context_key)
        return inputs

    def _set_outputs(self, context: WorkflowContext, result: Any) -> None:
        """Set outputs to context using mapping"""
        if not self.output_mapping:
            # Default: store entire result under step name
            context.set(self.name, result)
        else:
            if isinstance(result, dict):
                for result_key, context_key in self.output_mapping.items():
                    if result_key in result:
                        context.set(context_key, result[result_key])
            else:
                # Single value, use first output mapping
                if self.output_mapping:
                    first_key = list(self.output_mapping.values())[0]
                    context.set(first_key, result)


class Workflow:
    """
    Workflow - A sequence of steps that process data through agents and skills.

    Workflows orchestrate complex multi-step processes, with support for:
    - Sequential execution
    - Conditional branching
    - Parallel execution
    - Error handling
    - Data flow between steps

    Example:
        >>> workflow = (Workflow("ResearchWorkflow")
        ...     .add_step(Step("search", step_type=StepType.SKILL, skill=search_skill))
        ...     .add_step(Step("analyze", step_type=StepType.AGENT, agent=analyst_agent))
        ...     .add_step(Step("summarize", step_type=StepType.AGENT, agent=writer_agent))
        ... )
        >>> result = await workflow.run(topic="AI trends")
    """

    def __init__(self, name: str = "Workflow"):
        self.name = name
        self.steps: Dict[str, Step] = {}
        self.step_order: List[str] = []
        self.start_step: Optional[str] = None
        self._context: Optional[WorkflowContext] = None

    def add_step(self, step: Step) -> "Workflow":
        """Add a step to the workflow"""
        self.steps[step.name] = step
        self.step_order.append(step.name)

        if self.start_step is None:
            self.start_step = step.name

        # Auto-link sequential steps
        if len(self.step_order) > 1:
            prev_step_name = self.step_order[-2]
            prev_step = self.steps[prev_step_name]
            if prev_step.next_step is None:
                prev_step.next_step = step.name

        return self

    def set_start(self, step_name: str) -> "Workflow":
        """Set the starting step"""
        if step_name in self.steps:
            self.start_step = step_name
        return self

    def link(self, from_step: str, to_step: str) -> "Workflow":
        """Link two steps together"""
        if from_step in self.steps:
            self.steps[from_step].next_step = to_step
        return self

    async def run(
        self, context: Optional[WorkflowContext] = None, **initial_data
    ) -> WorkflowContext:
        """
        Run the workflow.

        Args:
            context: Optional existing context
            **initial_data: Initial data to add to context

        Returns:
            WorkflowContext with all results
        """
        # Initialize context
        self._context = context or WorkflowContext()
        self._context.update(initial_data)

        # Reset step statuses
        for step in self.steps.values():
            step.status = StepStatus.PENDING

        # Execute steps
        current_step_name = self.start_step

        while current_step_name:
            step = self.steps.get(current_step_name)
            if not step:
                break

            try:
                await step.execute(self._context)
                current_step_name = step.next_step

            except Exception:
                if step.on_error and step.on_error in self.steps:
                    current_step_name = step.on_error
                else:
                    raise

        return self._context

    def get_step(self, name: str) -> Optional[Step]:
        """Get a step by name"""
        return self.steps.get(name)

    def get_results(self) -> Dict[str, Any]:
        """Get results from all completed steps"""
        if not self._context:
            return {}
        return self._context.data

    def __repr__(self) -> str:
        return f"Workflow({self.name}, steps={len(self.steps)})"


# Convenience functions for creating steps
def agent_step(
    name: str, agent: Any, input_mapping: Optional[Dict[str, str]] = None, **kwargs
) -> Step:
    """Create an agent step"""
    return Step(
        name=name,
        step_type=StepType.AGENT,
        agent=agent,
        input_mapping=input_mapping or {},
        **kwargs,
    )


def skill_step(
    name: str, skill: Any, input_mapping: Optional[Dict[str, str]] = None, **kwargs
) -> Step:
    """Create a skill step"""
    return Step(
        name=name,
        step_type=StepType.SKILL,
        skill=skill,
        input_mapping=input_mapping or {},
        **kwargs,
    )


def function_step(
    name: str, handler: Callable, input_mapping: Optional[Dict[str, str]] = None, **kwargs
) -> Step:
    """Create a function step"""
    return Step(
        name=name,
        step_type=StepType.FUNCTION,
        handler=handler,
        input_mapping=input_mapping or {},
        **kwargs,
    )
