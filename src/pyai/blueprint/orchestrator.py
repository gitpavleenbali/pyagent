"""
Orchestrator - Multi-agent coordination
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class AgentRole(Enum):
    """Roles an agent can have in an orchestration"""

    WORKER = "worker"  # Executes tasks
    SUPERVISOR = "supervisor"  # Oversees workers
    ROUTER = "router"  # Routes requests
    SPECIALIST = "specialist"  # Domain expert


@dataclass
class AgentEntry:
    """Entry for an agent in the pool"""

    agent: Any
    role: AgentRole = AgentRole.WORKER
    name: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    max_concurrent: int = 1
    current_tasks: int = 0

    @property
    def available(self) -> bool:
        return self.current_tasks < self.max_concurrent


class AgentPool:
    """
    AgentPool - A pool of agents for orchestration.

    Manages a collection of agents with different roles and capabilities,
    enabling dynamic agent selection and load balancing.

    Example:
        >>> pool = AgentPool()
        >>> pool.add(research_agent, role=AgentRole.WORKER, capabilities=["research"])
        >>> pool.add(code_agent, role=AgentRole.WORKER, capabilities=["coding"])
        >>> pool.add(supervisor_agent, role=AgentRole.SUPERVISOR)
        >>>
        >>> agent = pool.get_by_capability("research")
    """

    def __init__(self):
        self._agents: Dict[str, AgentEntry] = {}

    def add(
        self,
        agent: Any,
        name: Optional[str] = None,
        role: AgentRole = AgentRole.WORKER,
        capabilities: Optional[List[str]] = None,
        max_concurrent: int = 1,
    ) -> None:
        """Add an agent to the pool"""
        agent_name = name or getattr(agent, "name", f"agent_{len(self._agents)}")

        self._agents[agent_name] = AgentEntry(
            agent=agent,
            role=role,
            name=agent_name,
            capabilities=capabilities or [],
            max_concurrent=max_concurrent,
        )

    def get(self, name: str) -> Optional[Any]:
        """Get an agent by name"""
        entry = self._agents.get(name)
        return entry.agent if entry else None

    def get_by_role(self, role: AgentRole) -> List[Any]:
        """Get all agents with a specific role"""
        return [entry.agent for entry in self._agents.values() if entry.role == role]

    def get_by_capability(self, capability: str) -> Optional[Any]:
        """Get an available agent with a specific capability"""
        for entry in self._agents.values():
            if capability in entry.capabilities and entry.available:
                return entry.agent
        return None

    def get_available(self, role: Optional[AgentRole] = None) -> List[Any]:
        """Get all available agents, optionally filtered by role"""
        agents = []
        for entry in self._agents.values():
            if not entry.available:
                continue
            if role and entry.role != role:
                continue
            agents.append(entry.agent)
        return agents

    def acquire(self, name: str) -> bool:
        """Acquire an agent for a task"""
        if name in self._agents:
            entry = self._agents[name]
            if entry.available:
                entry.current_tasks += 1
                return True
        return False

    def release(self, name: str) -> None:
        """Release an agent after task completion"""
        if name in self._agents:
            entry = self._agents[name]
            entry.current_tasks = max(0, entry.current_tasks - 1)

    def list(self) -> List[str]:
        """List all agent names"""
        return list(self._agents.keys())

    def __len__(self) -> int:
        return len(self._agents)


class Orchestrator(ABC):
    """
    Orchestrator - Base class for multi-agent coordination.

    Orchestrators manage how multiple agents work together to
    accomplish complex tasks. Different orchestration patterns
    can be implemented by subclassing.

    Example usage:
        >>> orchestrator = SupervisorOrchestrator(
        ...     supervisor=supervisor_agent,
        ...     workers=[worker1, worker2, worker3],
        ... )
        >>> result = await orchestrator.run("Complete this complex task")
    """

    def __init__(self, name: str = "Orchestrator"):
        self.name = name
        self.pool = AgentPool()

    @abstractmethod
    async def run(self, task: str, **kwargs) -> Any:
        """
        Run the orchestration on a task.

        Args:
            task: The task to accomplish
            **kwargs: Additional arguments

        Returns:
            Result of the orchestration
        """
        pass

    def add_agent(self, agent: Any, **kwargs) -> None:
        """Add an agent to the orchestrator"""
        self.pool.add(agent, **kwargs)


class SequentialOrchestrator(Orchestrator):
    """
    Orchestrator that runs agents in sequence.

    Each agent's output becomes the next agent's input.
    """

    def __init__(self, agents: List[Any], name: str = "Sequential"):
        super().__init__(name)
        self.sequence = agents
        for i, agent in enumerate(agents):
            self.pool.add(agent, name=f"agent_{i}")

    async def run(self, task: str, **kwargs) -> Any:
        """Run agents sequentially"""
        current_input = task

        for agent in self.sequence:
            result = await agent.run(current_input, **kwargs)
            current_input = result.content if hasattr(result, "content") else str(result)

        return current_input


class ParallelOrchestrator(Orchestrator):
    """
    Orchestrator that runs agents in parallel.

    All agents receive the same input and run concurrently.
    Results are aggregated.
    """

    def __init__(
        self,
        agents: List[Any],
        aggregator: Optional[Callable[[List[Any]], Any]] = None,
        name: str = "Parallel",
    ):
        super().__init__(name)
        self.agents = agents
        self.aggregator = aggregator or (lambda results: results)
        for i, agent in enumerate(agents):
            self.pool.add(agent, name=f"agent_{i}")

    async def run(self, task: str, **kwargs) -> Any:
        """Run agents in parallel"""
        tasks = [agent.run(task, **kwargs) for agent in self.agents]
        results = await asyncio.gather(*tasks)

        # Extract content from results
        outputs = [r.content if hasattr(r, "content") else r for r in results]

        return self.aggregator(outputs)


class RouterOrchestrator(Orchestrator):
    """
    Orchestrator that routes tasks to appropriate agents.

    Uses a routing function to determine which agent should
    handle each task.
    """

    def __init__(
        self,
        router: Callable[[str], str],  # task -> agent_name
        agents: Dict[str, Any] = None,
        name: str = "Router",
    ):
        super().__init__(name)
        self.router = router

        if agents:
            for agent_name, agent in agents.items():
                self.pool.add(agent, name=agent_name)

    def add_route(self, name: str, agent: Any, **kwargs) -> None:
        """Add an agent with a route name"""
        self.pool.add(agent, name=name, **kwargs)

    async def run(self, task: str, **kwargs) -> Any:
        """Route task to appropriate agent"""
        agent_name = self.router(task)
        agent = self.pool.get(agent_name)

        if not agent:
            raise ValueError(f"No agent found for route: {agent_name}")

        return await agent.run(task, **kwargs)


class SupervisorOrchestrator(Orchestrator):
    """
    Orchestrator with a supervisor that delegates to workers.

    The supervisor agent decides which worker agents to use
    and how to combine their results.
    """

    def __init__(self, supervisor: Any, workers: List[Any], name: str = "Supervisor"):
        super().__init__(name)
        self.supervisor = supervisor
        self.workers = workers

        self.pool.add(supervisor, name="supervisor", role=AgentRole.SUPERVISOR)
        for i, worker in enumerate(workers):
            self.pool.add(worker, name=f"worker_{i}", role=AgentRole.WORKER)

    async def run(self, task: str, max_iterations: int = 5, **kwargs) -> Any:
        """Run supervised orchestration"""
        context = {
            "original_task": task,
            "worker_results": [],
            "iterations": 0,
        }

        for _ in range(max_iterations):
            context["iterations"] += 1

            # Supervisor decides next action
            supervisor_prompt = self._build_supervisor_prompt(context)
            decision = await self.supervisor.run(supervisor_prompt)

            # Parse supervisor decision
            action = self._parse_supervisor_decision(decision)

            if action.get("complete"):
                return action.get("final_answer", decision.content)

            # Execute worker task if specified
            worker_idx = action.get("worker", 0)
            if 0 <= worker_idx < len(self.workers):
                worker = self.workers[worker_idx]
                worker_task = action.get("task", task)
                result = await worker.run(worker_task)
                context["worker_results"].append(
                    {
                        "worker": worker_idx,
                        "task": worker_task,
                        "result": result.content if hasattr(result, "content") else result,
                    }
                )

        return context["worker_results"][-1]["result"] if context["worker_results"] else None

    def _build_supervisor_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for supervisor"""
        prompt = f"Task: {context['original_task']}\n"
        prompt += f"Workers available: {len(self.workers)}\n"

        if context["worker_results"]:
            prompt += "\nPrevious results:\n"
            for wr in context["worker_results"]:
                prompt += f"- Worker {wr['worker']}: {wr['result'][:200]}...\n"

        prompt += "\nDecide: assign to a worker or provide final answer."
        return prompt

    def _parse_supervisor_decision(self, decision: Any) -> Dict[str, Any]:
        """Parse supervisor's decision"""
        content = decision.content if hasattr(decision, "content") else str(decision)

        # Simple parsing - could be made more sophisticated
        if "final" in content.lower() or "complete" in content.lower():
            return {"complete": True, "final_answer": content}

        return {"worker": 0, "task": content}
