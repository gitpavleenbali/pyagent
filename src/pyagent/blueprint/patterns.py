"""
Patterns - Common multi-agent orchestration patterns
"""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
import asyncio

from pyagent.blueprint.orchestrator import Orchestrator, AgentPool, AgentRole


class RouterPattern(Orchestrator):
    """
    Router Pattern - Route requests to specialized agents.
    
    Uses a classifier/router to determine which specialized agent
    should handle each request. Good for:
    - Multi-domain assistants
    - Skill-based routing
    - Load balancing
    
    Example:
        >>> router = RouterPattern()
        >>> router.add_route("code", code_agent, keywords=["code", "programming"])
        >>> router.add_route("math", math_agent, keywords=["calculate", "equation"])
        >>> result = await router.run("Help me write a Python function")
    """
    
    def __init__(
        self,
        classifier: Optional[Callable[[str], str]] = None,
        name: str = "Router"
    ):
        super().__init__(name)
        self._routes: Dict[str, Dict[str, Any]] = {}
        self._classifier = classifier
    
    def add_route(
        self,
        name: str,
        agent: Any,
        keywords: Optional[List[str]] = None,
        condition: Optional[Callable[[str], bool]] = None,
    ) -> "RouterPattern":
        """Add a route with optional matching criteria"""
        self._routes[name] = {
            "agent": agent,
            "keywords": keywords or [],
            "condition": condition,
        }
        self.pool.add(agent, name=name, role=AgentRole.SPECIALIST)
        return self
    
    def _default_classifier(self, task: str) -> str:
        """Default keyword-based classification"""
        task_lower = task.lower()
        
        for route_name, route_config in self._routes.items():
            # Check condition first
            if route_config.get("condition"):
                if route_config["condition"](task):
                    return route_name
            
            # Check keywords
            for keyword in route_config.get("keywords", []):
                if keyword.lower() in task_lower:
                    return route_name
        
        # Return first route as default
        return list(self._routes.keys())[0] if self._routes else None
    
    async def run(self, task: str, **kwargs) -> Any:
        """Route and execute task"""
        classifier = self._classifier or self._default_classifier
        route_name = classifier(task)
        
        if not route_name or route_name not in self._routes:
            raise ValueError(f"No route found for task")
        
        agent = self._routes[route_name]["agent"]
        return await agent.run(task, **kwargs)


class ChainPattern(Orchestrator):
    """
    Chain Pattern - Sequential agent processing.
    
    Each agent in the chain processes the output of the previous one.
    Good for:
    - Multi-step processing
    - Refinement workflows
    - Quality pipelines
    
    Example:
        >>> chain = (ChainPattern()
        ...     .add("draft", writer_agent)
        ...     .add("review", reviewer_agent)
        ...     .add("edit", editor_agent)
        ... )
        >>> final = await chain.run("Write about quantum computing")
    """
    
    def __init__(self, name: str = "Chain"):
        super().__init__(name)
        self._chain: List[Dict[str, Any]] = []
    
    def add(
        self,
        name: str,
        agent: Any,
        transform: Optional[Callable[[str, str], str]] = None,
    ) -> "ChainPattern":
        """
        Add an agent to the chain.
        
        Args:
            name: Step name
            agent: Agent to add
            transform: Optional function to transform input (prev_output, original) -> new_input
        """
        self._chain.append({
            "name": name,
            "agent": agent,
            "transform": transform,
        })
        self.pool.add(agent, name=name)
        return self
    
    async def run(self, task: str, **kwargs) -> Any:
        """Execute the chain"""
        original_task = task
        current_input = task
        results = []
        
        for step in self._chain:
            agent = step["agent"]
            transform = step["transform"]
            
            # Apply transform if provided
            if transform:
                current_input = transform(current_input, original_task)
            
            # Execute agent
            result = await agent.run(current_input, **kwargs)
            output = result.content if hasattr(result, 'content') else str(result)
            
            results.append({
                "step": step["name"],
                "output": output,
            })
            
            current_input = output
        
        return current_input


class MapReducePattern(Orchestrator):
    """
    Map-Reduce Pattern - Parallel processing with aggregation.
    
    Splits work across multiple agents (map), then combines results (reduce).
    Good for:
    - Large-scale analysis
    - Parallel research
    - Consensus building
    
    Example:
        >>> mr = MapReducePattern(
        ...     mapper_agent=analyst,
        ...     reducer_agent=synthesizer,
        ... )
        >>> result = await mr.run("Analyze these 5 companies", items=companies)
    """
    
    def __init__(
        self,
        mapper_agent: Any,
        reducer_agent: Any,
        num_workers: int = 3,
        name: str = "MapReduce"
    ):
        super().__init__(name)
        self.mapper = mapper_agent
        self.reducer = reducer_agent
        self.num_workers = num_workers
        
        self.pool.add(mapper_agent, name="mapper", role=AgentRole.WORKER)
        self.pool.add(reducer_agent, name="reducer", role=AgentRole.WORKER)
    
    async def run(
        self,
        task: str,
        items: Optional[List[Any]] = None,
        **kwargs
    ) -> Any:
        """Execute map-reduce"""
        items = items or [task]
        
        # Map phase - process items in parallel
        map_tasks = []
        for item in items:
            map_input = f"{task}\n\nItem to analyze: {item}"
            map_tasks.append(self.mapper.run(map_input, **kwargs))
        
        map_results = await asyncio.gather(*map_tasks)
        
        # Extract content from results
        mapped_outputs = [
            r.content if hasattr(r, 'content') else str(r)
            for r in map_results
        ]
        
        # Reduce phase - combine results
        reduce_input = f"{task}\n\nResults to synthesize:\n"
        for i, output in enumerate(mapped_outputs):
            reduce_input += f"\n--- Result {i+1} ---\n{output}\n"
        
        reduce_result = await self.reducer.run(reduce_input, **kwargs)
        
        return reduce_result.content if hasattr(reduce_result, 'content') else reduce_result


class SupervisorPattern(Orchestrator):
    """
    Supervisor Pattern - Hierarchical agent management.
    
    A supervisor agent delegates tasks to workers and synthesizes results.
    Good for:
    - Complex task decomposition
    - Quality control
    - Adaptive workflows
    
    Example:
        >>> supervisor = SupervisorPattern(
        ...     supervisor=project_manager,
        ...     workers={
        ...         "research": researcher,
        ...         "code": developer,
        ...         "review": reviewer,
        ...     }
        ... )
        >>> result = await supervisor.run("Build a web scraper")
    """
    
    def __init__(
        self,
        supervisor: Any,
        workers: Dict[str, Any],
        name: str = "Supervisor"
    ):
        super().__init__(name)
        self._supervisor = supervisor
        self._workers = workers
        
        self.pool.add(supervisor, name="supervisor", role=AgentRole.SUPERVISOR)
        for worker_name, worker in workers.items():
            self.pool.add(worker, name=worker_name, role=AgentRole.WORKER)
    
    async def run(self, task: str, max_iterations: int = 10, **kwargs) -> Any:
        """Execute supervised workflow"""
        memory = {
            "task": task,
            "steps": [],
            "iterations": 0,
        }
        
        while memory["iterations"] < max_iterations:
            memory["iterations"] += 1
            
            # Ask supervisor for next action
            decision = await self._get_supervisor_decision(memory)
            
            if decision.get("complete"):
                return decision.get("answer")
            
            # Execute worker task
            worker_name = decision.get("worker")
            worker_task = decision.get("task", task)
            
            if worker_name in self._workers:
                worker = self._workers[worker_name]
                result = await worker.run(worker_task, **kwargs)
                
                memory["steps"].append({
                    "worker": worker_name,
                    "task": worker_task,
                    "result": result.content if hasattr(result, 'content') else str(result),
                })
        
        # Return last result
        if memory["steps"]:
            return memory["steps"][-1]["result"]
        return None
    
    async def _get_supervisor_decision(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Get supervisor's decision on next action"""
        prompt = f"""Task: {memory['task']}

Available workers: {list(self._workers.keys())}

"""
        if memory["steps"]:
            prompt += "Work completed so far:\n"
            for step in memory["steps"]:
                prompt += f"- {step['worker']}: {step['result'][:200]}...\n"
        
        prompt += """
Decide your next action:
1. Assign to a worker: respond with WORKER: <name> TASK: <specific task>
2. Complete: respond with COMPLETE: <final answer>
"""
        
        result = await self._supervisor.run(prompt)
        content = result.content if hasattr(result, 'content') else str(result)
        
        # Parse decision
        if "COMPLETE:" in content:
            answer = content.split("COMPLETE:", 1)[1].strip()
            return {"complete": True, "answer": answer}
        
        if "WORKER:" in content:
            parts = content.split("WORKER:", 1)[1]
            if "TASK:" in parts:
                worker_part, task_part = parts.split("TASK:", 1)
                return {
                    "worker": worker_part.strip(),
                    "task": task_part.strip(),
                }
        
        # Default to first worker
        return {
            "worker": list(self._workers.keys())[0],
            "task": memory["task"],
        }


# Factory functions
def create_router(*agents: Any, classifier: Optional[Callable] = None) -> RouterPattern:
    """Quick factory for creating a router"""
    router = RouterPattern(classifier=classifier)
    for i, agent in enumerate(agents):
        name = getattr(agent, 'name', f'agent_{i}')
        router.add_route(name, agent)
    return router


def create_chain(*agents: Any) -> ChainPattern:
    """Quick factory for creating a chain"""
    chain = ChainPattern()
    for i, agent in enumerate(agents):
        name = getattr(agent, 'name', f'step_{i}')
        chain.add(name, agent)
    return chain
