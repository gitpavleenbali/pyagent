"""
Blueprint Module - Agent architecture and orchestration patterns

Blueprints define HOW agents are structured and orchestrated:
- Blueprint: Declarative agent configuration
- Workflow: Multi-step agent processes
- Pipeline: Data flow between agents/skills
- Orchestrator: Multi-agent coordination

Use blueprints to:
- Define reusable agent configurations
- Create complex multi-agent systems
- Build reproducible agent architectures
"""

from pyagent.blueprint.blueprint import Blueprint, AgentBlueprint
from pyagent.blueprint.workflow import Workflow, Step, StepType, WorkflowContext
from pyagent.blueprint.pipeline import Pipeline, PipelineStage
from pyagent.blueprint.orchestrator import Orchestrator, AgentPool
from pyagent.blueprint.patterns import (
    RouterPattern,
    ChainPattern,
    MapReducePattern,
    SupervisorPattern,
)

__all__ = [
    # Core
    "Blueprint",
    "AgentBlueprint",
    # Workflow
    "Workflow",
    "Step",
    "StepType",
    "WorkflowContext",
    # Pipeline
    "Pipeline",
    "PipelineStage",
    # Orchestration
    "Orchestrator",
    "AgentPool",
    # Patterns
    "RouterPattern",
    "ChainPattern",
    "MapReducePattern",
    "SupervisorPattern",
]
