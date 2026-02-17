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

from pyai.blueprint.blueprint import AgentBlueprint, Blueprint
from pyai.blueprint.orchestrator import AgentPool, Orchestrator
from pyai.blueprint.patterns import (
    ChainPattern,
    MapReducePattern,
    RouterPattern,
    SupervisorPattern,
)
from pyai.blueprint.pipeline import Pipeline, PipelineStage
from pyai.blueprint.workflow import Step, StepType, Workflow, WorkflowContext

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
