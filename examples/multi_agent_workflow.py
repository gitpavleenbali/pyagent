"""
Example: Multi-Agent Workflow

This example demonstrates how to create a multi-agent workflow
where different agents collaborate on a task.

NOTE: This is a REFERENCE example showing the workflow architecture.
      For simpler one-liner usage, see: comprehensive_examples.py

Authentication (if running with LLM):
    # Option 1: OpenAI API Key
    export OPENAI_API_KEY=sk-your-key
    
    # Option 2: Azure OpenAI with Azure AD (recommended - no key needed)
    export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
    export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
"""

import os
import sys

# Add paths for local development (works from any directory, including PyCharm)
_examples_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_examples_dir)
sys.path.insert(0, _project_dir)  # For pyagent imports
sys.path.insert(0, _examples_dir)  # For config_helper import

# Optional: Configure PyAgent for LLM-powered features
# This example demonstrates workflow structure without requiring LLM calls
try:
    from config_helper import setup_pyagent
    setup_pyagent(verbose=False)  # Silent - this example doesn't require LLM
except Exception:
    pass  # Workflow definitions work without LLM configuration

import asyncio
from pyagent import Agent
from pyagent.instructions import SystemPrompt
from pyagent.blueprint import Workflow, Step, StepType, WorkflowContext
from pyagent.blueprint.patterns import ChainPattern, SupervisorPattern


async def create_research_workflow():
    """
    Create a research workflow with multiple specialized agents.
    
    Workflow:
    1. Researcher: Finds information on a topic
    2. Analyst: Analyzes and structures the information
    3. Writer: Creates a polished report
    """
    
    # Create specialized agents (you would configure LLM for each)
    researcher = Agent(
        name="Researcher",
        instructions=SystemPrompt(
            identity="You are an expert researcher",
            capabilities=["Find relevant information", "Identify key sources"],
        ),
    )
    
    analyst = Agent(
        name="Analyst",
        instructions=SystemPrompt(
            identity="You are a data analyst",
            capabilities=["Structure information", "Identify patterns"],
        ),
    )
    
    writer = Agent(
        name="Writer",
        instructions=SystemPrompt(
            identity="You are a technical writer",
            capabilities=["Write clear reports", "Explain complex topics"],
        ),
    )
    
    # Create workflow
    workflow = (Workflow("ResearchPipeline")
        .add_step(Step(
            name="research",
            step_type=StepType.AGENT,
            agent=researcher,
            output_mapping={"content": "research_results"}
        ))
        .add_step(Step(
            name="analyze",
            step_type=StepType.AGENT,
            agent=analyst,
            input_mapping={"message": "research_results"},
            output_mapping={"content": "analysis"}
        ))
        .add_step(Step(
            name="write",
            step_type=StepType.AGENT,
            agent=writer,
            input_mapping={"message": "analysis"},
        ))
    )
    
    return workflow


async def create_chain_example():
    """
    Create a simple chain of agents using the ChainPattern.
    """
    
    draft_agent = Agent(name="Drafter")
    review_agent = Agent(name="Reviewer")
    edit_agent = Agent(name="Editor")
    
    chain = (ChainPattern("ContentChain")
        .add("draft", draft_agent)
        .add("review", review_agent)
        .add("edit", edit_agent)
    )
    
    return chain


async def create_supervisor_example():
    """
    Create a supervisor pattern where one agent manages workers.
    """
    
    supervisor = Agent(
        name="ProjectManager",
        instructions=SystemPrompt(
            identity="You are a project manager coordinating workers"
        ),
    )
    
    workers = {
        "researcher": Agent(name="Researcher"),
        "developer": Agent(name="Developer"),
        "tester": Agent(name="Tester"),
    }
    
    pattern = SupervisorPattern(
        supervisor=supervisor,
        workers=workers,
    )
    
    return pattern


async def main():
    # Example 1: Workflow
    print("=== Research Workflow ===")
    workflow = await create_research_workflow()
    print(f"Created workflow: {workflow}")
    
    # To run (would need configured LLMs):
    # result = await workflow.run(topic="Quantum Computing Trends")
    
    # Example 2: Chain
    print("\n=== Content Chain ===")
    chain = await create_chain_example()
    print(f"Created chain: {chain}")
    
    # Example 3: Supervisor
    print("\n=== Supervisor Pattern ===")
    supervisor = await create_supervisor_example()
    print(f"Created supervisor pattern with {len(supervisor._workers)} workers")


if __name__ == "__main__":
    asyncio.run(main())
