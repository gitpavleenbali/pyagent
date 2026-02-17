"""# pyright: reportMissingImports=false, reportUnusedImport=falseExample: Basic Agent Usage

This example demonstrates the lower-level Agent API.
For simpler one-liner usage, see: comprehensive_examples.py

NOTE: This is a REFERENCE example showing the core Agent architecture.

Authentication Options:
    # Option 1: OpenAI API Key
    export OPENAI_API_KEY=sk-your-key
    
    # Option 2: Azure OpenAI with Azure AD (recommended for enterprise)
    export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
    export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
    # Uses your Azure login automatically (az login)
"""

import os
import sys

# Add paths for local development (works from any directory, including PyCharm)
_examples_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_examples_dir)
sys.path.insert(0, _project_dir)  # For pyai imports
sys.path.insert(0, _examples_dir)  # For config_helper import

# Configure pyai with available credentials (supports OpenAI, Azure API Key, or Azure AD)
from config_helper import setup_pyai
if not setup_pyai():
    print("Please configure credentials - see instructions above")
    sys.exit(1)

import asyncio
from pyai import Agent
from pyai.instructions import SystemPrompt
from pyai.skills import SearchSkill, CodeSkill
from pyai.core.llm import OpenAIProvider, AzureOpenAIProvider, LLMConfig


def get_llm_provider():
    """Get the appropriate LLM provider based on environment configuration."""
    if os.environ.get("AZURE_OPENAI_ENDPOINT"):
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        
        return AzureOpenAIProvider(LLMConfig(
            api_key=api_key,  # Can be None for Azure AD auth
            api_base=endpoint,
            model=deployment,
            temperature=0.7,
        ))
    else:
        return OpenAIProvider(LLMConfig(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            temperature=0.7,
        ))


async def main():
    # 1. Create instructions
    instructions = SystemPrompt(
        identity="You are a helpful AI assistant with expertise in programming",
        capabilities=[
            "Answer questions about code",
            "Help debug issues",
            "Suggest best practices",
        ],
        goals=[
            "Provide accurate, helpful responses",
            "Explain concepts clearly",
        ],
        constraints=[
            "Always test code before suggesting",
            "Cite sources for claims",
        ],
    )
    
    # 2. Configure LLM (auto-detects OpenAI or Azure from environment)
    llm = get_llm_provider()
    
    # 3. Create skills
    skills = [
        SearchSkill(),
        CodeSkill(),
    ]
    
    # 4. Create agent
    agent = Agent(
        name="CodingAssistant",
        instructions=instructions,
        skills=skills,
        llm=llm,
    )
    
    # 5. Run agent
    response = await agent.run(
        "How do I implement a binary search in Python?"
    )
    
    print(f"Response: {response.content}")
    print(f"Success: {response.success}")
    print(f"Iterations: {response.iterations}")


if __name__ == "__main__":
    asyncio.run(main())
