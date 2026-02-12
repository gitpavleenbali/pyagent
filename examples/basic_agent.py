"""
Example: Basic Agent Usage

This example demonstrates the lower-level Agent API.
For simpler one-liner usage, see: comprehensive_examples.py

NOTE: This is a REFERENCE example showing the core Agent architecture.
"""

import os
import sys

# Add pyagent to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from pyagent import Agent
from pyagent.instructions import Instruction, SystemPrompt
from pyagent.skills import SearchSkill, CodeSkill
from pyagent.core.llm import OpenAIProvider, LLMConfig


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
    
    # 2. Configure LLM (you would use your real API key)
    llm = OpenAIProvider(LLMConfig(
        api_key="your-api-key-here",  # Replace with real key
        model="gpt-4",
        temperature=0.7,
    ))
    
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
