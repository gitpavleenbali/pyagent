"""
Instructions Module - Define agent behavior, persona, and reasoning

Instructions are the core mechanism for shaping how an agent thinks,
responds, and behaves. They include:

- Instruction: Base class for all instruction types
- SystemPrompt: The primary persona and behavior definition  
- Context: Dynamic context injection
- Persona: Reusable personality templates
- Guidelines: Behavioral rules and constraints
"""

from pyagent.instructions.instruction import Instruction
from pyagent.instructions.system_prompt import SystemPrompt
from pyagent.instructions.context import Context, DynamicContext
from pyagent.instructions.persona import Persona
from pyagent.instructions.guidelines import Guidelines, Rule

__all__ = [
    "Instruction",
    "SystemPrompt",
    "Context",
    "DynamicContext",
    "Persona",
    "Guidelines",
    "Rule",
]
