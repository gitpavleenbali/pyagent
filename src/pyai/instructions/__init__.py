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

from pyai.instructions.context import Context, DynamicContext
from pyai.instructions.guidelines import Guidelines, Rule
from pyai.instructions.instruction import Instruction
from pyai.instructions.persona import Persona
from pyai.instructions.system_prompt import SystemPrompt

__all__ = [
    "Instruction",
    "SystemPrompt",
    "Context",
    "DynamicContext",
    "Persona",
    "Guidelines",
    "Rule",
]
