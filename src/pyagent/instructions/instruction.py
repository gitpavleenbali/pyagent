"""
Instruction - Base class for defining agent behavior
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from abc import abstractmethod

from pyagent.core.base import BaseComponent


@dataclass
class Instruction(BaseComponent):
    """
    Base Instruction class - the foundation for agent behavior definition.
    
    Instructions tell an agent:
    - WHO it is (persona, identity)
    - WHAT it should do (tasks, goals)
    - HOW it should respond (style, format)
    - WHAT to avoid (constraints, limitations)
    
    Example:
        >>> instruction = Instruction(
        ...     content="You are a helpful coding assistant",
        ...     role="assistant",
        ...     goals=["Help users write better code"],
        ... )
        >>> prompt = instruction.render()
    """
    
    content: str = ""
    role: str = "assistant"
    goals: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)
    output_format: Optional[str] = None
    
    # Composition
    children: List["Instruction"] = field(default_factory=list)
    priority: int = 0  # Higher priority instructions are rendered first
    
    def validate(self) -> bool:
        """Validate instruction has minimum required content"""
        return bool(self.content or self.children)
    
    def render(self, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Render the instruction to a string prompt.
        
        Args:
            context: Optional context variables for templating
            
        Returns:
            Rendered instruction string
        """
        parts = []
        
        # Main content
        if self.content:
            content = self._render_template(self.content, context)
            parts.append(content)
        
        # Goals
        if self.goals:
            parts.append("\n## Goals")
            for goal in self.goals:
                parts.append(f"- {goal}")
        
        # Constraints
        if self.constraints:
            parts.append("\n## Constraints")
            for constraint in self.constraints:
                parts.append(f"- {constraint}")
        
        # Output format
        if self.output_format:
            parts.append(f"\n## Output Format\n{self.output_format}")
        
        # Examples
        if self.examples:
            parts.append("\n## Examples")
            for i, example in enumerate(self.examples, 1):
                parts.append(f"\n### Example {i}")
                if "input" in example:
                    parts.append(f"Input: {example['input']}")
                if "output" in example:
                    parts.append(f"Output: {example['output']}")
        
        # Render children
        if self.children:
            sorted_children = sorted(self.children, key=lambda x: -x.priority)
            for child in sorted_children:
                parts.append("\n" + child.render(context))
        
        return "\n".join(parts)
    
    def _render_template(self, template: str, context: Optional[Dict[str, Any]]) -> str:
        """Simple template rendering with {variable} substitution"""
        if not context:
            return template
        
        result = template
        for key, value in context.items():
            result = result.replace(f"{{{key}}}", str(value))
        
        return result
    
    def add_goal(self, goal: str) -> "Instruction":
        """Add a goal and return self for chaining"""
        self.goals.append(goal)
        return self
    
    def add_constraint(self, constraint: str) -> "Instruction":
        """Add a constraint and return self for chaining"""
        self.constraints.append(constraint)
        return self
    
    def add_example(self, input_text: str, output_text: str) -> "Instruction":
        """Add an example and return self for chaining"""
        self.examples.append({"input": input_text, "output": output_text})
        return self
    
    def compose(self, *instructions: "Instruction") -> "Instruction":
        """Compose multiple instructions together"""
        self.children.extend(instructions)
        return self
    
    def __add__(self, other: "Instruction") -> "Instruction":
        """Combine instructions with + operator"""
        combined = Instruction(
            content=self.content,
            role=self.role,
            goals=self.goals.copy(),
            constraints=self.constraints.copy(),
            examples=self.examples.copy(),
        )
        combined.children.append(self)
        combined.children.append(other)
        return combined
    
    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Instruction('{preview}')"


class InstructionBuilder:
    """
    Fluent builder for creating Instructions.
    
    Example:
        >>> instruction = (InstructionBuilder()
        ...     .as_role("coding assistant")
        ...     .with_goal("Write clean, efficient code")
        ...     .with_goal("Explain code clearly")
        ...     .with_constraint("Use Python best practices")
        ...     .with_format("```python\n# code here\n```")
        ...     .build())
    """
    
    def __init__(self):
        self._content = ""
        self._role = "assistant"
        self._goals = []
        self._constraints = []
        self._examples = []
        self._output_format = None
        self._children = []
    
    def with_content(self, content: str) -> "InstructionBuilder":
        """Set the main instruction content"""
        self._content = content
        return self
    
    def as_role(self, role: str) -> "InstructionBuilder":
        """Set the agent's role/identity"""
        self._role = role
        self._content = f"You are a {role}."
        return self
    
    def with_goal(self, goal: str) -> "InstructionBuilder":
        """Add a goal"""
        self._goals.append(goal)
        return self
    
    def with_constraint(self, constraint: str) -> "InstructionBuilder":
        """Add a constraint"""
        self._constraints.append(constraint)
        return self
    
    def with_example(self, input_text: str, output_text: str) -> "InstructionBuilder":
        """Add an example"""
        self._examples.append({"input": input_text, "output": output_text})
        return self
    
    def with_format(self, format_spec: str) -> "InstructionBuilder":
        """Set the output format"""
        self._output_format = format_spec
        return self
    
    def with_child(self, instruction: Instruction) -> "InstructionBuilder":
        """Add a child instruction"""
        self._children.append(instruction)
        return self
    
    def build(self) -> Instruction:
        """Build and return the Instruction"""
        return Instruction(
            content=self._content,
            role=self._role,
            goals=self._goals,
            constraints=self._constraints,
            examples=self._examples,
            output_format=self._output_format,
            children=self._children,
        )
