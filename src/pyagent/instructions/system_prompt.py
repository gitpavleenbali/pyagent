"""
SystemPrompt - The primary persona and behavior definition
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from pyagent.instructions.instruction import Instruction


class ResponseStyle(Enum):
    """Predefined response styles"""
    CONCISE = "concise"
    DETAILED = "detailed"
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"


@dataclass
class SystemPrompt(Instruction):
    """
    SystemPrompt - The primary instruction that defines an agent's identity.
    
    A SystemPrompt is the foundation of agent behavior, defining:
    - Identity: Who the agent is
    - Capabilities: What the agent can do
    - Style: How the agent communicates
    - Boundaries: What the agent should not do
    
    Example:
        >>> prompt = SystemPrompt(
        ...     identity="You are a senior software engineer",
        ...     capabilities=["Code review", "Architecture design"],
        ...     style=ResponseStyle.TECHNICAL,
        ... )
    """
    
    identity: str = ""
    capabilities: List[str] = field(default_factory=list)
    style: ResponseStyle = ResponseStyle.PROFESSIONAL
    boundaries: List[str] = field(default_factory=list)
    knowledge_cutoff: Optional[str] = None
    
    def validate(self) -> bool:
        """Validate system prompt has identity"""
        return bool(self.identity or self.content)
    
    def render(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Render the system prompt"""
        parts = []
        
        # Identity
        if self.identity:
            parts.append(self.identity)
        elif self.content:
            parts.append(self.content)
        
        # Capabilities
        if self.capabilities:
            parts.append("\n## Your Capabilities")
            for cap in self.capabilities:
                parts.append(f"- {cap}")
        
        # Style instructions
        style_instructions = self._get_style_instructions()
        if style_instructions:
            parts.append(f"\n## Communication Style\n{style_instructions}")
        
        # Boundaries
        if self.boundaries:
            parts.append("\n## Boundaries")
            for boundary in self.boundaries:
                parts.append(f"- {boundary}")
        
        # Knowledge cutoff
        if self.knowledge_cutoff:
            parts.append(f"\nKnowledge cutoff: {self.knowledge_cutoff}")
        
        # Render parent content (goals, constraints, examples)
        parent_content = super().render(context)
        if parent_content and parent_content != self.content:
            parts.append(parent_content)
        
        return "\n".join(parts)
    
    def _get_style_instructions(self) -> str:
        """Get style-specific instructions"""
        style_map = {
            ResponseStyle.CONCISE: "Be brief and to the point. Avoid unnecessary elaboration.",
            ResponseStyle.DETAILED: "Provide thorough explanations with examples and context.",
            ResponseStyle.CONVERSATIONAL: "Respond in a natural, friendly manner. Use casual language.",
            ResponseStyle.TECHNICAL: "Use precise technical language. Include code examples when relevant.",
            ResponseStyle.FRIENDLY: "Be warm and approachable. Use encouraging language.",
            ResponseStyle.PROFESSIONAL: "Maintain a professional tone. Be clear and respectful.",
        }
        return style_map.get(self.style, "")
    
    @classmethod
    def from_template(cls, template: str, **kwargs) -> "SystemPrompt":
        """Create a SystemPrompt from a predefined template"""
        templates = {
            "coding_assistant": cls(
                identity="You are an expert software engineer with deep knowledge of multiple programming languages.",
                capabilities=[
                    "Write clean, efficient code",
                    "Debug and troubleshoot issues",
                    "Explain complex concepts clearly",
                    "Review code and suggest improvements",
                ],
                style=ResponseStyle.TECHNICAL,
            ),
            "research_assistant": cls(
                identity="You are a research assistant skilled at finding, analyzing, and synthesizing information.",
                capabilities=[
                    "Search and analyze information",
                    "Summarize complex topics",
                    "Identify key insights and patterns",
                    "Provide balanced perspectives",
                ],
                style=ResponseStyle.DETAILED,
            ),
            "customer_support": cls(
                identity="You are a friendly customer support specialist dedicated to helping users.",
                capabilities=[
                    "Answer questions about products/services",
                    "Troubleshoot common issues",
                    "Guide users through processes",
                    "Escalate complex issues appropriately",
                ],
                style=ResponseStyle.FRIENDLY,
            ),
        }
        
        if template not in templates:
            raise ValueError(f"Unknown template: {template}. Available: {list(templates.keys())}")
        
        return templates[template]


class SystemPromptBuilder:
    """Fluent builder for SystemPrompt"""
    
    def __init__(self):
        self._identity = ""
        self._capabilities = []
        self._style = ResponseStyle.PROFESSIONAL
        self._boundaries = []
        self._goals = []
        self._constraints = []
    
    def with_identity(self, identity: str) -> "SystemPromptBuilder":
        """Set agent identity"""
        self._identity = identity
        return self
    
    def can(self, *capabilities: str) -> "SystemPromptBuilder":
        """Add capabilities"""
        self._capabilities.extend(capabilities)
        return self
    
    def with_style(self, style: ResponseStyle) -> "SystemPromptBuilder":
        """Set communication style"""
        self._style = style
        return self
    
    def must_not(self, *boundaries: str) -> "SystemPromptBuilder":
        """Add boundaries"""
        self._boundaries.extend(boundaries)
        return self
    
    def should(self, *goals: str) -> "SystemPromptBuilder":
        """Add goals"""
        self._goals.extend(goals)
        return self
    
    def build(self) -> SystemPrompt:
        """Build the SystemPrompt"""
        return SystemPrompt(
            identity=self._identity,
            capabilities=self._capabilities,
            style=self._style,
            boundaries=self._boundaries,
            goals=list(self._goals),
            constraints=list(self._constraints),
        )
