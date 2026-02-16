"""
Persona - Reusable personality templates for agents
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from pyagent.instructions.instruction import Instruction
from pyagent.instructions.system_prompt import ResponseStyle


class Trait(Enum):
    """Personality traits that affect agent behavior"""
    HELPFUL = "helpful"
    CURIOUS = "curious"
    PATIENT = "patient"
    THOROUGH = "thorough"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    EMPATHETIC = "empathetic"
    ASSERTIVE = "assertive"
    CAUTIOUS = "cautious"
    HUMOROUS = "humorous"


@dataclass
class Persona(Instruction):
    """
    Persona - A reusable personality definition for agents.
    
    Personas encapsulate personality traits, communication style,
    and behavioral patterns that can be applied to any agent.
    
    Example:
        >>> expert_persona = Persona(
        ...     name="TechExpert",
        ...     traits=[Trait.THOROUGH, Trait.ANALYTICAL],
        ...     voice="Speak with confidence and precision",
        ...     quirks=["Often uses analogies to explain concepts"],
        ... )
    """
    
    # Identity
    persona_name: str = "Assistant"
    backstory: str = ""
    expertise: List[str] = field(default_factory=list)
    
    # Personality
    traits: List[Trait] = field(default_factory=list)
    voice: str = ""  # How the persona speaks
    tone: str = "neutral"  # Overall emotional tone
    
    # Behavioral
    quirks: List[str] = field(default_factory=list)
    catchphrases: List[str] = field(default_factory=list)
    avoid_phrases: List[str] = field(default_factory=list)
    
    # Response style
    response_style: ResponseStyle = ResponseStyle.PROFESSIONAL
    
    def validate(self) -> bool:
        """Validate persona has basic identity"""
        return bool(self.persona_name)
    
    def render(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Render persona as instruction text"""
        parts = []
        
        # Core identity
        parts.append(f"You are {self.persona_name}.")
        
        if self.backstory:
            parts.append(f"\n{self.backstory}")
        
        # Expertise
        if self.expertise:
            parts.append("\n## Expertise")
            for exp in self.expertise:
                parts.append(f"- {exp}")
        
        # Personality traits
        if self.traits:
            trait_desc = self._describe_traits()
            parts.append(f"\n## Personality\n{trait_desc}")
        
        # Voice and tone
        if self.voice:
            parts.append(f"\n## Voice\n{self.voice}")
        
        # Quirks
        if self.quirks:
            parts.append("\n## Behavioral Notes")
            for quirk in self.quirks:
                parts.append(f"- {quirk}")
        
        # Things to avoid
        if self.avoid_phrases:
            parts.append("\n## Avoid")
            for phrase in self.avoid_phrases:
                parts.append(f"- Don't say: \"{phrase}\"")
        
        return "\n".join(parts)
    
    def _describe_traits(self) -> str:
        """Convert traits to descriptive text"""
        trait_descriptions = {
            Trait.HELPFUL: "You are genuinely helpful and eager to assist.",
            Trait.CURIOUS: "You ask thoughtful questions to understand better.",
            Trait.PATIENT: "You are patient and never rush the user.",
            Trait.THOROUGH: "You provide complete, comprehensive answers.",
            Trait.CREATIVE: "You think creatively and offer novel solutions.",
            Trait.ANALYTICAL: "You break down problems systematically.",
            Trait.EMPATHETIC: "You understand and acknowledge user feelings.",
            Trait.ASSERTIVE: "You confidently share your recommendations.",
            Trait.CAUTIOUS: "You carefully consider risks and edge cases.",
            Trait.HUMOROUS: "You use appropriate humor to keep things light.",
        }
        
        descriptions = [trait_descriptions.get(t, str(t)) for t in self.traits]
        return " ".join(descriptions)
    
    def with_trait(self, trait: Trait) -> "Persona":
        """Add a trait and return self for chaining"""
        if trait not in self.traits:
            self.traits.append(trait)
        return self
    
    def with_expertise(self, *areas: str) -> "Persona":
        """Add expertise areas"""
        self.expertise.extend(areas)
        return self
    
    @classmethod
    def default(cls) -> "Persona":
        """Create a default helpful assistant persona"""
        return cls(
            persona_name="Assistant",
            traits=[Trait.HELPFUL, Trait.PATIENT, Trait.THOROUGH],
            voice="Speak clearly and professionally",
            response_style=ResponseStyle.PROFESSIONAL,
        )
    
    @classmethod
    def expert(cls, domain: str) -> "Persona":
        """Create an expert persona for a domain"""
        return cls(
            persona_name=f"{domain.title()} Expert",
            traits=[Trait.THOROUGH, Trait.ANALYTICAL, Trait.ASSERTIVE],
            expertise=[domain],
            voice=f"Speak with authority on {domain} topics",
            response_style=ResponseStyle.TECHNICAL,
        )
    
    @classmethod
    def mentor(cls) -> "Persona":
        """Create a patient mentor persona"""
        return cls(
            persona_name="Mentor",
            traits=[Trait.PATIENT, Trait.HELPFUL, Trait.EMPATHETIC],
            voice="Speak encouragingly, guiding rather than just answering",
            quirks=[
                "Ask clarifying questions before diving in",
                "Offer to explain concepts if they seem unclear",
            ],
            response_style=ResponseStyle.FRIENDLY,
        )


# Pre-built personas
PERSONAS = {
    "default": Persona.default(),
    "coder": Persona(
        persona_name="Code Expert",
        expertise=["Software development", "Code review", "Best practices"],
        traits=[Trait.THOROUGH, Trait.ANALYTICAL],
        voice="Clear, technical, with code examples",
        response_style=ResponseStyle.TECHNICAL,
    ),
    "researcher": Persona(
        persona_name="Research Analyst",
        expertise=["Information synthesis", "Critical analysis"],
        traits=[Trait.CURIOUS, Trait.THOROUGH, Trait.CAUTIOUS],
        voice="Objective and well-sourced",
        response_style=ResponseStyle.DETAILED,
    ),
    "creative": Persona(
        persona_name="Creative Partner",
        traits=[Trait.CREATIVE, Trait.CURIOUS, Trait.HUMOROUS],
        voice="Imaginative and encouraging",
        response_style=ResponseStyle.CONVERSATIONAL,
    ),
}


def get_persona(name: str) -> Persona:
    """Get a pre-built persona by name"""
    if name not in PERSONAS:
        available = ", ".join(PERSONAS.keys())
        raise ValueError(f"Unknown persona: {name}. Available: {available}")
    return PERSONAS[name]
