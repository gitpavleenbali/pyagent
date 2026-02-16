"""
Blueprint - Declarative agent configuration
"""

from typing import Any, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
import yaml
import json

from pyagent.core.base import BaseComponent
from pyagent.core.llm import LLMConfig, ModelProvider


@dataclass
class SkillConfig:
    """Configuration for a skill in a blueprint"""
    
    name: str
    type: str  # Class name or registered skill name
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class InstructionConfig:
    """Configuration for instructions in a blueprint"""
    
    system_prompt: str = ""
    persona: Optional[str] = None  # Name of a pre-built persona
    guidelines: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


@dataclass
class AgentBlueprint(BaseComponent):
    """
    AgentBlueprint - A declarative specification for creating agents.
    
    Blueprints allow you to define agent configurations in a reusable,
    portable format. They can be:
    - Created programmatically
    - Loaded from YAML/JSON files
    - Shared between projects
    - Version controlled
    
    Example:
        >>> blueprint = AgentBlueprint(
        ...     name="ResearchAgent",
        ...     instructions=InstructionConfig(
        ...         system_prompt="You are a research assistant",
        ...         goals=["Find accurate information"]
        ...     ),
        ...     skills=[
        ...         SkillConfig(name="search", type="SearchSkill"),
        ...         SkillConfig(name="summarize", type="SummarySkill"),
        ...     ],
        ... )
        >>> agent = blueprint.build()
    """
    
    # Identity
    name: str = "Agent"
    version: str = "1.0.0"
    
    # Configuration
    instructions: Optional[InstructionConfig] = None
    skills: List[SkillConfig] = field(default_factory=list)
    llm: Optional[LLMConfig] = None
    
    # Agent settings
    max_iterations: int = 10
    timeout: float = 300.0
    enable_memory: bool = True
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    author: str = ""
    
    def validate(self) -> bool:
        """Validate blueprint configuration"""
        if not self.name:
            return False
        if not self.instructions and not self.skills:
            return False
        return True
    
    def build(self) -> "Agent":
        """
        Build an Agent from this blueprint.
        
        Returns:
            Configured Agent instance
        """
        from pyagent.core.agent import Agent, AgentConfig
        from pyagent.instructions import Instruction, SystemPrompt
        from pyagent.instructions.persona import get_persona, PERSONAS
        from pyagent.skills.registry import get_default_registry
        
        # Build instructions
        instruction = None
        if self.instructions:
            if self.instructions.persona and self.instructions.persona in PERSONAS:
                instruction = get_persona(self.instructions.persona)
            elif self.instructions.system_prompt:
                instruction = SystemPrompt(
                    content=self.instructions.system_prompt,
                    goals=self.instructions.goals,
                    constraints=self.instructions.constraints,
                )
        
        # Build skills
        skills = []
        registry = get_default_registry()
        for skill_config in self.skills:
            if not skill_config.enabled:
                continue
            
            # Try to get from registry
            skill = registry.get(skill_config.name)
            if skill:
                skills.append(skill)
            else:
                # Try to instantiate from type
                skill = self._instantiate_skill(skill_config)
                if skill:
                    skills.append(skill)
        
        # Build LLM provider
        llm = None
        if self.llm:
            llm = self._create_llm_provider(self.llm)
        
        # Create agent config
        config = AgentConfig(
            max_iterations=self.max_iterations,
            timeout_seconds=self.timeout,
            enable_memory=self.enable_memory,
        )
        
        return Agent(
            name=self.name,
            instructions=instruction,
            skills=skills,
            llm=llm,
            config=config,
        )
    
    def _instantiate_skill(self, config: SkillConfig):
        """Instantiate a skill from its configuration"""
        from pyagent.skills.builtin import (
            SearchSkill, CodeSkill, FileSkill, WebSkill, MathSkill
        )
        
        skill_types = {
            "SearchSkill": SearchSkill,
            "CodeSkill": CodeSkill,
            "FileSkill": FileSkill,
            "WebSkill": WebSkill,
            "MathSkill": MathSkill,
        }
        
        skill_class = skill_types.get(config.type)
        if skill_class:
            return skill_class(**config.config)
        return None
    
    def _create_llm_provider(self, config: LLMConfig):
        """Create an LLM provider from configuration"""
        from pyagent.core.llm import (
            OpenAIProvider, AzureOpenAIProvider, AnthropicProvider
        )
        
        provider_map = {
            ModelProvider.OPENAI: OpenAIProvider,
            ModelProvider.AZURE_OPENAI: AzureOpenAIProvider,
            ModelProvider.ANTHROPIC: AnthropicProvider,
        }
        
        provider_class = provider_map.get(config.provider)
        if provider_class:
            return provider_class(config)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize blueprint to dictionary"""
        return {
            "name": self.name,
            "version": self.version,
            "instructions": {
                "system_prompt": self.instructions.system_prompt if self.instructions else "",
                "persona": self.instructions.persona if self.instructions else None,
                "goals": self.instructions.goals if self.instructions else [],
                "constraints": self.instructions.constraints if self.instructions else [],
            },
            "skills": [
                {
                    "name": s.name,
                    "type": s.type,
                    "config": s.config,
                    "enabled": s.enabled,
                }
                for s in self.skills
            ],
            "llm": {
                "provider": self.llm.provider.value if self.llm else "openai",
                "model": self.llm.model if self.llm else "gpt-4",
            } if self.llm else None,
            "settings": {
                "max_iterations": self.max_iterations,
                "timeout": self.timeout,
                "enable_memory": self.enable_memory,
            },
            "metadata": {
                "tags": self.tags,
                "author": self.author,
            },
        }
    
    def to_yaml(self) -> str:
        """Serialize blueprint to YAML"""
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    def to_json(self) -> str:
        """Serialize blueprint to JSON"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentBlueprint":
        """Create blueprint from dictionary"""
        instructions = None
        if data.get("instructions"):
            inst_data = data["instructions"]
            instructions = InstructionConfig(
                system_prompt=inst_data.get("system_prompt", ""),
                persona=inst_data.get("persona"),
                goals=inst_data.get("goals", []),
                constraints=inst_data.get("constraints", []),
            )
        
        skills = []
        for skill_data in data.get("skills", []):
            skills.append(SkillConfig(
                name=skill_data["name"],
                type=skill_data.get("type", skill_data["name"]),
                config=skill_data.get("config", {}),
                enabled=skill_data.get("enabled", True),
            ))
        
        llm = None
        if data.get("llm"):
            llm_data = data["llm"]
            llm = LLMConfig(
                provider=ModelProvider(llm_data.get("provider", "openai")),
                model=llm_data.get("model", "gpt-4"),
            )
        
        settings = data.get("settings", {})
        metadata = data.get("metadata", {})
        
        return cls(
            name=data.get("name", "Agent"),
            version=data.get("version", "1.0.0"),
            instructions=instructions,
            skills=skills,
            llm=llm,
            max_iterations=settings.get("max_iterations", 10),
            timeout=settings.get("timeout", 300.0),
            enable_memory=settings.get("enable_memory", True),
            tags=metadata.get("tags", []),
            author=metadata.get("author", ""),
        )
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> "AgentBlueprint":
        """Create blueprint from YAML string"""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_yaml_file(cls, path: str) -> "AgentBlueprint":
        """Load blueprint from YAML file"""
        with open(path, 'r') as f:
            return cls.from_yaml(f.read())
    
    @classmethod
    def from_json(cls, json_str: str) -> "AgentBlueprint":
        """Create blueprint from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)


# Alias for backwards compatibility
Blueprint = AgentBlueprint


class BlueprintRegistry:
    """
    Registry for storing and retrieving blueprints.
    
    Example:
        >>> registry = BlueprintRegistry()
        >>> registry.register(my_blueprint)
        >>> blueprint = registry.get("MyAgent")
    """
    
    def __init__(self):
        self._blueprints: Dict[str, AgentBlueprint] = {}
    
    def register(self, blueprint: AgentBlueprint) -> None:
        """Register a blueprint"""
        self._blueprints[blueprint.name] = blueprint
    
    def get(self, name: str) -> Optional[AgentBlueprint]:
        """Get a blueprint by name"""
        return self._blueprints.get(name)
    
    def list(self) -> List[str]:
        """List all registered blueprint names"""
        return list(self._blueprints.keys())
    
    def build(self, name: str) -> Optional["Agent"]:
        """Build an agent from a registered blueprint"""
        blueprint = self.get(name)
        if blueprint:
            return blueprint.build()
        return None
