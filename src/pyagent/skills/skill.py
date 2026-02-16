"""
Skill - Base class for agent capabilities
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

from pyagent.core.base import BaseComponent, Executable

if TYPE_CHECKING:
    from pyagent.core.agent import Agent


class SkillStatus(Enum):
    """Status of a skill execution"""
    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"
    CANCELLED = "cancelled"


@dataclass
class SkillResult:
    """
    Result from a skill execution.
    
    Standardizes skill outputs for consistent handling by agents.
    """
    
    status: SkillStatus = SkillStatus.SUCCESS
    data: Any = None
    message: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if skill execution was successful"""
        return self.status == SkillStatus.SUCCESS
    
    @classmethod
    def ok(cls, data: Any = None, message: str = "") -> "SkillResult":
        """Create a successful result"""
        return cls(status=SkillStatus.SUCCESS, data=data, message=message)
    
    @classmethod
    def fail(cls, error: str, data: Any = None) -> "SkillResult":
        """Create a failure result"""
        return cls(status=SkillStatus.FAILURE, error=error, data=data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "status": self.status.value,
            "data": self.data,
            "message": self.message,
            "error": self.error,
            "metadata": self.metadata,
        }
    
    def __str__(self) -> str:
        if self.success:
            return str(self.data) if self.data else self.message
        return f"Error: {self.error}"


@dataclass
class SkillParameter:
    """Definition of a skill parameter"""
    
    name: str
    description: str
    type: str = "string"  # string, integer, number, boolean, array, object
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None  # Allowed values
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format"""
        schema = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


class Skill(BaseComponent, Executable, ABC):
    """
    Skill - Base class for agent capabilities.
    
    A Skill represents something an agent can DO. Skills are:
    - Self-describing: They explain what they do for LLM tool-use
    - Executable: They perform actions when invoked
    - Composable: They can be combined with other skills
    
    Example:
        >>> class SearchSkill(Skill):
        ...     name = "search"
        ...     description = "Search the web for information"
        ...     
        ...     async def execute(self, query: str) -> SkillResult:
        ...         results = await web_search(query)
        ...         return SkillResult.ok(results)
    """
    
    # Skill identity
    name: str = "skill"
    description: str = "A skill"
    
    # Parameters this skill accepts
    parameters: List[SkillParameter] = field(default_factory=list)
    
    # Execution settings
    timeout: float = 30.0
    retries: int = 0
    requires_confirmation: bool = False
    
    # Agent binding
    _agent: Optional["Agent"] = None
    
    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs
    ):
        # Handle dataclass-style initialization
        if name:
            self.name = name
        if description:
            self.description = description
        
        # Initialize base component
        super().__init__(name=self.name, description=self.description, **kwargs)
        
        self._agent = None
        self.parameters = []
    
    def bind_to_agent(self, agent: "Agent") -> None:
        """Bind this skill to an agent"""
        self._agent = agent
    
    @property
    def agent(self) -> Optional["Agent"]:
        """Get the bound agent"""
        return self._agent
    
    def validate(self) -> bool:
        """Validate skill configuration"""
        return bool(self.name and self.description)
    
    async def validate_input(self, **kwargs) -> bool:
        """Validate input parameters"""
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                return False
        return True
    
    @abstractmethod
    async def execute(self, **kwargs) -> SkillResult:
        """
        Execute the skill with given parameters.
        
        This is the main method that performs the skill's action.
        Must be implemented by subclasses.
        
        Args:
            **kwargs: Parameters for the skill
            
        Returns:
            SkillResult with the outcome
        """
        pass
    
    def to_tool_definition(self) -> Dict[str, Any]:
        """
        Convert skill to OpenAI-compatible tool definition.
        
        This allows skills to be used with function calling.
        """
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
    
    def add_parameter(
        self,
        name: str,
        description: str,
        type: str = "string",
        required: bool = True,
        **kwargs
    ) -> "Skill":
        """Add a parameter to the skill"""
        param = SkillParameter(
            name=name,
            description=description,
            type=type,
            required=required,
            **kwargs
        )
        self.parameters.append(param)
        return self
    
    async def __call__(self, **kwargs) -> SkillResult:
        """Allow skill to be called directly"""
        return await self.execute(**kwargs)
    
    def __repr__(self) -> str:
        return f"Skill({self.name})"


class PassthroughSkill(Skill):
    """
    A skill that passes through to a callable function.
    
    Useful for quickly wrapping existing functions as skills.
    """
    
    def __init__(
        self,
        func: callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self._func = func
        super().__init__(
            name=name or func.__name__,
            description=description or func.__doc__ or "No description",
        )
    
    async def execute(self, **kwargs) -> SkillResult:
        """Execute the wrapped function"""
        try:
            result = self._func(**kwargs)
            # Handle async functions
            if hasattr(result, '__await__'):
                result = await result
            return SkillResult.ok(result)
        except Exception as e:
            return SkillResult.fail(str(e))


class CompositeSkill(Skill):
    """
    A skill composed of multiple sub-skills.
    
    Executes skills in sequence, passing results between them.
    """
    
    def __init__(
        self,
        skills: List[Skill],
        name: str = "composite",
        description: str = "A composite skill",
    ):
        super().__init__(name=name, description=description)
        self.skills = skills
    
    async def execute(self, **kwargs) -> SkillResult:
        """Execute all sub-skills in sequence"""
        results = []
        current_input = kwargs
        
        for skill in self.skills:
            result = await skill.execute(**current_input)
            if not result.success:
                return result  # Stop on first failure
            
            results.append(result)
            
            # Pass output as input to next skill
            if result.data and isinstance(result.data, dict):
                current_input = {**current_input, **result.data}
        
        return SkillResult.ok(
            data={"results": [r.to_dict() for r in results]},
            message=f"Executed {len(self.skills)} skills"
        )
