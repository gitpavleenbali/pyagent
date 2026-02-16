"""
Agent - The central orchestrator for AI agent behavior
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import asyncio

from pyagent.core.base import BaseComponent, Executable
from pyagent.core.memory import Memory, ConversationMemory
from pyagent.core.llm import LLMProvider


@dataclass
class AgentConfig:
    """Configuration for an Agent instance"""
    
    max_iterations: int = 10
    timeout_seconds: float = 300.0
    verbose: bool = False
    enable_memory: bool = True
    enable_logging: bool = True
    retry_on_failure: bool = True
    max_retries: int = 3


@dataclass
class AgentResponse:
    """Response from an agent execution"""
    
    content: str
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    skill_results: List[Dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    tokens_used: int = 0


class Agent(BaseComponent, Executable):
    """
    Agent - The core class for building AI agents.
    
    An Agent combines:
    - Instructions: Define behavior and persona
    - Skills: Capabilities the agent can use
    - Memory: Context and conversation history
    - LLM: The language model powering reasoning
    
    Example:
        >>> agent = Agent(
        ...     name="ResearchAgent",
        ...     instructions=Instruction("You are a research assistant"),
        ...     skills=[WebSearchSkill(), SummarySkill()],
        ... )
        >>> response = await agent.run("Find recent papers on transformers")
    """
    
    def __init__(
        self,
        name: str = "Agent",
        instructions: Optional["Instruction"] = None,
        skills: Optional[List["Skill"]] = None,
        llm: Optional[LLMProvider] = None,
        memory: Optional[Memory] = None,
        config: Optional[AgentConfig] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        self.instructions = instructions
        self.skills = skills or []
        self.llm = llm
        self.memory = memory or ConversationMemory()
        self.config = config or AgentConfig()
        
        # Internal state
        self._is_running = False
        self._current_iteration = 0
        self._skill_registry: Dict[str, "Skill"] = {}
        
        # Register skills
        for skill in self.skills:
            self.register_skill(skill)
    
    def register_skill(self, skill: "Skill") -> None:
        """Register a skill with the agent"""
        self._skill_registry[skill.name] = skill
        skill.bind_to_agent(self)
    
    def unregister_skill(self, skill_name: str) -> None:
        """Remove a skill from the agent"""
        if skill_name in self._skill_registry:
            del self._skill_registry[skill_name]
    
    def get_skill(self, name: str) -> Optional["Skill"]:
        """Get a registered skill by name"""
        return self._skill_registry.get(name)
    
    @property
    def available_skills(self) -> List[str]:
        """List of available skill names"""
        return list(self._skill_registry.keys())
    
    def validate(self) -> bool:
        """Validate agent configuration"""
        if not self.instructions:
            return False
        if self.llm is None:
            return False
        return True
    
    async def validate_input(self, message: str, **kwargs) -> bool:
        """Validate input before processing"""
        if not message or not isinstance(message, str):
            return False
        return True
    
    async def execute(self, message: str, **kwargs) -> AgentResponse:
        """Execute the agent with a message"""
        return await self.run(message, **kwargs)
    
    async def run(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResponse:
        """
        Run the agent with a user message.
        
        Args:
            message: The user's input message
            context: Additional context for this run
            **kwargs: Additional arguments
            
        Returns:
            AgentResponse with the result
        """
        if not await self.validate_input(message):
            return AgentResponse(
                content="Invalid input provided",
                success=False,
            )
        
        self._is_running = True
        self._current_iteration = 0
        
        try:
            # Add message to memory
            if self.config.enable_memory:
                self.memory.add_message("user", message)
            
            # Build the prompt with instructions
            system_prompt = self._build_system_prompt()
            
            # Main agent loop
            while self._current_iteration < self.config.max_iterations:
                self._current_iteration += 1
                
                # Get LLM response
                response = await self._get_llm_response(
                    system_prompt=system_prompt,
                    messages=self.memory.get_messages(),
                    context=context,
                )
                
                # Check if any skills should be invoked
                skill_calls = self._parse_skill_calls(response)
                
                if skill_calls:
                    # Execute skills
                    skill_results = await self._execute_skills(skill_calls)
                    
                    # Add skill results to context
                    if self.config.enable_memory:
                        for result in skill_results:
                            self.memory.add_message("tool", str(result))
                else:
                    # No skill calls, we have a final answer
                    if self.config.enable_memory:
                        self.memory.add_message("assistant", response)
                    
                    return AgentResponse(
                        content=response,
                        success=True,
                        iterations=self._current_iteration,
                    )
            
            # Max iterations reached
            return AgentResponse(
                content="Maximum iterations reached",
                success=False,
                iterations=self._current_iteration,
            )
            
        finally:
            self._is_running = False
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt from instructions"""
        if self.instructions is None:
            return "You are a helpful AI assistant."
        return self.instructions.render()
    
    async def _get_llm_response(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Get response from the LLM"""
        if self.llm is None:
            raise ValueError("No LLM provider configured")
        
        return await self.llm.complete(
            system_prompt=system_prompt,
            messages=messages,
            context=context,
        )
    
    def _parse_skill_calls(self, response: str) -> List[Dict[str, Any]]:
        """Parse skill/tool calls from LLM response"""
        # This is a simplified parser - real implementation would be more robust
        skill_calls = []
        # Parse based on the response format
        # Example: <skill name="search">query</skill>
        return skill_calls
    
    async def _execute_skills(
        self,
        skill_calls: List[Dict[str, Any]]
    ) -> List[Any]:
        """Execute parsed skill calls"""
        results = []
        for call in skill_calls:
            skill_name = call.get("name")
            skill = self.get_skill(skill_name)
            if skill:
                result = await skill.execute(**call.get("params", {}))
                results.append(result)
        return results
    
    def __repr__(self) -> str:
        return (
            f"Agent(name='{self.name}', "
            f"skills={len(self.skills)}, "
            f"memory={type(self.memory).__name__})"
        )
