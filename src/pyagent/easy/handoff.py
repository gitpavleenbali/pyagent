"""
handoff() - Multi-agent handoffs in one line

Seamlessly transfer conversations between specialized agents.
Inspired by OpenAI Agents SDK's handoff pattern but simplified to PyAgent's philosophy.

Examples:
    >>> from pyagent import handoff, agent
    
    # Simple handoff
    >>> support = agent("customer support")
    >>> billing = agent("billing specialist")
    >>> result = handoff(support, billing, "I need help with my invoice")
    
    # Automatic routing
    >>> team = handoff.team([
    ...     agent(persona="coder", name="CodeBot"),
    ...     agent(persona="writer", name="WriteBot"),
    ...     agent(persona="analyst", name="DataBot"),
    ... ])
    >>> result = team.route("Write a Python function to analyze data")
    # Automatically routes to best agent(s)
    
    # Chain handoffs
    >>> result = handoff.chain([researcher, analyst, writer], topic="AI trends")
"""

from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
import json

from pyagent.easy.llm_interface import get_llm
from pyagent.easy.agent_factory import SimpleAgent, agent

# Type alias for clarity
Agent = SimpleAgent


@dataclass
class HandoffResult:
    """Result of a handoff operation."""
    
    final_output: str
    agents_used: List[str]
    handoff_log: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = True
    
    def __str__(self) -> str:
        return self.final_output
    
    def __repr__(self) -> str:
        return f"HandoffResult(agents={self.agents_used}, success={self.success})"


@dataclass  
class AgentTeam:
    """A team of agents that can route to the best one."""
    
    agents: List[Agent]
    router_prompt: str = None
    
    def __post_init__(self):
        if not self.router_prompt:
            agent_list = "\n".join(
                f"- {a.name}: {a.instructions[:100]}..." 
                for a in self.agents
            )
            self.router_prompt = f"""You are a routing agent. Given a user request, 
determine which agent(s) should handle it.

Available agents:
{agent_list}

Return a JSON object with:
- "primary": name of the primary agent to handle this
- "backup": name of backup agent (optional)
- "reason": brief explanation of routing decision
"""
    
    def route(
        self,
        message: str,
        *,
        allow_multi: bool = False,
        model: str = None
    ) -> HandoffResult:
        """
        Route a message to the best agent(s).
        
        Args:
            message: The user message to route
            allow_multi: Allow multiple agents to collaborate
            model: Override default model
        """
        llm = get_llm(model=model) if model else get_llm()
        
        # Get routing decision
        routing = llm.json(
            f"Route this request: {message}",
            system=self.router_prompt,
            temperature=0.1
        )
        
        primary_name = routing.get("primary", self.agents[0].name)
        
        # Find the agent
        primary_agent = None
        for a in self.agents:
            if a.name.lower() == primary_name.lower():
                primary_agent = a
                break
        
        if not primary_agent:
            primary_agent = self.agents[0]
        
        # Execute
        result = primary_agent(message)
        
        return HandoffResult(
            final_output=result,
            agents_used=[primary_agent.name],
            handoff_log=[{
                "type": "routing",
                "decision": routing,
                "agent": primary_agent.name,
                "output": result
            }]
        )
    
    def collaborate(
        self,
        message: str,
        *,
        strategy: str = "sequential",  # "sequential", "parallel", "vote"
        model: str = None
    ) -> HandoffResult:
        """
        Have all agents collaborate on a task.
        
        Args:
            message: The task to collaborate on
            strategy: How to combine results
            model: Override default model
        """
        llm = get_llm(model=model) if model else get_llm()
        
        results = []
        for agent in self.agents:
            response = agent(message)
            results.append({
                "agent": agent.name,
                "response": response
            })
        
        # Synthesize results
        synthesis_prompt = f"""Multiple experts have responded to: "{message}"

Responses:
{json.dumps(results, indent=2)}

Synthesize these into a single, comprehensive response that combines the best insights."""
        
        final = llm.complete(synthesis_prompt, temperature=0.3)
        
        return HandoffResult(
            final_output=final,
            agents_used=[a.name for a in self.agents],
            handoff_log=results
        )


def handoff(
    from_agent: Agent,
    to_agent: Agent,
    message: str,
    *,
    context: str = None,
    reason: str = None,
    model: str = None
) -> HandoffResult:
    """
    Hand off a conversation from one agent to another.
    
    Args:
        from_agent: The originating agent
        to_agent: The agent to hand off to
        message: The message/task to hand off
        context: Additional context to pass
        reason: Reason for the handoff
        model: Override default model
        
    Returns:
        HandoffResult with the final output
        
    Examples:
        >>> support = agent("customer support specialist")
        >>> billing = agent("billing expert") 
        >>> result = handoff(support, billing, "Customer needs invoice help")
    """
    llm = get_llm(model=model) if model else get_llm()
    
    # First agent processes and prepares handoff
    handoff_prep = f"""You are handing off this conversation to another agent.
    
Original request: {message}
{f"Additional context: {context}" if context else ""}
{f"Reason for handoff: {reason}" if reason else ""}

Summarize the key information the next agent needs to know."""

    from_response = from_agent(handoff_prep)
    
    # Second agent receives and processes
    handoff_receive = f"""You are receiving a handoff from {from_agent.name}.

Handoff summary: {from_response}

Original request: {message}

Please help the user with their request."""

    to_response = to_agent(handoff_receive)
    
    return HandoffResult(
        final_output=to_response,
        agents_used=[from_agent.name, to_agent.name],
        handoff_log=[
            {"agent": from_agent.name, "action": "prepare_handoff", "output": from_response},
            {"agent": to_agent.name, "action": "receive_handoff", "output": to_response}
        ]
    )


def team(agents: List[Agent], *, router_prompt: str = None) -> AgentTeam:
    """
    Create a team of agents with automatic routing.
    
    Args:
        agents: List of agents in the team
        router_prompt: Custom routing prompt
        
    Returns:
        AgentTeam that can route and collaborate
        
    Examples:
        >>> my_team = handoff.team([
        ...     agent(persona="coder"),
        ...     agent(persona="writer"),
        ...     agent(persona="researcher")
        ... ])
        >>> result = my_team.route("Write code to analyze trends")
    """
    return AgentTeam(agents=agents, router_prompt=router_prompt)


def chain(
    agents: List[Agent],
    task: str,
    *,
    pass_output: bool = True,
    model: str = None
) -> HandoffResult:
    """
    Chain multiple agents in sequence, passing output from one to the next.
    
    Args:
        agents: List of agents to chain
        task: Initial task
        pass_output: Whether to pass each agent's output to the next
        model: Override default model
        
    Returns:
        HandoffResult with final output
        
    Examples:
        >>> researcher = agent(persona="researcher")
        >>> analyst = agent(persona="analyst")
        >>> writer = agent(persona="writer")
        >>> result = handoff.chain(
        ...     [researcher, analyst, writer],
        ...     task="Create a report on AI trends"
        ... )
    """
    current_input = task
    handoff_log = []
    
    for i, agent in enumerate(agents):
        if i == 0:
            prompt = current_input
        else:
            prompt = f"""Previous agent output:
{current_input}

Continue building on this for the task: {task}"""
        
        response = agent(prompt)
        handoff_log.append({
            "step": i + 1,
            "agent": agent.name,
            "input": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "output": response
        })
        
        if pass_output:
            current_input = response
    
    return HandoffResult(
        final_output=current_input,
        agents_used=[a.name for a in agents],
        handoff_log=handoff_log
    )


# Attach functions to handoff for convenience
handoff.team = team
handoff.chain = chain
