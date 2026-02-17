# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Agent Builder

Build agent instances from configuration.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Union

from .loader import AgentConfig, load_agent
from .schema import AgentSchema


class AgentBuilder:
    """Build agents from configuration.

    Creates fully configured agent instances from AgentConfig or YAML files.

    Example:
        # From file
        agent = AgentBuilder.from_file("agents/research.yaml")
        result = agent.run("Research quantum computing")

        # From config
        config = load_agent("agents/research.yaml")
        agent = AgentBuilder.build(config)

        # With custom tools
        builder = AgentBuilder()
        builder.register_tool("web_search", my_search_function)
        agent = builder.from_file("agents/research.yaml")
    """

    def __init__(self):
        """Initialize builder with tool registry."""
        self._tool_registry: Dict[str, Callable] = {}
        self._guardrail_registry: Dict[str, Callable] = {}

    def register_tool(self, name: str, func: Callable) -> "AgentBuilder":
        """Register a tool function.

        Args:
            name: Tool name (matches name in config)
            func: Tool function

        Returns:
            Self for chaining
        """
        self._tool_registry[name] = func
        return self

    def register_guardrail(self, name: str, func: Callable) -> "AgentBuilder":
        """Register a guardrail function.

        Args:
            name: Guardrail name
            func: Guardrail function

        Returns:
            Self for chaining
        """
        self._guardrail_registry[name] = func
        return self

    def build(self, config: AgentConfig) -> Any:
        """Build an agent from configuration.

        Args:
            config: Agent configuration

        Returns:
            Configured agent instance
        """
        from pyai.core.agent import Agent

        schema = config.schema

        # Create base agent
        agent = Agent(
            name=schema.name,
            instructions=schema.instructions,
        )

        # Configure model if specified
        if schema.model:
            model_config = {
                "provider": schema.model.provider,
                "model_id": schema.model.model_id,
                "temperature": schema.model.temperature,
            }
            if schema.model.max_tokens:
                model_config["max_tokens"] = schema.model.max_tokens
            agent._model_config = model_config

        # Add tools
        for tool_schema in schema.tools:
            if tool_schema.name in self._tool_registry:
                func = self._tool_registry[tool_schema.name]
                agent.add_tool(func, name=tool_schema.name)

        # Add guardrails (if registered)
        for guardrail_name in schema.guardrails:
            if guardrail_name in self._guardrail_registry:
                agent._guardrails.append(self._guardrail_registry[guardrail_name])

        # Set metadata
        agent._metadata = schema.metadata
        agent._description = schema.description

        return agent

    def from_file(self, path: Union[str, Path]) -> Any:
        """Build agent from configuration file.

        Args:
            path: Path to YAML/JSON config file

        Returns:
            Configured agent instance

        Example:
            agent = AgentBuilder().from_file("research_agent.yaml")
        """
        config = load_agent(path)
        return self.build(config)

    @classmethod
    def quick_build(cls, path: Union[str, Path]) -> Any:
        """Quick method to build agent from file.

        Args:
            path: Path to config file

        Returns:
            Agent instance

        Example:
            agent = AgentBuilder.quick_build("agent.yaml")
        """
        builder = cls()
        return builder.from_file(path)


# Convenience function
def agent_from_yaml(path: Union[str, Path]) -> Any:
    """Create an agent from a YAML configuration file.

    This is a convenience function for quickly loading agents.

    Args:
        path: Path to YAML file

    Returns:
        Configured agent

    Example:
        from pyai.config import agent_from_yaml

        agent = agent_from_yaml("agents/assistant.yaml")
        result = agent.run("Hello!")
    """
    return AgentBuilder.quick_build(path)


def agent_from_config(config: Dict[str, Any]) -> Any:
    """Create an agent from a configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        Configured agent

    Example:
        agent = agent_from_config({
            "name": "helper",
            "instructions": "You are helpful",
            "model": {"provider": "azure", "model_id": "gpt-4o"}
        })
    """

    schema = AgentSchema.from_dict(config)
    agent_config = AgentConfig(schema=schema, raw_config=config)

    builder = AgentBuilder()
    return builder.build(agent_config)
