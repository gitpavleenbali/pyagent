# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Agent Configuration Schema

Pydantic schemas for validating agent configurations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class OutputFormat(Enum):
    """Output format options."""

    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"


@dataclass
class ToolSchema:
    """Schema for a tool definition.

    Example:
        tool:
          name: get_weather
          description: Get current weather
          parameters:
            city:
              type: string
              required: true
    """

    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    returns: Optional[str] = None
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolSchema":
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            parameters=data.get("parameters", {}),
            returns=data.get("returns"),
            enabled=data.get("enabled", True),
        )


@dataclass
class ModelSchema:
    """Schema for model configuration.

    Example:
        model:
          provider: azure
          model_id: gpt-4o
          temperature: 0.7
          max_tokens: 2000
    """

    provider: str = "auto"
    model_id: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelSchema":
        """Create from dictionary."""
        return cls(
            provider=data.get("provider", "auto"),
            model_id=data.get("model_id") or data.get("model"),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens"),
            top_p=data.get("top_p", 1.0),
            extra=data.get("extra", {}),
        )


@dataclass
class AgentSchema:
    """Schema for complete agent configuration.

    Example YAML:
        name: research_assistant
        description: A helpful research assistant

        instructions: |
          You are a research assistant that helps find information.
          Be thorough and cite sources.

        model:
          provider: azure
          model_id: gpt-4o
          temperature: 0.5

        tools:
          - name: web_search
            description: Search the web
          - name: fetch_url
            description: Fetch webpage content

        guardrails:
          - no_harmful_content
          - verify_sources

        output_format: markdown
    """

    name: str
    description: str = ""
    instructions: str = ""
    model: Optional[ModelSchema] = None
    tools: List[ToolSchema] = field(default_factory=list)
    guardrails: List[str] = field(default_factory=list)
    output_format: OutputFormat = OutputFormat.TEXT
    handoffs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSchema":
        """Create AgentSchema from a dictionary.

        Args:
            data: Dictionary with agent configuration

        Returns:
            AgentSchema instance
        """
        # Parse model config
        model = None
        if "model" in data:
            model_data = data["model"]
            if isinstance(model_data, str):
                # Simple format: model: gpt-4o
                model = ModelSchema(model_id=model_data)
            else:
                model = ModelSchema.from_dict(model_data)

        # Parse tools
        tools = []
        for tool_data in data.get("tools", []):
            if isinstance(tool_data, str):
                tools.append(ToolSchema(name=tool_data))
            else:
                tools.append(ToolSchema.from_dict(tool_data))

        # Parse output format
        output_format = OutputFormat.TEXT
        if "output_format" in data:
            try:
                output_format = OutputFormat(data["output_format"])
            except ValueError:
                pass

        return cls(
            name=data.get("name", "unnamed_agent"),
            description=data.get("description", ""),
            instructions=data.get("instructions", data.get("system_prompt", "")),
            model=model,
            tools=tools,
            guardrails=data.get("guardrails", []),
            output_format=output_format,
            handoffs=data.get("handoffs", []),
            metadata=data.get("metadata", {}),
            version=data.get("version", "1.0"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
            "version": self.version,
        }

        if self.model:
            result["model"] = {
                "provider": self.model.provider,
                "model_id": self.model.model_id,
                "temperature": self.model.temperature,
            }
            if self.model.max_tokens:
                result["model"]["max_tokens"] = self.model.max_tokens

        if self.tools:
            result["tools"] = [{"name": t.name, "description": t.description} for t in self.tools]

        if self.guardrails:
            result["guardrails"] = self.guardrails

        if self.output_format != OutputFormat.TEXT:
            result["output_format"] = self.output_format.value

        if self.handoffs:
            result["handoffs"] = self.handoffs

        if self.metadata:
            result["metadata"] = self.metadata

        return result


def validate_config(config: Dict[str, Any]) -> tuple:
    """Validate an agent configuration dictionary.

    Args:
        config: Agent configuration dictionary

    Returns:
        Tuple of (is_valid, errors)
    """
    errors = []

    # Check required fields
    if "name" not in config:
        errors.append("Missing required field: 'name'")

    # Validate model config
    if "model" in config and isinstance(config["model"], dict):
        model = config["model"]
        if "provider" in model:
            valid_providers = [
                "auto",
                "azure",
                "openai",
                "ollama",
                "anthropic",
                "gemini",
                "litellm",
            ]
            if model["provider"] not in valid_providers:
                errors.append(f"Invalid provider: {model['provider']}")

        if "temperature" in model:
            temp = model["temperature"]
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                errors.append("Temperature must be between 0 and 2")

    # Validate tools
    if "tools" in config:
        for i, tool in enumerate(config["tools"]):
            if isinstance(tool, dict) and "name" not in tool:
                errors.append(f"Tool at index {i} missing 'name' field")

    # Validate output format
    if "output_format" in config:
        valid_formats = ["text", "json", "markdown"]
        if config["output_format"] not in valid_formats:
            errors.append(f"Invalid output_format: {config['output_format']}")

    return len(errors) == 0, errors
