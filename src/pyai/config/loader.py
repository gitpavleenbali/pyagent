# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Agent Configuration Loader

Load agents from YAML/JSON configuration files.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .schema import AgentSchema, validate_config


@dataclass
class AgentConfig:
    """Loaded agent configuration with path information.

    Attributes:
        schema: The parsed agent schema
        source_path: Path to the source configuration file
        raw_config: Raw configuration dictionary
    """

    schema: AgentSchema
    source_path: Optional[Path] = None
    raw_config: Dict[str, Any] = None

    @property
    def name(self) -> str:
        """Get agent name."""
        return self.schema.name

    @property
    def instructions(self) -> str:
        """Get agent instructions."""
        return self.schema.instructions


def load_agent(path: Union[str, Path], validate: bool = True) -> AgentConfig:
    """Load an agent configuration from a file.

    Supports both YAML (.yaml, .yml) and JSON (.json) formats.

    Args:
        path: Path to the configuration file
        validate: Whether to validate the configuration

    Returns:
        AgentConfig object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If validation fails

    Example:
        config = load_agent("agents/research_assistant.yaml")
        print(config.name)
        print(config.instructions)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Agent config not found: {path}")

    # Load file content
    content = path.read_text(encoding="utf-8")

    # Parse based on extension
    ext = path.suffix.lower()
    if ext in (".yaml", ".yml"):
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML config files. "
                "Install with: pip install pyai[yaml] or pip install pyyaml"
            )
        config_dict = yaml.safe_load(content)
    elif ext == ".json":
        config_dict = json.loads(content)
    else:
        # Try to infer format
        try:
            config_dict = json.loads(content)
        except json.JSONDecodeError:
            if YAML_AVAILABLE:
                config_dict = yaml.safe_load(content)
            else:
                raise ValueError(f"Unable to parse config file: {path}")

    # Handle None (empty file)
    if config_dict is None:
        config_dict = {}

    # Validate if requested
    if validate:
        is_valid, errors = validate_config(config_dict)
        if not is_valid:
            raise ValueError(f"Invalid agent config: {', '.join(errors)}")

    # Parse schema
    schema = AgentSchema.from_dict(config_dict)

    return AgentConfig(
        schema=schema,
        source_path=path,
        raw_config=config_dict,
    )


def load_agents_from_dir(
    directory: Union[str, Path], recursive: bool = False, validate: bool = True
) -> List[AgentConfig]:
    """Load all agent configurations from a directory.

    Args:
        directory: Path to directory containing config files
        recursive: Whether to search subdirectories
        validate: Whether to validate configurations

    Returns:
        List of AgentConfig objects

    Example:
        configs = load_agents_from_dir("agents/")
        for config in configs:
            print(f"Found agent: {config.name}")
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    configs = []
    extensions = {".yaml", ".yml", ".json"}

    if recursive:
        files = directory.rglob("*")
    else:
        files = directory.iterdir()

    for file_path in files:
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            try:
                config = load_agent(file_path, validate=validate)
                configs.append(config)
            except (ValueError, json.JSONDecodeError):
                # Skip invalid files but could log warning
                continue

    return configs


def load_agent_from_string(
    content: str, format: str = "yaml", validate: bool = True
) -> AgentConfig:
    """Load agent configuration from a string.

    Args:
        content: Configuration content as string
        format: Format of the content ("yaml" or "json")
        validate: Whether to validate the configuration

    Returns:
        AgentConfig object

    Example:
        config_str = '''
        name: my_agent
        instructions: You are helpful
        '''
        config = load_agent_from_string(config_str)
    """
    if format.lower() in ("yaml", "yml"):
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required for YAML parsing")
        config_dict = yaml.safe_load(content)
    elif format.lower() == "json":
        config_dict = json.loads(content)
    else:
        raise ValueError(f"Unknown format: {format}")

    if config_dict is None:
        config_dict = {}

    if validate:
        is_valid, errors = validate_config(config_dict)
        if not is_valid:
            raise ValueError(f"Invalid agent config: {', '.join(errors)}")

    schema = AgentSchema.from_dict(config_dict)

    return AgentConfig(
        schema=schema,
        source_path=None,
        raw_config=config_dict,
    )


def save_agent_config(
    config: AgentConfig, path: Union[str, Path], format: Optional[str] = None
) -> None:
    """Save an agent configuration to a file.

    Args:
        config: AgentConfig to save
        path: Path to save to
        format: Output format (inferred from extension if not provided)

    Example:
        save_agent_config(config, "my_agent.yaml")
    """
    path = Path(path)

    # Determine format
    if format is None:
        ext = path.suffix.lower()
        if ext in (".yaml", ".yml"):
            format = "yaml"
        else:
            format = "json"

    # Convert to dict
    config_dict = config.schema.to_dict()

    # Write file
    if format == "yaml":
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required for YAML output")
        content = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
    else:
        content = json.dumps(config_dict, indent=2)

    path.write_text(content, encoding="utf-8")
