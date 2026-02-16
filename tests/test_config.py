# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Tests for the config module.

Tests agent configuration loading including:
- YAML/JSON parsing
- Schema validation
- Agent building
"""

import pytest
import tempfile
import os
from pathlib import Path


class TestAgentSchema:
    """Tests for AgentSchema."""
    
    def test_schema_import(self):
        """Test that AgentSchema can be imported."""
        from pyagent.config import AgentSchema
        assert AgentSchema is not None
    
    def test_schema_from_dict_minimal(self):
        """Test parsing minimal config."""
        from pyagent.config.schema import AgentSchema
        
        config = {"name": "test_agent"}
        schema = AgentSchema.from_dict(config)
        
        assert schema.name == "test_agent"
        assert schema.description == ""
        assert schema.instructions == ""
    
    def test_schema_from_dict_full(self):
        """Test parsing full config."""
        from pyagent.config.schema import AgentSchema
        
        config = {
            "name": "research_assistant",
            "description": "A research helper",
            "instructions": "You help with research",
            "model": {
                "provider": "azure",
                "model_id": "gpt-4o",
                "temperature": 0.5,
            },
            "tools": [
                {"name": "web_search", "description": "Search the web"},
                "fetch_url",  # Simple string format
            ],
            "guardrails": ["no_harmful_content"],
            "output_format": "markdown",
        }
        
        schema = AgentSchema.from_dict(config)
        
        assert schema.name == "research_assistant"
        assert schema.description == "A research helper"
        assert schema.model.provider == "azure"
        assert schema.model.model_id == "gpt-4o"
        assert len(schema.tools) == 2
        assert schema.tools[0].name == "web_search"
        assert schema.tools[1].name == "fetch_url"
        assert "no_harmful_content" in schema.guardrails
    
    def test_schema_to_dict(self):
        """Test converting schema to dict."""
        from pyagent.config.schema import AgentSchema, ModelSchema
        
        schema = AgentSchema(
            name="test",
            description="Test agent",
            instructions="Be helpful",
            model=ModelSchema(provider="azure", model_id="gpt-4o"),
        )
        
        d = schema.to_dict()
        
        assert d["name"] == "test"
        assert d["description"] == "Test agent"
        assert d["model"]["provider"] == "azure"


class TestToolSchema:
    """Tests for ToolSchema."""
    
    def test_tool_schema_import(self):
        """Test that ToolSchema can be imported."""
        from pyagent.config import ToolSchema
        assert ToolSchema is not None
    
    def test_tool_from_dict(self):
        """Test creating tool from dict."""
        from pyagent.config.schema import ToolSchema
        
        data = {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "city": {"type": "string", "required": True}
            },
        }
        
        tool = ToolSchema.from_dict(data)
        
        assert tool.name == "get_weather"
        assert "city" in tool.parameters


class TestValidateConfig:
    """Tests for validate_config function."""
    
    def test_validate_valid_config(self):
        """Test validating a valid config."""
        from pyagent.config.schema import validate_config
        
        config = {
            "name": "test",
            "model": {"provider": "azure", "temperature": 0.7}
        }
        
        is_valid, errors = validate_config(config)
        assert is_valid == True
        assert len(errors) == 0
    
    def test_validate_missing_name(self):
        """Test validation catches missing name."""
        from pyagent.config.schema import validate_config
        
        config = {"instructions": "Be helpful"}
        
        is_valid, errors = validate_config(config)
        assert is_valid == False
        assert any("name" in e.lower() for e in errors)
    
    def test_validate_invalid_temperature(self):
        """Test validation catches invalid temperature."""
        from pyagent.config.schema import validate_config
        
        config = {
            "name": "test",
            "model": {"temperature": 5.0}  # Invalid
        }
        
        is_valid, errors = validate_config(config)
        assert is_valid == False


class TestAgentConfig:
    """Tests for AgentConfig."""
    
    def test_agentconfig_import(self):
        """Test that AgentConfig can be imported."""
        from pyagent.config import AgentConfig
        assert AgentConfig is not None
    
    def test_agentconfig_properties(self):
        """Test AgentConfig properties."""
        from pyagent.config.loader import AgentConfig
        from pyagent.config.schema import AgentSchema
        
        schema = AgentSchema(
            name="test",
            instructions="Be helpful"
        )
        config = AgentConfig(schema=schema)
        
        assert config.name == "test"
        assert config.instructions == "Be helpful"


class TestLoadAgent:
    """Tests for load_agent function."""
    
    def test_load_agent_import(self):
        """Test that load_agent can be imported."""
        from pyagent.config import load_agent
        assert load_agent is not None
    
    def test_load_agent_yaml(self):
        """Test loading agent from YAML file."""
        from pyagent.config import load_agent
        
        yaml_content = """
name: test_agent
description: A test agent
instructions: |
  You are a helpful assistant.
  Be concise.

model:
  provider: azure
  model_id: gpt-4o
  temperature: 0.7

tools:
  - name: search
    description: Search the web
"""
        
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            config = load_agent(temp_path)
            
            assert config.name == "test_agent"
            assert "helpful" in config.instructions
            assert config.schema.model.provider == "azure"
            assert len(config.schema.tools) == 1
        finally:
            os.unlink(temp_path)
    
    def test_load_agent_json(self):
        """Test loading agent from JSON file."""
        from pyagent.config import load_agent
        import json
        
        json_content = {
            "name": "json_agent",
            "instructions": "Be helpful",
            "model": {"provider": "openai", "model_id": "gpt-4"}
        }
        
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(json_content, f)
            temp_path = f.name
        
        try:
            config = load_agent(temp_path)
            assert config.name == "json_agent"
            assert config.schema.model.provider == "openai"
        finally:
            os.unlink(temp_path)
    
    def test_load_agent_not_found(self):
        """Test loading non-existent file raises error."""
        from pyagent.config import load_agent
        
        with pytest.raises(FileNotFoundError):
            load_agent("/nonexistent/path/agent.yaml")


class TestLoadAgentsFromDir:
    """Tests for load_agents_from_dir function."""
    
    def test_load_from_dir(self):
        """Test loading multiple agents from directory."""
        from pyagent.config.loader import load_agents_from_dir
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two agent files
            agent1 = Path(tmpdir) / "agent1.yaml"
            agent1.write_text("name: agent1\ninstructions: Hello")
            
            agent2 = Path(tmpdir) / "agent2.yaml"
            agent2.write_text("name: agent2\ninstructions: World")
            
            configs = load_agents_from_dir(tmpdir)
            
            assert len(configs) == 2
            names = {c.name for c in configs}
            assert "agent1" in names
            assert "agent2" in names


class TestAgentBuilder:
    """Tests for AgentBuilder."""
    
    def test_builder_import(self):
        """Test that AgentBuilder can be imported."""
        from pyagent.config import AgentBuilder
        assert AgentBuilder is not None
    
    def test_builder_creation(self):
        """Test creating a builder."""
        from pyagent.config import AgentBuilder
        
        builder = AgentBuilder()
        assert builder is not None
    
    def test_builder_register_tool(self):
        """Test registering tools."""
        from pyagent.config import AgentBuilder
        
        def my_tool(x):
            return f"Result: {x}"
        
        builder = AgentBuilder()
        builder.register_tool("my_tool", my_tool)
        
        assert "my_tool" in builder._tool_registry
    
    def test_builder_build_agent(self):
        """Test building agent from config."""
        from pyagent.config import AgentBuilder
        from pyagent.config.loader import AgentConfig
        from pyagent.config.schema import AgentSchema
        
        schema = AgentSchema(
            name="built_agent",
            description="A built agent",
            instructions="You are helpful"
        )
        config = AgentConfig(schema=schema)
        
        builder = AgentBuilder()
        agent = builder.build(config)
        
        assert agent is not None
        assert agent.name == "built_agent"
    
    def test_builder_from_file(self):
        """Test building agent from file."""
        from pyagent.config import AgentBuilder
        
        yaml_content = """
name: file_agent
instructions: You help with tasks
"""
        
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            builder = AgentBuilder()
            agent = builder.from_file(temp_path)
            
            assert agent.name == "file_agent"
        finally:
            os.unlink(temp_path)


class TestConfigIntegration:
    """Integration tests for config module."""
    
    def test_module_exports(self):
        """Test that all expected exports are available."""
        from pyagent import config
        
        assert hasattr(config, "AgentConfig")
        assert hasattr(config, "AgentSchema")
        assert hasattr(config, "ToolSchema")
        assert hasattr(config, "load_agent")
        assert hasattr(config, "AgentBuilder")
        assert hasattr(config, "validate_config")
    
    def test_main_init_exports(self):
        """Test that config is exported from main pyagent."""
        import pyagent
        
        assert hasattr(pyagent, "config")
        assert hasattr(pyagent, "load_agent")
        assert hasattr(pyagent, "AgentConfig")
        assert hasattr(pyagent, "AgentBuilder")
    
    def test_full_workflow(self):
        """Test complete config workflow."""
        from pyagent.config import load_agent, AgentBuilder
        
        yaml_content = """
name: workflow_test
description: Test agent for workflow
instructions: |
  You are a helpful research assistant.
  Always cite sources.

model:
  provider: azure
  model_id: gpt-4o
  temperature: 0.7
  max_tokens: 2000

tools:
  - name: web_search
    description: Search the web for information
  - name: summarize
    description: Summarize text content

guardrails:
  - verify_sources
  - no_harmful_content

output_format: markdown
metadata:
  version: "2.0"
  author: "test"
"""
        
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            # Load config
            config = load_agent(temp_path)
            
            # Verify parsed correctly
            assert config.name == "workflow_test"
            assert config.schema.model.max_tokens == 2000
            assert len(config.schema.tools) == 2
            assert len(config.schema.guardrails) == 2
            
            # Build agent
            builder = AgentBuilder()
            agent = builder.build(config)
            
            assert agent.name == "workflow_test"
        finally:
            os.unlink(temp_path)
