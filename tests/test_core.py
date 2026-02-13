"""
PyAgent Core Module Unit Tests
==============================

Unit tests for core modules: Agent, Memory, LLM providers.
These tests do NOT require live API connections.
"""

import pytest
from unittest.mock import MagicMock, patch, Mock
from dataclasses import dataclass


class TestAgentFactory:
    """Unit tests for agent factory."""
    
    def test_import_agent_function(self):
        """Test importing agent function."""
        from pyagent.easy.agent_factory import agent
        assert agent is not None
        assert callable(agent)
    
    def test_import_simple_agent_class(self):
        """Test importing SimpleAgent class."""
        from pyagent.easy.agent_factory import SimpleAgent
        assert SimpleAgent is not None
    
    def test_create_agent_with_string(self):
        """Test creating agent with string instructions."""
        from pyagent.easy.agent_factory import agent
        
        my_agent = agent("You are a helpful assistant")
        assert my_agent is not None
        assert hasattr(my_agent, '__call__')
    
    def test_create_agent_with_name(self):
        """Test creating agent with custom name."""
        from pyagent.easy.agent_factory import agent
        
        my_agent = agent("Test instructions", name="TestBot")
        assert my_agent.name == "TestBot"
    
    def test_create_agent_with_persona(self):
        """Test creating agent with persona."""
        from pyagent.easy.agent_factory import agent
        
        my_agent = agent(persona="researcher")
        assert my_agent is not None
        # Should have research-related instructions
        assert "research" in my_agent.instructions.lower() or my_agent.instructions != ""
    
    def test_agent_has_history(self):
        """Test that agent has history property."""
        from pyagent.easy.agent_factory import agent
        
        my_agent = agent("Test")
        assert hasattr(my_agent, 'history')
        assert isinstance(my_agent.history, list)
    
    def test_agent_reset_memory(self):
        """Test agent reset method."""
        from pyagent.easy.agent_factory import agent
        
        my_agent = agent("Test", memory=True)
        # Agent starts with system message internally
        assert hasattr(my_agent, 'reset')
        my_agent.reset()
        # After reset, history (excluding system) should be empty
        assert len(my_agent.history) == 0


class TestSimpleAgentClass:
    """Unit tests for SimpleAgent class."""
    
    def test_simple_agent_init(self):
        """Test SimpleAgent initialization."""
        from pyagent.easy.agent_factory import SimpleAgent
        
        agent = SimpleAgent(instructions="Test instructions")
        assert agent.instructions == "Test instructions"
    
    def test_simple_agent_default_name(self):
        """Test SimpleAgent default name."""
        from pyagent.easy.agent_factory import SimpleAgent
        
        agent = SimpleAgent(instructions="Test")
        assert agent.name == "Agent"
    
    def test_simple_agent_custom_name(self):
        """Test SimpleAgent custom name."""
        from pyagent.easy.agent_factory import SimpleAgent
        
        agent = SimpleAgent(instructions="Test", name="CustomBot")
        assert agent.name == "CustomBot"
    
    def test_simple_agent_repr(self):
        """Test SimpleAgent string representation."""
        from pyagent.easy.agent_factory import SimpleAgent
        
        agent = SimpleAgent(instructions="Test", name="TestBot")
        repr_str = repr(agent)
        assert "TestBot" in repr_str
        assert "Agent" in repr_str


class TestMemory:
    """Unit tests for memory module."""
    
    def test_import_memory(self):
        """Test importing memory module."""
        from pyagent.core import memory
        assert memory is not None
    
    def test_memory_class_exists(self):
        """Test Memory base class exists."""
        from pyagent.core.memory import Memory
        assert Memory is not None
    
    def test_conversation_memory_exists(self):
        """Test ConversationMemory class exists."""
        from pyagent.core.memory import ConversationMemory
        assert ConversationMemory is not None
    
    def test_vector_memory_exists(self):
        """Test VectorMemory class exists."""
        from pyagent.core.memory import VectorMemory
        assert VectorMemory is not None


class TestLLMInterface:
    """Unit tests for LLM interface."""
    
    def test_import_llm_interface(self):
        """Test importing LLM interface."""
        from pyagent.easy.llm_interface import LLMInterface
        assert LLMInterface is not None
    
    def test_llm_interface_has_chat(self):
        """Test LLMInterface has chat method."""
        from pyagent.easy.llm_interface import LLMInterface
        
        llm = LLMInterface()
        assert hasattr(llm, 'chat')
        assert callable(llm.chat)
    
    def test_llm_interface_has_complete(self):
        """Test LLMInterface has complete method."""
        from pyagent.easy.llm_interface import LLMInterface
        
        llm = LLMInterface()
        assert hasattr(llm, 'complete')
        assert callable(llm.complete)


class TestConfig:
    """Unit tests for configuration."""
    
    def test_import_configure(self):
        """Test importing configure function."""
        from pyagent.easy.config import configure
        assert configure is not None
        assert callable(configure)
    
    def test_import_get_config(self):
        """Test importing get_config function."""
        from pyagent.easy.config import get_config
        assert get_config is not None
        assert callable(get_config)
    
    def test_config_properties(self):
        """Test PyAgentConfig has expected properties."""
        from pyagent.easy.config import PyAgentConfig
        
        cfg = PyAgentConfig()
        assert hasattr(cfg, 'api_key')
        assert hasattr(cfg, 'model')
        assert hasattr(cfg, 'provider')
        assert hasattr(cfg, 'azure_endpoint')
    
    def test_configure_sets_values(self):
        """Test configure function sets values."""
        from pyagent.easy.config import configure, get_config, reset_config
        
        # Save original state
        reset_config()
        
        # Configure with test values
        configure(model="test-model", temperature=0.5)
        
        cfg = get_config()
        assert cfg.model == "test-model"
        assert cfg.temperature == 0.5
        
        # Cleanup
        reset_config()


class TestSkills:
    """Unit tests for skills module."""
    
    def test_import_skill_base(self):
        """Test importing Skill base class."""
        from pyagent.skills import Skill
        assert Skill is not None
    
    def test_import_tool_skill(self):
        """Test importing ToolSkill."""
        from pyagent.skills import ToolSkill
        assert ToolSkill is not None
    
    def test_import_action_skill(self):
        """Test importing ActionSkill."""
        from pyagent.skills import ActionSkill
        assert ActionSkill is not None
    
    def test_import_skill_registry(self):
        """Test importing SkillRegistry."""
        from pyagent.skills import SkillRegistry
        assert SkillRegistry is not None
    
    def test_skill_registry_methods(self):
        """Test SkillRegistry has required methods."""
        from pyagent.skills import SkillRegistry
        
        registry = SkillRegistry()
        assert hasattr(registry, 'register')
        assert hasattr(registry, 'get')
        assert hasattr(registry, 'find_by_tag')
        assert hasattr(registry, 'list_skills')


class TestInstructions:
    """Unit tests for instructions module."""
    
    def test_import_instruction(self):
        """Test importing Instruction."""
        from pyagent.instructions import Instruction
        assert Instruction is not None
    
    def test_import_system_prompt(self):
        """Test importing SystemPrompt."""
        from pyagent.instructions import SystemPrompt
        assert SystemPrompt is not None
    
    def test_import_context(self):
        """Test importing Context."""
        from pyagent.instructions import Context
        assert Context is not None
    
    def test_import_persona(self):
        """Test importing Persona."""
        from pyagent.instructions import Persona
        assert Persona is not None
    
    def test_import_guidelines(self):
        """Test importing Guidelines."""
        from pyagent.instructions import Guidelines
        assert Guidelines is not None


class TestBlueprint:
    """Unit tests for blueprint module."""
    
    def test_import_blueprint(self):
        """Test importing Blueprint."""
        from pyagent.blueprint import Blueprint
        assert Blueprint is not None
    
    def test_import_workflow(self):
        """Test importing Workflow."""
        from pyagent.blueprint import Workflow
        assert Workflow is not None
    
    def test_import_pipeline(self):
        """Test importing Pipeline."""
        from pyagent.blueprint import Pipeline
        assert Pipeline is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
