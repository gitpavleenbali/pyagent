"""# pyright: reportMissingImports=falsepyai Smoke Tests
===================

Quick sanity checks to verify basic functionality.
These tests should run fast and NOT require any API connections.

Run: pytest tests/test_smoke.py -v
"""

import pytest
import sys
import importlib


class TestImports:
    """Smoke tests for all module imports."""
    
    def test_import_pyai(self):
        """Test importing main pyai package."""
        import pyai
        assert pyai is not None
        assert hasattr(pyai, '__version__')
    
    def test_import_easy_module(self):
        """Test importing easy module."""
        from pyai import easy
        assert easy is not None
    
    def test_import_core_module(self):
        """Test importing core module."""
        from pyai import core
        assert core is not None
    
    def test_import_skills_module(self):
        """Test importing skills module."""
        from pyai import skills
        assert skills is not None
    
    def test_import_blueprint_module(self):
        """Test importing blueprint module."""
        from pyai import blueprint
        assert blueprint is not None
    
    def test_import_instructions_module(self):
        """Test importing instructions module."""
        from pyai import instructions
        assert instructions is not None
    
    def test_import_integrations_module(self):
        """Test importing integrations module."""
        from pyai import integrations
        assert integrations is not None
    
    def test_import_orchestrator_module(self):
        """Test importing orchestrator module."""
        from pyai import orchestrator
        assert orchestrator is not None
    
    def test_import_usecases_module(self):
        """Test importing usecases module."""
        from pyai import usecases
        assert usecases is not None


class TestLazyImports:
    """Smoke tests for lazy-loaded functions."""
    
    def test_lazy_import_ask(self):
        """Test lazy importing ask function."""
        from pyai import ask
        assert ask is not None
        assert callable(ask)
    
    def test_lazy_import_agent(self):
        """Test lazy importing agent function."""
        from pyai import agent
        assert agent is not None
        assert callable(agent)
    
    def test_lazy_import_research(self):
        """Test lazy importing research function."""
        from pyai import research
        assert research is not None
        assert callable(research)
    
    def test_lazy_import_summarize(self):
        """Test lazy importing summarize function."""
        from pyai import summarize
        assert summarize is not None
        assert callable(summarize)
    
    def test_lazy_import_extract(self):
        """Test lazy importing extract function."""
        from pyai import extract
        assert extract is not None
        assert callable(extract)
    
    def test_lazy_import_generate(self):
        """Test lazy importing generate function."""
        from pyai import generate
        assert generate is not None
        assert callable(generate)
    
    def test_lazy_import_translate(self):
        """Test lazy importing translate function."""
        from pyai import translate
        assert translate is not None
        assert callable(translate)
    
    def test_lazy_import_chat(self):
        """Test lazy importing chat function."""
        from pyai import chat
        assert chat is not None
        assert callable(chat)
    
    def test_lazy_import_code(self):
        """Test lazy importing code module."""
        from pyai import code
        assert code is not None
    
    def test_lazy_import_rag(self):
        """Test lazy importing rag module."""
        from pyai import rag
        assert rag is not None
    
    def test_lazy_import_fetch(self):
        """Test lazy importing fetch module."""
        from pyai import fetch
        assert fetch is not None
    
    def test_lazy_import_analyze(self):
        """Test lazy importing analyze module."""
        from pyai import analyze
        assert analyze is not None
    
    def test_lazy_import_handoff(self):
        """Test lazy importing handoff."""
        from pyai import handoff
        assert handoff is not None
        assert callable(handoff)
    
    def test_lazy_import_mcp(self):
        """Test lazy importing mcp."""
        from pyai import mcp
        assert mcp is not None
    
    def test_lazy_import_guardrails(self):
        """Test lazy importing guardrails."""
        from pyai import guardrails
        assert guardrails is not None
    
    def test_lazy_import_trace(self):
        """Test lazy importing trace."""
        from pyai import trace
        assert trace is not None


class TestBasicFunctionality:
    """Smoke tests for basic functionality without API calls."""
    
    def test_create_agent_no_call(self):
        """Test creating an agent without calling it."""
        from pyai import agent
        
        my_agent = agent("You are a test agent")
        assert my_agent is not None
        assert hasattr(my_agent, 'instructions')
        assert hasattr(my_agent, '__call__')
    
    def test_create_multiple_agents(self):
        """Test creating multiple agents."""
        from pyai import agent
        
        agent1 = agent("First agent")
        agent2 = agent("Second agent")
        agent3 = agent("Third agent")
        
        assert agent1 is not agent2
        assert agent2 is not agent3
    
    def test_mcp_tool_decorator(self):
        """Test MCP tool decorator."""
        from pyai import mcp
        
        @mcp.tool("test_tool")
        def my_tool(x: int) -> int:
            """Test tool."""
            return x * 2
        
        assert my_tool is not None
        assert my_tool.name == "test_tool"
    
    def test_mcp_server_creation(self):
        """Test MCP server creation."""
        from pyai import mcp
        
        @mcp.tool("adder")
        def add(a: int, b: int) -> int:
            """Add numbers."""
            return a + b
        
        server = mcp.server("test", tools=[add])
        assert server is not None
        assert server.name == "test"
    
    def test_guardrails_validators_exist(self):
        """Test guardrails validators exist."""
        from pyai import guardrails
        
        assert hasattr(guardrails, 'no_pii')
        assert hasattr(guardrails, 'no_injection')
        assert hasattr(guardrails, 'redact_pii')
    
    def test_trace_enable_disable(self):
        """Test trace enable/disable."""
        from pyai import trace
        
        trace.enable()
        assert trace.enabled == True
        
        trace.disable()
        assert trace.enabled == False
    
    def test_trace_span_creation(self):
        """Test trace span creation."""
        from pyai import trace
        
        trace.enable()
        
        with trace.span("test_span") as span:
            span.log("test event")
        
        spans = trace.get_spans()
        assert len(spans) > 0
        
        trace.clear()
        trace.disable()
    
    def test_handoff_team_creation(self):
        """Test handoff team creation."""
        from pyai import handoff, agent
        
        a1 = agent("Agent 1")
        a2 = agent("Agent 2")
        
        team = handoff.team([a1, a2])
        assert team is not None
        assert hasattr(team, 'route')
        assert hasattr(team, 'agents')
        assert len(team.agents) == 2


class TestIntegrationsSmoke:
    """Smoke tests for integrations module."""
    
    def test_langchain_adapter_functions(self):
        """Test LangChain adapter has expected functions."""
        from pyai.integrations import langchain
        
        assert hasattr(langchain, 'import_tool')
        assert hasattr(langchain, 'import_chain')
        assert hasattr(langchain, 'export_agent')
        assert hasattr(langchain, 'import_retriever')
    
    def test_semantic_kernel_adapter_functions(self):
        """Test Semantic Kernel adapter has expected functions."""
        from pyai.integrations import semantic_kernel
        
        assert hasattr(semantic_kernel, 'create_kernel')
        assert hasattr(semantic_kernel, 'import_plugin')
        assert hasattr(semantic_kernel, 'create_plan')
        assert hasattr(semantic_kernel, 'execute_plan')
        assert hasattr(semantic_kernel, 'create_memory')
    
    def test_vector_db_connect_function(self):
        """Test vector_db has connect function."""
        from pyai.integrations import vector_db
        
        assert hasattr(vector_db, 'connect')
        assert callable(vector_db.connect)
    
    def test_vector_db_store_classes(self):
        """Test vector_db has store classes."""
        from pyai.integrations import vector_db
        
        assert hasattr(vector_db, 'VectorStore')
        assert hasattr(vector_db, 'AzureAISearchStore')
        assert hasattr(vector_db, 'PineconeStore')
        assert hasattr(vector_db, 'ChromaStore')
        assert hasattr(vector_db, 'FAISSStore')
        assert hasattr(vector_db, 'QdrantStore')


class TestOrchestratorSmoke:
    """Smoke tests for orchestrator module."""
    
    def test_orchestrator_classes(self):
        """Test orchestrator has expected classes."""
        from pyai.orchestrator import (
            Orchestrator, Task, Workflow, ScheduledJob,
            TaskStatus, ExecutionPattern, AgentPatterns
        )
        
        assert Orchestrator is not None
        assert Task is not None
        assert Workflow is not None
        assert ScheduledJob is not None
        assert TaskStatus is not None
        assert ExecutionPattern is not None
        assert AgentPatterns is not None
    
    def test_create_task(self):
        """Test creating a task."""
        from pyai.orchestrator import Task, TaskStatus
        
        task = Task(name="Test Task")
        assert task.name == "Test Task"
        assert task.status == TaskStatus.PENDING
    
    def test_create_workflow(self):
        """Test creating a workflow."""
        from pyai.orchestrator import Workflow, Task
        
        task = Task(name="Step 1")
        workflow = Workflow(name="Test Workflow", steps=[task])
        
        assert workflow.name == "Test Workflow"
        assert len(workflow.steps) == 1
    
    def test_create_orchestrator(self):
        """Test creating an orchestrator."""
        from pyai.orchestrator import Orchestrator
        
        orch = Orchestrator()
        assert orch is not None
        assert hasattr(orch, 'submit')
        assert hasattr(orch, 'schedule')
        assert hasattr(orch, 'status')
    
    def test_agent_patterns_methods(self):
        """Test AgentPatterns has expected methods."""
        from pyai.orchestrator import AgentPatterns
        
        assert hasattr(AgentPatterns, 'supervisor')
        assert hasattr(AgentPatterns, 'consensus')
        assert hasattr(AgentPatterns, 'debate')
        assert hasattr(AgentPatterns, 'chain_of_thought')


class TestUseCasesSmoke:
    """Smoke tests for usecases module."""
    
    def test_general_usecases_exist(self):
        """Test general use cases exist."""
        from pyai.usecases import (
            customer_service, sales, operations,
            development, gaming, list_usecases
        )
        
        assert customer_service is not None
        assert sales is not None
        assert operations is not None
        assert development is not None
        assert gaming is not None
        assert callable(list_usecases)
    
    def test_industry_usecases_exist(self):
        """Test industry use cases exist."""
        from pyai.usecases.industry import (
            telecom, healthcare, finance,
            ecommerce, education, list_industries
        )
        
        assert telecom is not None
        assert healthcare is not None
        assert finance is not None
        assert ecommerce is not None
        assert education is not None
        assert callable(list_industries)
    
    def test_list_usecases_returns_dict(self):
        """Test list_usecases returns dictionary."""
        from pyai.usecases import list_usecases
        
        result = list_usecases()
        assert isinstance(result, dict)
        assert 'customer_service' in result
        assert 'sales' in result
    
    def test_list_industries_returns_dict(self):
        """Test list_industries returns dictionary."""
        from pyai.usecases.industry import list_industries
        
        result = list_industries()
        assert isinstance(result, dict)
        assert 'telecom' in result
        assert 'healthcare' in result
        assert 'finance' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
