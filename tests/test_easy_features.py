"""
Tests for PyAgent Easy Module Features
======================================

Tests for handoff, mcp, guardrails, and trace modules.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestHandoffImport:
    """Test handoff module imports."""
    
    def test_import_handoff_from_pyagent(self):
        """Test importing handoff from pyagent."""
        from pyagent import handoff
        assert handoff is not None
    
    def test_import_handoff_from_easy(self):
        """Test importing handoff from pyagent.easy."""
        from pyagent.easy.handoff import handoff
        assert handoff is not None
    
    def test_handoff_is_callable(self):
        """Test that handoff is callable."""
        from pyagent import handoff
        assert callable(handoff)


class TestHandoffFunction:
    """Tests for handoff function."""
    
    def test_handoff_has_team_method(self):
        """Test that handoff has team method."""
        from pyagent import handoff
        assert hasattr(handoff, 'team')
        assert callable(handoff.team)
    
    def test_handoff_has_chain_method(self):
        """Test that handoff has chain method."""
        from pyagent import handoff
        assert hasattr(handoff, 'chain')
        assert callable(handoff.chain)
    
    def test_handoff_team_creates_team(self):
        """Test that handoff.team creates an AgentTeam."""
        from pyagent import handoff
        from pyagent.easy.agent_factory import agent
        
        a1 = agent("Agent 1")
        a2 = agent("Agent 2")
        
        team = handoff.team([a1, a2])
        assert team is not None
        assert hasattr(team, 'route')


class TestMCPImport:
    """Test MCP module imports."""
    
    def test_import_mcp_from_pyagent(self):
        """Test importing mcp from pyagent."""
        from pyagent import mcp
        assert mcp is not None
    
    def test_import_mcp_from_easy(self):
        """Test importing mcp from pyagent.easy."""
        from pyagent.easy.mcp import mcp
        assert mcp is not None


class TestMCPFunction:
    """Tests for MCP function."""
    
    def test_mcp_has_tool_decorator(self):
        """Test that mcp has tool decorator."""
        from pyagent import mcp
        assert hasattr(mcp, 'tool')
        assert callable(mcp.tool)
    
    def test_mcp_has_server_method(self):
        """Test that mcp has server method."""
        from pyagent import mcp
        assert hasattr(mcp, 'server')
        assert callable(mcp.server)
    
    def test_mcp_has_connect_method(self):
        """Test that mcp has connect method."""
        from pyagent import mcp
        assert hasattr(mcp, 'connect')
        assert callable(mcp.connect)
    
    def test_mcp_tool_decorator(self):
        """Test that @mcp.tool decorator works."""
        from pyagent import mcp
        
        @mcp.tool("test_tool")
        def my_tool(x: int) -> int:
            """A test tool."""
            return x * 2
        
        # The decorated function should be a Tool
        assert my_tool is not None
        assert hasattr(my_tool, 'name')
        assert my_tool.name == "test_tool"
    
    def test_mcp_server_creation(self):
        """Test creating an MCP server."""
        from pyagent import mcp
        
        @mcp.tool("adder")
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b
        
        server = mcp.server("test-server", tools=[add])
        assert server is not None
        assert server.name == "test-server"


class TestGuardrailsImport:
    """Test guardrails module imports."""
    
    def test_import_guardrails_from_pyagent(self):
        """Test importing guardrails from pyagent."""
        from pyagent import guardrails
        assert guardrails is not None
    
    def test_import_guardrails_from_easy(self):
        """Test importing guardrails from pyagent.easy."""
        from pyagent.easy.guardrails import guardrails
        assert guardrails is not None


class TestGuardrailsFunction:
    """Tests for guardrails function."""
    
    def test_guardrails_has_validate_method(self):
        """Test that guardrails has validate method."""
        from pyagent import guardrails
        assert hasattr(guardrails, 'validate')
        assert callable(guardrails.validate)
    
    def test_guardrails_has_filter_output_method(self):
        """Test that guardrails has filter_output method."""
        from pyagent import guardrails
        assert hasattr(guardrails, 'filter_output')
        assert callable(guardrails.filter_output)
    
    def test_guardrails_has_wrap_method(self):
        """Test that guardrails has wrap method."""
        from pyagent import guardrails
        assert hasattr(guardrails, 'wrap')
        assert callable(guardrails.wrap)
    
    def test_guardrails_has_no_pii_validator(self):
        """Test that guardrails has no_pii validator."""
        from pyagent import guardrails
        assert hasattr(guardrails, 'no_pii')
        assert callable(guardrails.no_pii)
    
    def test_guardrails_has_no_injection_validator(self):
        """Test that guardrails has no_injection validator."""
        from pyagent import guardrails
        assert hasattr(guardrails, 'no_injection')
        assert callable(guardrails.no_injection)
    
    def test_guardrails_has_redact_pii_filter(self):
        """Test that guardrails has redact_pii filter."""
        from pyagent import guardrails
        assert hasattr(guardrails, 'redact_pii')
        assert callable(guardrails.redact_pii)
    
    def test_guardrails_has_protect_method(self):
        """Test that guardrails has protect method."""
        from pyagent import guardrails
        assert hasattr(guardrails, 'protect')
        assert callable(guardrails.protect)


class TestTraceImport:
    """Test trace module imports."""
    
    def test_import_trace_from_pyagent(self):
        """Test importing trace from pyagent."""
        from pyagent import trace
        assert trace is not None
    
    def test_import_trace_from_easy(self):
        """Test importing trace from pyagent.easy."""
        from pyagent.easy.trace import trace
        assert trace is not None


class TestTraceFunction:
    """Tests for trace function."""
    
    def test_trace_has_enable_method(self):
        """Test that trace has enable method."""
        from pyagent import trace
        assert hasattr(trace, 'enable')
        assert callable(trace.enable)
    
    def test_trace_has_disable_method(self):
        """Test that trace has disable method."""
        from pyagent import trace
        assert hasattr(trace, 'disable')
        assert callable(trace.disable)
    
    def test_trace_has_span_method(self):
        """Test that trace has span method."""
        from pyagent import trace
        assert hasattr(trace, 'span')
        assert callable(trace.span)
    
    def test_trace_has_show_method(self):
        """Test that trace has show method."""
        from pyagent import trace
        assert hasattr(trace, 'show')
        assert callable(trace.show)
    
    def test_trace_enable_disable(self):
        """Test enabling and disabling tracing."""
        from pyagent import trace
        
        trace.enable()
        assert trace.enabled == True
        
        trace.disable()
        assert trace.enabled == False
    
    def test_trace_span_context_manager(self):
        """Test trace span as context manager."""
        from pyagent import trace
        
        trace.enable()
        
        with trace.span("test_span") as span:
            span.log("test event")
        
        # Check span was recorded
        summary = trace.summary()
        assert summary["total_spans"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
