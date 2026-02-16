# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Tests for the runner module.

Tests structured execution pattern including:
- Runner class
- RunConfig
- RunResult
- StreamingRunner
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock


class TestRunConfig:
    """Tests for RunConfig."""
    
    def test_runconfig_import(self):
        """Test that RunConfig can be imported."""
        from pyagent.runner import RunConfig
        assert RunConfig is not None
    
    def test_runconfig_defaults(self):
        """Test default values."""
        from pyagent.runner import RunConfig
        
        config = RunConfig()
        assert config.max_turns == 10
        assert config.max_time == 300.0
        assert config.stop_on_tool_error == False
        assert config.verbose == False
    
    def test_runconfig_custom_values(self):
        """Test custom configuration."""
        from pyagent.runner import RunConfig
        
        config = RunConfig(
            max_turns=5,
            max_time=60.0,
            stop_on_tool_error=True,
            verbose=True
        )
        
        assert config.max_turns == 5
        assert config.max_time == 60.0
        assert config.stop_on_tool_error == True


class TestRunContext:
    """Tests for RunContext."""
    
    def test_context_import(self):
        """Test that RunContext can be imported."""
        from pyagent.runner.executor import RunContext
        assert RunContext is not None
    
    def test_context_creation(self):
        """Test creating a context."""
        from pyagent.runner.executor import RunContext
        
        ctx = RunContext(run_id="test-123")
        assert ctx.run_id == "test-123"
        assert ctx.turn_count == 0
    
    def test_context_variables(self):
        """Test context variable storage."""
        from pyagent.runner.executor import RunContext
        
        ctx = RunContext(run_id="test-123")
        ctx.set_variable("key", "value")
        
        assert ctx.get_variable("key") == "value"
        assert ctx.get_variable("missing", "default") == "default"
    
    def test_context_elapsed_time(self):
        """Test elapsed time calculation."""
        from pyagent.runner.executor import RunContext
        import time
        
        ctx = RunContext(run_id="test-123")
        time.sleep(0.1)
        
        assert ctx.elapsed_time() >= 0.1


class TestRunResult:
    """Tests for RunResult."""
    
    def test_runresult_import(self):
        """Test that RunResult can be imported."""
        from pyagent.runner import RunResult
        assert RunResult is not None
    
    def test_runresult_creation(self):
        """Test creating a result."""
        from pyagent.runner import RunResult
        from pyagent.runner.executor import RunStatus
        
        result = RunResult(
            run_id="test-123",
            status=RunStatus.COMPLETED,
            output="Hello, world!",
            turn_count=1,
            elapsed_time=0.5
        )
        
        assert result.run_id == "test-123"
        assert result.success == True
        assert result.output == "Hello, world!"
    
    def test_runresult_failed(self):
        """Test failed result."""
        from pyagent.runner import RunResult
        from pyagent.runner.executor import RunStatus
        
        result = RunResult(
            run_id="test-123",
            status=RunStatus.FAILED,
            error="Something went wrong"
        )
        
        assert result.success == False
        assert result.error == "Something went wrong"
    
    def test_runresult_to_dict(self):
        """Test to_dict conversion."""
        from pyagent.runner import RunResult
        from pyagent.runner.executor import RunStatus
        
        result = RunResult(
            run_id="test-123",
            status=RunStatus.COMPLETED,
            output="Test"
        )
        
        d = result.to_dict()
        assert d["run_id"] == "test-123"
        assert d["status"] == "completed"
        assert d["output"] == "Test"


class TestRunner:
    """Tests for Runner class."""
    
    def test_runner_import(self):
        """Test that Runner can be imported."""
        from pyagent.runner import Runner
        assert Runner is not None
    
    def test_runner_creation(self):
        """Test creating a runner."""
        from pyagent.runner import Runner, RunConfig
        
        config = RunConfig(max_turns=5)
        runner = Runner(config=config)
        
        assert runner.config.max_turns == 5
    
    def test_runner_execute_callable(self):
        """Test executing a callable agent."""
        from pyagent.runner import Runner
        
        def simple_agent(input_text):
            return f"Echo: {input_text}"
        
        result = Runner.run(simple_agent, "Hello")
        
        assert result.success == True
        assert "Echo: Hello" in result.output
    
    def test_runner_execute_with_run_method(self):
        """Test executing agent with run method."""
        from pyagent.runner import Runner
        
        class MockAgent:
            def run(self, input_text):
                return f"Processed: {input_text}"
        
        agent = MockAgent()
        result = Runner.run(agent, "Test input")
        
        assert result.success == True
        assert "Processed: Test input" in result.output
    
    def test_runner_max_turns(self):
        """Test max turns limit."""
        from pyagent.runner import Runner, RunConfig
        
        turn_count = 0
        
        def infinite_agent(input_text):
            nonlocal turn_count
            turn_count += 1
            return {"tool_calls": [{"name": "test"}]}  # Continues loop
        
        config = RunConfig(max_turns=3)
        result = Runner.run(infinite_agent, "Test", config=config)
        
        # Should stop after max_turns
        assert result.turn_count <= 3
    
    def test_runner_error_handling(self):
        """Test error handling."""
        from pyagent.runner import Runner
        
        def error_agent(input_text):
            raise ValueError("Intentional error")
        
        result = Runner.run(error_agent, "Test")
        
        assert result.success == False
        assert "Intentional error" in result.error
    
    def test_runner_run_id_generation(self):
        """Test that run_id is generated."""
        from pyagent.runner import Runner
        
        def simple_agent(input_text):
            return "OK"
        
        result = Runner.run(simple_agent, "Test")
        
        assert result.run_id is not None
        assert result.run_id.startswith("run-")


class TestRunnerAsync:
    """Async tests for Runner."""
    
    @pytest.mark.asyncio
    async def test_runner_async_execution(self):
        """Test async execution."""
        from pyagent.runner import Runner
        
        async def async_agent(input_text):
            await asyncio.sleep(0.01)
            return f"Async: {input_text}"
        
        result = await Runner.run_async(async_agent, "Test")
        
        assert result.success == True
        assert "Async: Test" in result.output
    
    @pytest.mark.asyncio
    async def test_runner_async_with_sync_agent(self):
        """Test async runner with sync agent."""
        from pyagent.runner import Runner
        
        def sync_agent(input_text):
            return f"Sync: {input_text}"
        
        result = await Runner.run_async(sync_agent, "Test")
        
        assert result.success == True
        assert "Sync: Test" in result.output


class TestStreamEvent:
    """Tests for StreamEvent."""
    
    def test_stream_event_import(self):
        """Test that StreamEvent can be imported."""
        from pyagent.runner import StreamEvent
        assert StreamEvent is not None
    
    def test_stream_event_creation(self):
        """Test creating a stream event."""
        from pyagent.runner.streaming import StreamEvent, EventType
        
        event = StreamEvent(
            type=EventType.TOKEN,
            data="Hello",
            run_id="test-123"
        )
        
        assert event.type == EventType.TOKEN
        assert event.data == "Hello"
    
    def test_stream_event_to_dict(self):
        """Test to_dict conversion."""
        from pyagent.runner.streaming import StreamEvent, EventType
        
        event = StreamEvent(
            type=EventType.RUN_START,
            data={"input": "test"},
            run_id="test-123"
        )
        
        d = event.to_dict()
        assert d["type"] == "run_start"
        assert d["run_id"] == "test-123"


class TestStreamingRunner:
    """Tests for StreamingRunner."""
    
    def test_streaming_runner_import(self):
        """Test that StreamingRunner can be imported."""
        from pyagent.runner import StreamingRunner
        assert StreamingRunner is not None
    
    @pytest.mark.asyncio
    async def test_streaming_execution(self):
        """Test streaming execution produces events."""
        from pyagent.runner import StreamingRunner
        from pyagent.runner.streaming import EventType
        
        def simple_agent(input_text):
            return f"Response to: {input_text}"
        
        events = []
        async for event in StreamingRunner.stream(simple_agent, "Test"):
            events.append(event)
        
        # Should have start and end events
        event_types = [e.type for e in events]
        assert EventType.RUN_START in event_types
        assert EventType.RUN_END in event_types
    
    @pytest.mark.asyncio
    async def test_streaming_token_events(self):
        """Test that token events are emitted."""
        from pyagent.runner import StreamingRunner
        from pyagent.runner.streaming import EventType
        
        def simple_agent(input_text):
            return "Token output"
        
        token_events = []
        async for event in StreamingRunner.stream(simple_agent, "Test"):
            if event.type == EventType.TOKEN:
                token_events.append(event)
        
        # Should have at least one token event
        assert len(token_events) >= 1


class TestRunnerIntegration:
    """Integration tests for runner module."""
    
    def test_module_exports(self):
        """Test that all expected exports are available."""
        from pyagent import runner
        
        assert hasattr(runner, "Runner")
        assert hasattr(runner, "RunConfig")
        assert hasattr(runner, "RunResult")
        assert hasattr(runner, "RunContext")
        assert hasattr(runner, "StreamingRunner")
        assert hasattr(runner, "StreamEvent")
    
    def test_main_init_exports(self):
        """Test that runner is exported from main pyagent."""
        import pyagent
        
        assert hasattr(pyagent, "runner")
        assert hasattr(pyagent, "Runner")
        assert hasattr(pyagent, "RunConfig")
        assert hasattr(pyagent, "RunResult")
    
    def test_full_runner_workflow(self):
        """Test complete runner workflow."""
        from pyagent import Runner, RunConfig
        
        # Create a mock agent
        class TestAgent:
            def __init__(self):
                self.call_count = 0
            
            def run(self, input_text):
                self.call_count += 1
                return f"Response #{self.call_count}: {input_text}"
        
        # Configure and run
        agent = TestAgent()
        config = RunConfig(max_turns=5, verbose=True)
        
        result = Runner.run(agent, "Test query", config=config)
        
        assert result.success == True
        assert "Response #1" in result.output
        assert result.elapsed_time >= 0  # Can be 0 for very fast executions
        assert result.turn_count == 1
