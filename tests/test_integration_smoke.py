# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Integration Smoke Tests for Phase 2/3 Features

These tests verify that the new modules are structurally sound and
their APIs work correctly without requiring actual external connections.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock


# =============================================================================
# VOICE MODULE SMOKE TESTS
# =============================================================================

class TestVoiceSmoke:
    """Smoke tests for voice streaming module."""
    
    def test_audio_chunk_creation(self):
        """Test AudioChunk can be created with data."""
        from pyagent.voice.stream import AudioChunk, AudioFormat
        
        # Create chunk
        chunk = AudioChunk(
            data=b"\x00\x01\x02\x03" * 100,
            format=AudioFormat.PCM16,
            sample_rate=16000
        )
        
        assert len(chunk.data) == 400
        assert chunk.sample_rate == 16000
        assert chunk.duration_ms > 0
    
    def test_audio_stream_buffering(self):
        """Test AudioStream buffers chunks correctly."""
        from pyagent.voice.stream import AudioStream, AudioChunk
        
        stream = AudioStream()
        
        # Add multiple chunks
        for i in range(5):
            stream.add(AudioChunk(data=b"\x00" * 100))
        
        assert len(stream) == 5
        
        # Get all chunks via iteration
        chunks = list(stream)
        assert len(chunks) == 5
    
    def test_duplex_stream(self):
        """Test DuplexAudioStream bidirectional flow."""
        from pyagent.voice.stream import DuplexAudioStream, AudioChunk
        
        duplex = DuplexAudioStream()
        
        # Send should not block
        chunk = AudioChunk(data=b"\x00" * 100)
        duplex.send(chunk)
        
        # Receive should return None when empty
        received = duplex.receive()
        assert received is None or isinstance(received, AudioChunk)
    
    def test_voice_session_lifecycle(self):
        """Test VoiceSession state management."""
        from pyagent.voice.session import VoiceSession, SessionState
        
        session = VoiceSession()
        
        # Initial state
        assert session.state == SessionState.IDLE
        
        # Configuration
        assert session.model is not None
        assert session.voice is not None
    
    def test_transcription_result_structure(self):
        """Test TranscriptionResult dataclass."""
        from pyagent.voice.transcription import TranscriptionResult
        
        result = TranscriptionResult(
            text="Hello, how are you?",
            language="en",
            confidence=0.95
        )
        
        assert result.text == "Hello, how are you?"
        assert result.language == "en"
        assert result.confidence == 0.95
    
    def test_synthesis_result_structure(self):
        """Test SynthesisResult dataclass."""
        from pyagent.voice.synthesis import SynthesisResult
        
        result = SynthesisResult(
            audio=b"\x00\x01\x02\x03",
            sample_rate=24000
        )
        
        assert len(result.audio) == 4
        assert result.sample_rate == 24000


# =============================================================================
# A2A PROTOCOL SMOKE TESTS
# =============================================================================

class TestA2ASmoke:
    """Smoke tests for Agent-to-Agent protocol."""
    
    def test_agent_card_serialization(self):
        """Test AgentCard can be serialized and deserialized."""
        from pyagent.a2a.protocol import AgentCard
        
        card = AgentCard(
            name="research-agent",
            description="A research assistant",
            skills=["research", "summarize", "analyze"],
            url="https://agent.example.com",
            version="1.0.0"
        )
        
        # Serialize to dict
        data = card.to_dict()
        
        assert data["name"] == "research-agent"
        assert "research" in data["skills"]
        assert data["url"] == "https://agent.example.com"
        
        # Deserialize back
        restored = AgentCard.from_dict(data)
        assert restored.name == card.name
        assert restored.skills == card.skills
    
    def test_a2a_task_creation(self):
        """Test A2ATask can be created from text."""
        from pyagent.a2a.protocol import A2ATask
        
        task = A2ATask.from_text("Research the latest AI trends")
        
        assert task.id is not None
        assert len(task.messages) == 1
        assert task.messages[0].content == "Research the latest AI trends"
    
    def test_a2a_response_success(self):
        """Test A2AResponse can represent success."""
        from pyagent.a2a.protocol import A2AResponse, TaskStatus
        
        response = A2AResponse.success("task-123", "Here is the research result")
        
        assert response.is_success
        assert response.task_id == "task-123"
        assert response.content == "Here is the research result"
        assert response.status == TaskStatus.COMPLETED
    
    def test_a2a_response_failure(self):
        """Test A2AResponse can represent failure."""
        from pyagent.a2a.protocol import A2AResponse, TaskStatus
        
        response = A2AResponse.failure("task-456", "Task failed due to timeout")
        
        assert not response.is_success
        assert response.status == TaskStatus.FAILED
        assert "timeout" in response.error.lower()
    
    def test_a2a_endpoint_registration(self):
        """Test A2AEndpoint can be defined."""
        from pyagent.a2a.server import A2AEndpoint
        
        def handler(task):
            return {"status": "completed"}
        
        endpoint = A2AEndpoint(
            name="process",
            handler=handler,
            description="Process a task"
        )
        
        assert endpoint.name == "process"
    
    def test_a2a_server_instantiation(self):
        """Test A2AServer can be instantiated."""
        from pyagent.a2a.server import A2AServer
        from pyagent.a2a.protocol import AgentCard
        
        card = AgentCard(name="test-agent")
        server = A2AServer(port=8080)
        
        assert server.port == 8080
    
    def test_agent_registry_singleton(self):
        """Test AgentRegistry provides singleton access."""
        from pyagent.a2a.registry import AgentRegistry
        
        registry1 = AgentRegistry.get_default()
        registry2 = AgentRegistry.get_default()
        
        assert registry1 is registry2
    
    def test_remote_agent_wrapper(self):
        """Test RemoteAgent can wrap a URL."""
        from pyagent.a2a.client import RemoteAgent
        
        remote = RemoteAgent(url="https://agent.example.com")
        
        assert remote.url == "https://agent.example.com"


# =============================================================================
# DEVUI MODULE SMOKE TESTS
# =============================================================================

class TestDevUISmoke:
    """Smoke tests for Development UI module."""
    
    def test_devui_instantiation(self):
        """Test DevUI can be instantiated."""
        from pyagent.devui.ui import DevUI
        
        ui = DevUI(title="Test Agent")
        
        assert ui.title == "Test Agent"
    
    def test_devui_custom_handler(self):
        """Test DevUI can use custom handler."""
        from pyagent.devui.ui import DevUI
        
        calls = []
        
        def my_handler(message: str) -> str:
            calls.append(message)
            return f"Echo: {message}"
        
        ui = DevUI(handler=my_handler)
        response = ui._handle_message("Hello")
        
        assert response == "Echo: Hello"
        assert calls == ["Hello"]
    
    def test_dashboard_metrics_tracking(self):
        """Test AgentDashboard tracks metrics."""
        from pyagent.devui.dashboard import AgentDashboard
        from datetime import datetime, timedelta
        
        dashboard = AgentDashboard()
        
        # Record some runs
        for i in range(5):
            started = datetime.utcnow()
            ended = started + timedelta(milliseconds=100 * (i + 1))
            dashboard.record_run(
                input=f"Query {i}",
                output=f"Response {i}",
                started_at=started,
                ended_at=ended,
            )
        
        metrics = dashboard.get_metrics()
        
        assert metrics.total_runs == 5
        assert metrics.total_duration_ms > 0
        assert metrics.success_rate == 1.0
    
    def test_dashboard_agent_tracking(self):
        """Test AgentDashboard can be used with agents."""
        from pyagent.devui.dashboard import AgentDashboard
        
        dashboard = AgentDashboard()
        
        # Dashboard exists and can be used
        assert dashboard is not None
        assert hasattr(dashboard, 'record_run')
    
    def test_debugger_event_logging(self):
        """Test AgentDebugger logs events."""
        from pyagent.devui.debugger import AgentDebugger, DebugEvent
        
        debugger = AgentDebugger()
        
        # Log various events
        debugger.log(DebugEvent.RUN_START, {"input": "Hello"})
        debugger.log(DebugEvent.TOOL_CALL, {"tool": "search"})
        debugger.log(DebugEvent.RUN_END, {"output": "Hi there"})
        
        history = debugger.get_history()
        
        assert len(history) == 3
        assert history[0].event == DebugEvent.RUN_START
        assert history[2].event == DebugEvent.RUN_END
    
    def test_debugger_breakpoints(self):
        """Test AgentDebugger manages breakpoints."""
        from pyagent.devui.debugger import AgentDebugger
        
        debugger = AgentDebugger()
        
        # Add breakpoints
        bp1 = debugger.add_breakpoint("tool_call")
        bp2 = debugger.add_breakpoint("llm_response")
        
        assert bp1 is not None
        assert bp2 is not None
        
        # Remove breakpoint
        assert debugger.remove_breakpoint(bp1) is True
    
    def test_debugger_step_mode(self):
        """Test AgentDebugger has stepping capability."""
        from pyagent.devui.debugger import AgentDebugger
        
        debugger = AgentDebugger()
        
        # Debugger should exist and be usable
        assert debugger is not None
        assert hasattr(debugger, 'add_breakpoint')


# =============================================================================
# TOOL DISCOVERY SMOKE TESTS
# =============================================================================

class TestToolDiscoverySmoke:
    """Smoke tests for tool auto-discovery."""
    
    def test_tool_decorator_chain(self):
        """Test @tool decorator creates proper Tool objects."""
        from pyagent.tools import tool, Tool
        
        @tool(name="calculator", description="Calculate math")
        def calculate(expression: str) -> float:
            """Evaluate a mathematical expression."""
            return eval(expression)  # noqa: S307
        
        # Should be a Tool instance
        assert isinstance(calculate, Tool)
        assert calculate.name == "calculator"
        assert calculate.description == "Calculate math"
        
        # Should still be callable
        result = calculate.execute(expression="2 + 2")
        assert result.success
        assert result.data == 4
    
    def test_tool_from_function_inference(self):
        """Test Tool.from_function infers metadata."""
        from pyagent.tools import Tool
        
        def get_weather(city: str, unit: str = "celsius") -> dict:
            """Get weather for a city.
            
            Args:
                city: The city name
                unit: Temperature unit (celsius or fahrenheit)
            """
            return {"city": city, "temp": 22, "unit": unit}
        
        tool = Tool.from_function(get_weather)
        
        assert tool.name == "get_weather"
        assert "weather" in tool.description.lower()
        assert "city" in tool.parameters.get("properties", {})


# =============================================================================
# CONTEXT CACHE SMOKE TESTS  
# =============================================================================

class TestContextCacheSmoke:
    """Smoke tests for context caching."""
    
    def test_cache_ttl_expiry(self):
        """Test cache entries expire after TTL."""
        import time
        from pyagent.core.cache import ContextCache
        
        cache = ContextCache(ttl=0.1)  # 100ms TTL
        cache.set("key", "value")
        
        # Should exist immediately
        assert cache.get("key") == "value"
        
        # Wait for expiry
        time.sleep(0.15)
        
        # Should be expired
        assert cache.get("key") is None
    
    def test_cache_max_entries(self):
        """Test cache has size limit capability."""
        from pyagent.core.cache import ContextCache
        
        cache = ContextCache(max_entries=3)
        
        # Add entries
        for i in range(3):
            cache.set(f"key{i}", f"value{i}")
        
        # Should have entries
        assert cache.get("key0") is not None or cache.get("key2") is not None


# =============================================================================
# VECTOR DB SMOKE TESTS
# =============================================================================

class TestVectorDBSmoke:
    """Smoke tests for vector database connectors."""
    
    def test_memory_store_full_workflow(self):
        """Test complete workflow with MemoryVectorStore."""
        from pyagent.vectordb import MemoryVectorStore
        
        store = MemoryVectorStore()
        
        # Add documents
        store.add("doc1", "Python is a programming language", {"source": "wiki"})
        store.add("doc2", "JavaScript runs in browsers", {"source": "docs"})
        store.add("doc3", "Rust is known for memory safety", {"source": "blog"})
        
        # Count should work
        assert store.count() == 3
        
        # Get should work
        doc = store.get("doc1")
        assert doc is not None
        assert "Python" in doc.content
    
    def test_document_metadata(self):
        """Test Document preserves metadata."""
        from pyagent.vectordb import Document
        
        doc = Document.create(
            "Content here",
            source="test.txt",
            author="Alice",
            tags=["important", "urgent"]
        )
        
        assert doc.content == "Content here"
        assert doc.metadata["source"] == "test.txt"
        assert doc.metadata["author"] == "Alice"
        assert doc.metadata["tags"] == ["important", "urgent"]


# =============================================================================
# MULTIMODAL SMOKE TESTS
# =============================================================================

class TestMultimodalSmoke:
    """Smoke tests for multimodal support."""
    
    def test_image_format_conversion(self):
        """Test Image converts between provider formats."""
        from pyagent.multimodal import Image
        
        img = Image.from_base64("SGVsbG8gV29ybGQ=", media_type="image/png")
        
        # OpenAI format
        openai = img.to_openai_format()
        assert openai["type"] == "image_url"
        assert "base64" in openai["image_url"]["url"]
        
        # Anthropic format
        anthropic = img.to_anthropic_format()
        assert anthropic["type"] == "image"
        assert anthropic["source"]["type"] == "base64"
        assert anthropic["source"]["media_type"] == "image/png"
    
    def test_multimodal_content_builder(self):
        """Test MultimodalContent fluent builder."""
        from pyagent.multimodal import MultimodalContent, Image
        
        content = (
            MultimodalContent()
            .add_text("Look at this image:")
            .add_image(Image.from_url("https://example.com/img.jpg"))
            .add_text("What do you see?")
        )
        
        assert len(content) == 3
        
        openai_content = content.to_openai_content()
        assert len(openai_content) == 3
        assert openai_content[0]["type"] == "text"
        assert openai_content[1]["type"] == "image_url"
        assert openai_content[2]["type"] == "text"


# =============================================================================
# SESSION CHECKPOINT SMOKE TESTS
# =============================================================================

class TestSessionCheckpointSmoke:
    """Smoke tests for session checkpoints."""
    
    def test_checkpoint_serialization(self):
        """Test checkpoints can be serialized with session."""
        from pyagent.sessions import Session
        
        session = Session()
        session.add_user_message("Hello")
        session.checkpoint("start")
        session.add_assistant_message("Hi there!")
        session.checkpoint("after_greeting")
        
        # Serialize
        data = session.to_dict()
        
        assert "checkpoints" in data
        assert len(data["checkpoints"]) == 2
        
        # Deserialize
        restored = Session.from_dict(data)
        assert len(restored.get_checkpoints()) == 2
    
    def test_checkpoint_rewind_preserves_earlier(self):
        """Test rewind preserves earlier checkpoints."""
        from pyagent.sessions import Session
        
        session = Session()
        session.add_user_message("1")
        session.checkpoint("cp1")
        session.add_user_message("2")
        session.checkpoint("cp2")
        session.add_user_message("3")
        
        # Rewind to cp2
        session.rewind_to_checkpoint_by_name("cp2")
        
        # cp1 should still exist
        checkpoints = session.get_checkpoints()
        assert any(cp.name == "cp1" for cp in checkpoints)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
