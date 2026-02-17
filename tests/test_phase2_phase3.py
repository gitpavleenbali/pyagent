# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Tests for new Phase 2/3 features:
- Tool Auto-Discovery
- Context Caching
- Session Checkpoints/Rewind
- Multimodal
- Vector DB Connectors
- A2A Protocol
- Dev UI
- Voice Streaming
"""

import pytest
import tempfile
import os
import threading
import time


# =============================================================================
# TOOL AUTO-DISCOVERY TESTS
# =============================================================================

class TestToolBase:
    """Test Tool class and @tool decorator."""
    
    def test_import_tool_class(self):
        from pyai.tools import Tool
        assert Tool is not None
    
    def test_import_tool_decorator(self):
        from pyai.tools import tool
        assert callable(tool)
    
    def test_tool_from_function(self):
        from pyai.tools import Tool
        
        def my_func(x: int, y: str = "default") -> str:
            """A test function."""
            return f"{y}: {x}"
        
        t = Tool.from_function(my_func)
        assert t.name == "my_func"
        assert "test function" in t.description.lower()
        assert callable(t.func)
    
    def test_tool_decorator(self):
        from pyai.tools import tool
        
        @tool(name="custom_name", description="Custom desc")
        def my_tool(a: int) -> int:
            return a * 2
        
        assert my_tool.name == "custom_name"
        assert my_tool.description == "Custom desc"
    
    def test_tool_execution(self):
        from pyai.tools import tool
        
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b
        
        result = add.execute(a=2, b=3)
        assert result.success is True
        assert result.data == 5
    
    def test_tool_openai_schema(self):
        from pyai.tools import tool
        
        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Sunny in {city}"
        
        schema = get_weather.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"


class TestToolDiscovery:
    """Test tool discovery from directories."""
    
    def test_import_discovery(self):
        from pyai.tools import ToolDiscovery
        assert ToolDiscovery is not None
    
    def test_discovery_instance(self):
        from pyai.tools import ToolDiscovery
        
        discovery = ToolDiscovery()
        assert discovery is not None
        assert hasattr(discovery, 'scan')
        assert hasattr(discovery, 'get_all_tools')
    
    def test_discover_tools_function(self):
        from pyai.tools import discover_tools
        assert callable(discover_tools)
    
    def test_load_tools_from_directory(self):
        from pyai.tools import load_tools_from_directory
        assert callable(load_tools_from_directory)


class TestToolWatcher:
    """Test hot-reload tool watcher."""
    
    def test_import_watcher(self):
        from pyai.tools import ToolWatcher
        assert ToolWatcher is not None
    
    def test_watcher_instance(self):
        from pyai.tools import ToolWatcher
        
        watcher = ToolWatcher(".")
        assert watcher is not None
        assert hasattr(watcher, 'start')
        assert hasattr(watcher, 'stop')


# =============================================================================
# CONTEXT CACHING TESTS
# =============================================================================

class TestContextCache:
    """Test context caching."""
    
    def test_import_context_cache(self):
        from pyai.core.cache import ContextCache
        assert ContextCache is not None
    
    def test_cache_basic(self):
        from pyai.core.cache import ContextCache
        
        cache = ContextCache(ttl=10)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
    
    def test_cache_miss(self):
        from pyai.core.cache import ContextCache
        
        cache = ContextCache()
        assert cache.get("nonexistent") is None
        assert cache.get("nonexistent", "default") == "default"
    
    def test_cache_delete(self):
        from pyai.core.cache import ContextCache
        
        cache = ContextCache()
        cache.set("key", "value")
        assert cache.delete("key") is True
        assert cache.get("key") is None
    
    def test_cache_clear(self):
        from pyai.core.cache import ContextCache
        
        cache = ContextCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None
    
    def test_cache_decorator(self):
        from pyai.core.cache import ContextCache
        
        cache = ContextCache()
        call_count = 0
        
        @cache.cached
        def expensive_func():
            nonlocal call_count
            call_count += 1
            return "result"
        
        assert expensive_func() == "result"
        assert expensive_func() == "result"
        assert call_count == 1  # Should only call once
    
    def test_cache_stats(self):
        from pyai.core.cache import ContextCache
        
        cache = ContextCache()
        cache.set("key", "value")
        cache.get("key")  # Hit
        cache.get("missing")  # Miss
        
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1


# =============================================================================
# SESSION REWIND TESTS
# =============================================================================

class TestSessionCheckpoint:
    """Test session checkpoint and rewind."""
    
    def test_import_checkpoint(self):
        from pyai.sessions import SessionCheckpoint
        assert SessionCheckpoint is not None
    
    def test_session_checkpoint(self):
        from pyai.sessions import Session
        
        session = Session()
        session.add_user_message("Message 1")
        
        cp = session.checkpoint("before_research")
        assert cp.name == "before_research"
        assert cp.message_index == 1
    
    def test_session_rewind_to_checkpoint(self):
        from pyai.sessions import Session
        
        session = Session()
        session.add_user_message("Message 1")
        cp = session.checkpoint()
        session.add_assistant_message("Message 2")
        session.add_user_message("Message 3")
        
        assert len(session.messages) == 3
        
        success = session.rewind_to_checkpoint(cp.id)
        assert success is True
        assert len(session.messages) == 1
    
    def test_session_rewind_by_name(self):
        from pyai.sessions import Session
        
        session = Session()
        session.add_user_message("Start")
        session.checkpoint("start_point")
        session.add_assistant_message("Response")
        
        success = session.rewind_to_checkpoint_by_name("start_point")
        assert success is True
        assert len(session.messages) == 1
    
    def test_session_rewind_n_messages(self):
        from pyai.sessions import Session
        
        session = Session()
        session.add_user_message("1")
        session.add_assistant_message("2")
        session.add_user_message("3")
        
        removed = session.rewind_n_messages(2)
        assert removed == 2
        assert len(session.messages) == 1
    
    def test_checkpoint_context_restore(self):
        from pyai.sessions import Session
        
        session = Session()
        session.context["key"] = "original"
        cp = session.checkpoint()
        session.context["key"] = "modified"
        
        session.rewind_to_checkpoint(cp.id, restore_context=True)
        assert session.context["key"] == "original"


# =============================================================================
# MULTIMODAL TESTS
# =============================================================================

class TestImage:
    """Test Image class."""
    
    def test_import_image(self):
        from pyai.multimodal import Image
        assert Image is not None
    
    def test_image_from_url(self):
        from pyai.multimodal import Image
        
        img = Image.from_url("https://example.com/image.png")
        assert img.data == "https://example.com/image.png"
        assert img.media_type == "image/png"
    
    def test_image_from_base64(self):
        from pyai.multimodal import Image
        
        img = Image.from_base64("SGVsbG8=", media_type="image/jpeg")
        assert img.data == "SGVsbG8="
        assert img.media_type == "image/jpeg"
    
    def test_image_openai_format(self):
        from pyai.multimodal import Image
        
        img = Image.from_url("https://example.com/img.jpg")
        fmt = img.to_openai_format()
        
        assert fmt["type"] == "image_url"
        assert "url" in fmt["image_url"]
    
    def test_image_anthropic_format(self):
        from pyai.multimodal import Image
        
        img = Image.from_base64("data", media_type="image/png")
        fmt = img.to_anthropic_format()
        
        assert fmt["type"] == "image"
        assert fmt["source"]["type"] == "base64"


class TestAudio:
    """Test Audio class."""
    
    def test_import_audio(self):
        from pyai.multimodal import Audio
        assert Audio is not None
    
    def test_audio_from_base64(self):
        from pyai.multimodal import Audio, AudioFormat
        
        audio = Audio.from_base64("SGVsbG8=", format=AudioFormat.WAV)
        assert audio.data == "SGVsbG8="
        assert audio.format == AudioFormat.WAV
    
    def test_audio_media_type(self):
        from pyai.multimodal import Audio, AudioFormat
        
        audio = Audio.from_base64("data", format=AudioFormat.MP3)
        assert audio.media_type == "audio/mpeg"


class TestMultimodalContent:
    """Test multimodal content composition."""
    
    def test_import_content(self):
        from pyai.multimodal import MultimodalContent
        assert MultimodalContent is not None
    
    def test_content_add_text(self):
        from pyai.multimodal import MultimodalContent
        
        content = MultimodalContent()
        content.add_text("Hello world")
        
        assert len(content) == 1
        assert content.get_text() == "Hello world"
    
    def test_content_chaining(self):
        from pyai.multimodal import MultimodalContent, Image
        
        content = (
            MultimodalContent()
            .add_text("Describe this:")
            .add_image(Image.from_url("https://example.com/img.jpg"))
        )
        
        assert len(content) == 2
    
    def test_content_to_openai(self):
        from pyai.multimodal import MultimodalContent
        
        content = MultimodalContent()
        content.add_text("Hello")
        
        openai_content = content.to_openai_content()
        assert openai_content[0]["type"] == "text"
        assert openai_content[0]["text"] == "Hello"


# =============================================================================
# VECTOR DB TESTS
# =============================================================================

class TestMemoryVectorStore:
    """Test in-memory vector store."""
    
    def test_import_memory_store(self):
        from pyai.vectordb import MemoryVectorStore
        assert MemoryVectorStore is not None
    
    def test_store_add_get(self):
        from pyai.vectordb import MemoryVectorStore
        
        store = MemoryVectorStore()
        store.add("doc1", "Hello world", {"source": "test"})
        
        doc = store.get("doc1")
        assert doc is not None
        assert doc.content == "Hello world"
        assert doc.metadata["source"] == "test"
    
    def test_store_search(self):
        from pyai.vectordb import MemoryVectorStore
        
        store = MemoryVectorStore()
        store.add("doc1", "Python is a programming language")
        store.add("doc2", "JavaScript is also a programming language")
        store.add("doc3", "Cats are animals")
        
        results = store.search("programming", k=2)
        assert len(results) == 2
    
    def test_store_delete(self):
        from pyai.vectordb import MemoryVectorStore
        
        store = MemoryVectorStore()
        store.add("doc1", "Content")
        
        assert store.delete("doc1") is True
        assert store.get("doc1") is None
    
    def test_store_count(self):
        from pyai.vectordb import MemoryVectorStore
        
        store = MemoryVectorStore()
        store.add("doc1", "A")
        store.add("doc2", "B")
        
        assert store.count() == 2
        assert len(store) == 2


class TestDocument:
    """Test Document class."""
    
    def test_import_document(self):
        from pyai.vectordb import Document
        assert Document is not None
    
    def test_document_create(self):
        from pyai.vectordb import Document
        
        doc = Document.create("Hello world", source="test")
        assert doc.content == "Hello world"
        assert doc.metadata["source"] == "test"
        assert doc.id is not None


# =============================================================================
# A2A PROTOCOL TESTS
# =============================================================================

class TestA2AProtocol:
    """Test A2A protocol types."""
    
    def test_import_agent_card(self):
        from pyai.a2a import AgentCard
        assert AgentCard is not None
    
    def test_agent_card_creation(self):
        from pyai.a2a import AgentCard
        
        card = AgentCard(
            name="research-agent",
            description="Research assistant",
            skills=["research", "summarize"],
        )
        
        assert card.name == "research-agent"
        assert "research" in card.skills
    
    def test_agent_card_to_dict(self):
        from pyai.a2a import AgentCard
        
        card = AgentCard(name="test")
        data = card.to_dict()
        
        assert data["name"] == "test"
        assert "protocols" in data
    
    def test_a2a_task_creation(self):
        from pyai.a2a import A2ATask
        
        task = A2ATask.from_text("Hello, what can you do?")
        assert len(task.messages) == 1
        assert task.messages[0].content == "Hello, what can you do?"
    
    def test_a2a_response(self):
        from pyai.a2a import A2AResponse, TaskStatus
        
        response = A2AResponse.success("task-123", "Here is the result")
        assert response.is_success is True
        assert response.content == "Here is the result"


class TestA2AServer:
    """Test A2A server."""
    
    def test_import_server(self):
        from pyai.a2a import A2AServer
        assert A2AServer is not None
    
    def test_endpoint_creation(self):
        from pyai.a2a import A2AEndpoint, A2ATask, A2AResponse
        
        def handler(task):
            return A2AResponse.success(task.id, "Done")
        
        endpoint = A2AEndpoint(name="test", handler=handler)
        assert endpoint.name == "test"


class TestA2AClient:
    """Test A2A client."""
    
    def test_import_client(self):
        from pyai.a2a import A2AClient
        assert A2AClient is not None
    
    def test_import_remote_agent(self):
        from pyai.a2a import RemoteAgent
        assert RemoteAgent is not None


class TestAgentRegistry:
    """Test agent registry."""
    
    def test_import_registry(self):
        from pyai.a2a import AgentRegistry
        assert AgentRegistry is not None
    
    def test_registry_default(self):
        from pyai.a2a import AgentRegistry
        
        registry = AgentRegistry.get_default()
        assert registry is not None


# =============================================================================
# DEV UI TESTS
# =============================================================================

class TestDevUI:
    """Test development UI."""
    
    def test_import_devui(self):
        from pyai.devui import DevUI
        assert DevUI is not None
    
    def test_import_launch_ui(self):
        from pyai.devui import launch_ui
        assert callable(launch_ui)
    
    def test_devui_creation(self):
        from pyai.devui import DevUI
        
        ui = DevUI(title="Test UI")
        assert ui.title == "Test UI"
    
    def test_devui_with_handler(self):
        from pyai.devui import DevUI
        
        def my_handler(msg):
            return f"Echo: {msg}"
        
        ui = DevUI(handler=my_handler)
        response = ui._handle_message("Hello")
        assert response == "Echo: Hello"


class TestAgentDashboard:
    """Test agent dashboard."""
    
    def test_import_dashboard(self):
        from pyai.devui import AgentDashboard
        assert AgentDashboard is not None
    
    def test_dashboard_metrics(self):
        from pyai.devui import AgentDashboard
        from datetime import datetime
        
        dashboard = AgentDashboard()
        dashboard.record_run(
            input="Hello",
            output="Hi there",
            started_at=datetime.utcnow(),
            ended_at=datetime.utcnow(),
        )
        
        metrics = dashboard.get_metrics()
        assert metrics.total_runs == 1


class TestAgentDebugger:
    """Test agent debugger."""
    
    def test_import_debugger(self):
        from pyai.devui import AgentDebugger
        assert AgentDebugger is not None
    
    def test_debugger_log(self):
        from pyai.devui.debugger import AgentDebugger, DebugEvent
        
        debugger = AgentDebugger()
        entry = debugger.log(DebugEvent.RUN_START, {"input": "test"})
        
        assert entry.event == DebugEvent.RUN_START
        assert entry.data["input"] == "test"
    
    def test_debugger_breakpoint(self):
        from pyai.devui import AgentDebugger
        
        debugger = AgentDebugger()
        bp_id = debugger.add_breakpoint("tool_call")
        
        assert bp_id is not None
        assert debugger.remove_breakpoint(bp_id) is True


# =============================================================================
# VOICE STREAMING TESTS
# =============================================================================

class TestAudioStream:
    """Test audio streaming."""
    
    def test_import_audio_stream(self):
        from pyai.voice import AudioStream
        assert AudioStream is not None
    
    def test_audio_chunk(self):
        from pyai.voice.stream import AudioChunk, AudioFormat
        
        chunk = AudioChunk(
            data=b"\x00" * 100,
            format=AudioFormat.PCM16,
            sample_rate=16000,
        )
        
        assert len(chunk.data) == 100
        assert chunk.duration_ms > 0
    
    def test_stream_add_chunks(self):
        from pyai.voice.stream import AudioStream, AudioChunk
        
        stream = AudioStream()
        stream.add(AudioChunk(data=b"\x00\x00"))
        stream.add(AudioChunk(data=b"\x01\x01"))
        
        assert len(stream) == 2


class TestVoiceSession:
    """Test voice session."""
    
    def test_import_session(self):
        from pyai.voice import VoiceSession
        assert VoiceSession is not None
    
    def test_session_state(self):
        from pyai.voice.session import VoiceSession, SessionState
        
        session = VoiceSession()
        assert session.state == SessionState.IDLE


class TestTranscriber:
    """Test speech-to-text."""
    
    def test_import_transcriber(self):
        from pyai.voice import Transcriber
        assert Transcriber is not None
    
    def test_transcription_result(self):
        from pyai.voice.transcription import TranscriptionResult
        
        result = TranscriptionResult(
            text="Hello world",
            language="en",
        )
        
        assert result.text == "Hello world"


class TestSynthesizer:
    """Test text-to-speech."""
    
    def test_import_synthesizer(self):
        from pyai.voice import Synthesizer
        assert Synthesizer is not None
    
    def test_synthesis_result(self):
        from pyai.voice.synthesis import SynthesisResult
        
        result = SynthesisResult(audio=b"\x00\x00")
        assert len(result.audio) == 2


# =============================================================================
# MAIN INIT EXPORTS TESTS
# =============================================================================

class TestMainExports:
    """Test main __init__.py exports for new features."""
    
    def test_export_tools(self):
        import pyai
        assert hasattr(pyai, 'tools')
        assert hasattr(pyai, 'Tool')
        assert hasattr(pyai, 'ToolDiscovery')
    
    def test_export_context_cache(self):
        import pyai
        assert hasattr(pyai, 'ContextCache')
        assert hasattr(pyai, 'cache_context')
    
    def test_export_multimodal(self):
        import pyai
        assert hasattr(pyai, 'multimodal')
        assert hasattr(pyai, 'Image')
        assert hasattr(pyai, 'Audio')
        assert hasattr(pyai, 'Video')
    
    def test_export_vectordb(self):
        import pyai
        assert hasattr(pyai, 'vectordb')
        assert hasattr(pyai, 'VectorStore')
        assert hasattr(pyai, 'MemoryVectorStore')
    
    def test_export_a2a(self):
        import pyai
        assert hasattr(pyai, 'a2a')
        assert hasattr(pyai, 'A2AServer')
        assert hasattr(pyai, 'A2AClient')
    
    def test_export_devui(self):
        import pyai
        assert hasattr(pyai, 'devui')
        assert hasattr(pyai, 'DevUI')
        assert hasattr(pyai, 'launch_ui')
    
    def test_export_voice(self):
        import pyai
        assert hasattr(pyai, 'voice')
        assert hasattr(pyai, 'VoiceSession')
        assert hasattr(pyai, 'AudioStream')
