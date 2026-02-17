# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Voice Session - Real-time voice interactions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from .stream import AudioChunk, AudioStream, DuplexAudioStream


class SessionState(Enum):
    """Voice session state."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    CLOSED = "closed"


@dataclass
class VoiceMessage:
    """A message in the voice conversation."""

    role: str  # "user" or "assistant"
    text: str
    audio: Optional[AudioStream] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class VoiceSession:
    """Real-time voice session with an agent.

    Handles:
    - Speech-to-text (transcription)
    - Agent processing
    - Text-to-speech (synthesis)
    - Bidirectional streaming

    Example:
        from pyai.voice import VoiceSession

        session = VoiceSession(agent)
        await session.connect()

        # Send audio
        session.send_audio(audio_chunk)

        # Receive audio
        async for chunk in session.receive_audio():
            play(chunk)
    """

    def __init__(
        self,
        agent: Optional[Any] = None,
        transcriber: Optional["Transcriber"] = None,
        synthesizer: Optional["Synthesizer"] = None,
        model: str = "gpt-4o-realtime",
        voice: str = "alloy",
    ):
        """Initialize voice session.

        Args:
            agent: pyai Agent instance
            transcriber: Speech-to-text provider
            synthesizer: Text-to-speech provider
            model: Voice model to use
            voice: Voice for synthesis
        """
        self.agent = agent
        self.transcriber = transcriber
        self.synthesizer = synthesizer
        self.model = model
        self.voice = voice

        self._state = SessionState.IDLE
        self._messages: List[VoiceMessage] = []
        self._duplex = DuplexAudioStream()
        self._callbacks: Dict[str, List[Callable]] = {
            "transcript": [],
            "response": [],
            "audio": [],
            "state_change": [],
        }

    @property
    def state(self) -> SessionState:
        """Get current session state."""
        return self._state

    def _set_state(self, state: SessionState):
        """Update state and notify."""
        old_state = self._state
        self._state = state
        self._emit("state_change", old_state, state)

    async def connect(self):
        """Connect and start the session."""
        self._set_state(SessionState.IDLE)

    async def disconnect(self):
        """Disconnect and close the session."""
        self._set_state(SessionState.CLOSED)
        self._duplex.close()

    def send_audio(self, chunk: AudioChunk):
        """Send audio for processing.

        Args:
            chunk: Audio chunk to process
        """
        if self._state == SessionState.CLOSED:
            raise RuntimeError("Session is closed")

        self._duplex.send(chunk)

        if self._state == SessionState.IDLE:
            self._set_state(SessionState.LISTENING)

    def end_audio_input(self):
        """Signal end of audio input."""
        self._duplex.close_input()
        if self._state == SessionState.LISTENING:
            self._set_state(SessionState.PROCESSING)

    async def receive_audio(self) -> AsyncIterator[AudioChunk]:
        """Receive output audio.

        Yields:
            Audio chunks from agent response
        """
        async for chunk in self._duplex.receive():
            yield chunk

    async def process(self) -> VoiceMessage:
        """Process current audio input.

        Returns:
            Agent response message
        """
        self._set_state(SessionState.PROCESSING)

        # 1. Transcribe audio
        input_audio = self._duplex.input_stream.get_audio_bytes()
        transcript = ""

        if self.transcriber and input_audio:
            result = await self.transcriber.transcribe_async(input_audio)
            transcript = result.text
            self._emit("transcript", transcript)

        # 2. Process with agent
        response_text = ""
        if self.agent and transcript:
            result = self.agent.run(transcript)
            if hasattr(result, "output"):
                response_text = result.output
            elif isinstance(result, str):
                response_text = result
            else:
                response_text = str(result)

            self._emit("response", response_text)

        # 3. Synthesize audio
        self._set_state(SessionState.SPEAKING)

        if self.synthesizer and response_text:
            audio_result = await self.synthesizer.synthesize_async(
                response_text,
                voice=self.voice,
            )

            for chunk in audio_result.chunks:
                self._duplex.receive_chunk(chunk)
                self._emit("audio", chunk)

        self._duplex.close_output()
        self._set_state(SessionState.IDLE)

        # Create response message
        message = VoiceMessage(
            role="assistant",
            text=response_text,
        )
        self._messages.append(message)

        return message

    def on(self, event: str, callback: Callable):
        """Register event callback.

        Events:
            - transcript: User speech transcribed
            - response: Agent text response
            - audio: Output audio chunk
            - state_change: Session state changed

        Args:
            event: Event name
            callback: Callback function
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit(self, event: str, *args):
        """Emit event to callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args)
            except:
                pass

    @property
    def messages(self) -> List[VoiceMessage]:
        """Get conversation history."""
        return list(self._messages)

    def clear(self):
        """Clear conversation history."""
        self._messages.clear()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.disconnect()
