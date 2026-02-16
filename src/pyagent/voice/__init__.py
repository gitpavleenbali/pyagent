# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
PyAgent Voice Streaming

Bidirectional voice streaming for real-time voice interactions.
Like OpenAI Realtime API.

Example:
    from pyagent.voice import VoiceSession, AudioStream
    
    # Create voice session
    session = VoiceSession(agent)
    
    # Start streaming
    async with session.stream() as stream:
        async for audio in stream:
            play_audio(audio)
"""

from .session import VoiceSession
from .stream import AudioStream, AudioChunk
from .transcription import Transcriber, TranscriptionResult
from .synthesis import Synthesizer, SynthesisResult

__all__ = [
    "VoiceSession",
    "AudioStream",
    "AudioChunk",
    "Transcriber",
    "TranscriptionResult",
    "Synthesizer",
    "SynthesisResult",
]
