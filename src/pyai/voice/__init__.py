# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
pyai Voice Streaming

Bidirectional voice streaming for real-time voice interactions.
Like OpenAI Realtime API.

Example:
    from pyai.voice import VoiceSession, AudioStream

    # Create voice session
    session = VoiceSession(agent)

    # Start streaming
    async with session.stream() as stream:
        async for audio in stream:
            play_audio(audio)
"""

from .session import VoiceSession
from .stream import AudioChunk, AudioStream
from .synthesis import SynthesisResult, Synthesizer
from .transcription import Transcriber, TranscriptionResult

__all__ = [
    "VoiceSession",
    "AudioStream",
    "AudioChunk",
    "Transcriber",
    "TranscriptionResult",
    "Synthesizer",
    "SynthesisResult",
]
