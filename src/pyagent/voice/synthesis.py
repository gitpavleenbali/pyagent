# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Text-to-Speech Synthesis.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .stream import AudioStream, AudioChunk, AudioFormat


@dataclass
class SynthesisResult:
    """Result of speech synthesis.
    
    Attributes:
        audio: Complete audio data
        chunks: Audio in chunks for streaming
        format: Audio format
        sample_rate: Sample rate
        duration_seconds: Audio duration
    """
    audio: bytes
    chunks: List[AudioChunk] = field(default_factory=list)
    format: AudioFormat = AudioFormat.MP3
    sample_rate: int = 24000
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_stream(self) -> AudioStream:
        """Convert to audio stream."""
        stream = AudioStream(
            format=self.format,
            sample_rate=self.sample_rate,
        )
        for chunk in self.chunks:
            stream.add(chunk)
        stream.close()
        return stream


class Synthesizer:
    """Text-to-speech synthesis.
    
    Supports multiple providers:
    - OpenAI TTS
    - Azure Speech Services
    - Google Cloud Text-to-Speech
    
    Example:
        synth = Synthesizer(provider="openai", voice="alloy")
        result = synth.synthesize("Hello, world!")
        play_audio(result.audio)
    """
    
    # Available voices by provider
    OPENAI_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "tts-1",
        voice: str = "alloy",
        api_key: Optional[str] = None,
        response_format: str = "mp3",
        speed: float = 1.0,
    ):
        """Initialize synthesizer.
        
        Args:
            provider: TTS provider
            model: Model to use
            voice: Voice ID
            api_key: API key (or from environment)
            response_format: Output format
            speed: Speech speed (0.25 to 4.0)
        """
        self.provider = provider
        self.model = model
        self.voice = voice
        self.api_key = api_key
        self.response_format = response_format
        self.speed = speed
        
        self._client = None
    
    def _get_openai_client(self):
        """Get OpenAI client."""
        if self._client:
            return self._client
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required")
        
        self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> SynthesisResult:
        """Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            voice: Override voice
            speed: Override speed
            
        Returns:
            Synthesis result with audio
        """
        if self.provider == "openai":
            return self._synthesize_openai(text, voice, speed)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    async def synthesize_async(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> SynthesisResult:
        """Synthesize speech asynchronously.
        
        Args:
            text: Text to synthesize
            voice: Override voice
            speed: Override speed
            
        Returns:
            Synthesis result
        """
        import asyncio
        return await asyncio.to_thread(self.synthesize, text, voice, speed)
    
    def _synthesize_openai(
        self,
        text: str,
        voice: Optional[str],
        speed: Optional[float],
    ) -> SynthesisResult:
        """Synthesize using OpenAI TTS."""
        client = self._get_openai_client()
        
        response = client.audio.speech.create(
            model=self.model,
            voice=voice or self.voice,
            input=text,
            response_format=self.response_format,
            speed=speed or self.speed,
        )
        
        # Get audio bytes
        audio_bytes = response.content
        
        # Determine format
        format_map = {
            "mp3": AudioFormat.MP3,
            "wav": AudioFormat.WAV,
            "opus": AudioFormat.OGG,
            "aac": AudioFormat.MP3,
            "flac": AudioFormat.PCM16,
            "pcm": AudioFormat.PCM16,
        }
        audio_format = format_map.get(self.response_format, AudioFormat.MP3)
        
        # Create chunks (single chunk for non-streaming)
        chunk = AudioChunk(
            data=audio_bytes,
            format=audio_format,
            sample_rate=24000,
            is_final=True,
        )
        
        return SynthesisResult(
            audio=audio_bytes,
            chunks=[chunk],
            format=audio_format,
            sample_rate=24000,
        )
    
    def stream(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> AudioStream:
        """Stream synthesized audio.
        
        Args:
            text: Text to synthesize
            voice: Override voice
            
        Returns:
            Audio stream
        """
        result = self.synthesize(text, voice)
        return result.to_stream()


class StreamingSynthesizer:
    """Streaming text-to-speech synthesis.
    
    For real-time synthesis with chunked output.
    
    Example:
        synth = StreamingSynthesizer()
        
        async for chunk in synth.synthesize_stream("Hello world"):
            play_chunk(chunk)
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "tts-1",
        voice: str = "alloy",
        api_key: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model
        self.voice = voice
        self.api_key = api_key
    
    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        chunk_size: int = 4096,
    ):
        """Stream synthesized audio.
        
        Args:
            text: Text to synthesize
            voice: Override voice
            chunk_size: Bytes per chunk
            
        Yields:
            Audio chunks
        """
        synth = Synthesizer(
            provider=self.provider,
            model=self.model,
            voice=voice or self.voice,
            api_key=self.api_key,
        )
        
        result = await synth.synthesize_async(text, voice)
        
        # Yield in chunks
        audio = result.audio
        for i in range(0, len(audio), chunk_size):
            chunk_data = audio[i:i + chunk_size]
            is_final = i + chunk_size >= len(audio)
            
            yield AudioChunk(
                data=chunk_data,
                format=result.format,
                sample_rate=result.sample_rate,
                is_final=is_final,
            )
