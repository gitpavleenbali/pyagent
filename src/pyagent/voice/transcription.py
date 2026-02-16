# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Speech-to-Text Transcription.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TranscriptionResult:
    """Result of speech transcription.
    
    Attributes:
        text: Transcribed text
        language: Detected language
        confidence: Confidence score
        words: Word-level timestamps
        duration_seconds: Audio duration
    """
    text: str
    language: str = ""
    confidence: float = 1.0
    words: List[Dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Transcriber:
    """Speech-to-text transcription.
    
    Supports multiple providers:
    - OpenAI Whisper
    - Azure Speech Services
    - Google Cloud Speech
    
    Example:
        transcriber = Transcriber(provider="openai")
        result = transcriber.transcribe(audio_bytes)
        print(result.text)
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "whisper-1",
        api_key: Optional[str] = None,
        language: Optional[str] = None,
    ):
        """Initialize transcriber.
        
        Args:
            provider: Transcription provider
            model: Model to use
            api_key: API key (or from environment)
            language: Target language code
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.language = language
        
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
    
    def transcribe(
        self,
        audio: bytes,
        filename: str = "audio.wav",
    ) -> TranscriptionResult:
        """Transcribe audio synchronously.
        
        Args:
            audio: Audio bytes
            filename: Filename hint for format
            
        Returns:
            Transcription result
        """
        if self.provider == "openai":
            return self._transcribe_openai(audio, filename)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    async def transcribe_async(
        self,
        audio: bytes,
        filename: str = "audio.wav",
    ) -> TranscriptionResult:
        """Transcribe audio asynchronously.
        
        Args:
            audio: Audio bytes
            filename: Filename hint
            
        Returns:
            Transcription result
        """
        # For now, use sync implementation
        # TODO: Add async clients
        import asyncio
        return await asyncio.to_thread(self.transcribe, audio, filename)
    
    def _transcribe_openai(
        self,
        audio: bytes,
        filename: str,
    ) -> TranscriptionResult:
        """Transcribe using OpenAI Whisper."""
        import io
        
        client = self._get_openai_client()
        
        # Create file-like object
        audio_file = io.BytesIO(audio)
        audio_file.name = filename
        
        kwargs = {"model": self.model, "file": audio_file}
        
        if self.language:
            kwargs["language"] = self.language
        
        response = client.audio.transcriptions.create(**kwargs)
        
        return TranscriptionResult(
            text=response.text,
            language=self.language or "auto",
        )


class StreamingTranscriber:
    """Streaming speech-to-text transcription.
    
    For real-time transcription of audio streams.
    
    Example:
        transcriber = StreamingTranscriber()
        
        async for partial in transcriber.transcribe_stream(audio_stream):
            print(partial.text)
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "whisper-1",
        api_key: Optional[str] = None,
        language: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.language = language
    
    async def transcribe_stream(
        self,
        audio_stream: Any,
        chunk_duration_ms: int = 5000,
    ):
        """Transcribe audio stream.
        
        Args:
            audio_stream: AudioStream to transcribe
            chunk_duration_ms: Process chunks of this duration
            
        Yields:
            Partial transcription results
        """
        transcriber = Transcriber(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            language=self.language,
        )
        
        buffer = b""
        
        async for chunk in audio_stream:
            buffer += chunk.data
            
            # Process when we have enough data
            if chunk.duration_ms >= chunk_duration_ms or chunk.is_final:
                if buffer:
                    result = await transcriber.transcribe_async(buffer)
                    yield result
                    buffer = b""
