# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Audio streaming types and utilities.
"""

import base64
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
import asyncio


class AudioFormat(Enum):
    """Supported audio formats for streaming."""
    PCM16 = "pcm16"  # Raw PCM 16-bit
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"


@dataclass
class AudioChunk:
    """A chunk of audio data in a stream.
    
    Attributes:
        data: Audio data (raw bytes or base64)
        format: Audio format
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        is_final: Whether this is the last chunk
        timestamp: Chunk timestamp
    """
    data: bytes
    format: AudioFormat = AudioFormat.PCM16
    sample_rate: int = 16000
    channels: int = 1
    is_final: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_base64(self) -> str:
        """Encode audio data as base64."""
        return base64.b64encode(self.data).decode("utf-8")
    
    @classmethod
    def from_base64(
        cls,
        data: str,
        format: AudioFormat = AudioFormat.PCM16,
        **kwargs
    ) -> "AudioChunk":
        """Create chunk from base64 data."""
        return cls(
            data=base64.b64decode(data),
            format=format,
            **kwargs,
        )
    
    @property
    def duration_ms(self) -> float:
        """Estimate duration of this chunk in milliseconds."""
        if self.format == AudioFormat.PCM16:
            # 2 bytes per sample for 16-bit audio
            samples = len(self.data) // (2 * self.channels)
            return (samples / self.sample_rate) * 1000
        return 0.0


class AudioStream:
    """A bidirectional audio stream.
    
    Supports both synchronous and asynchronous iteration.
    
    Example:
        stream = AudioStream()
        
        # Add chunks
        stream.add(chunk)
        stream.add(chunk2)
        stream.close()
        
        # Iterate
        for chunk in stream:
            process(chunk)
    """
    
    def __init__(
        self,
        format: AudioFormat = AudioFormat.PCM16,
        sample_rate: int = 16000,
        channels: int = 1,
    ):
        self.format = format
        self.sample_rate = sample_rate
        self.channels = channels
        
        self._chunks: List[AudioChunk] = []
        self._closed = False
        self._position = 0
        self._async_queue: Optional[asyncio.Queue] = None
        self._event = asyncio.Event() if asyncio.get_event_loop().is_running() else None
    
    def add(self, chunk: AudioChunk):
        """Add a chunk to the stream.
        
        Args:
            chunk: Audio chunk to add
        """
        if self._closed:
            raise RuntimeError("Stream is closed")
        
        self._chunks.append(chunk)
        
        if self._async_queue:
            try:
                self._async_queue.put_nowait(chunk)
            except:
                pass
    
    def close(self):
        """Close the stream."""
        self._closed = True
        
        if self._async_queue:
            try:
                self._async_queue.put_nowait(None)  # Sentinel
            except:
                pass
    
    @property
    def is_closed(self) -> bool:
        """Check if stream is closed."""
        return self._closed
    
    def __iter__(self) -> Iterator[AudioChunk]:
        """Synchronous iteration."""
        while True:
            if self._position < len(self._chunks):
                chunk = self._chunks[self._position]
                self._position += 1
                yield chunk
            elif self._closed:
                break
            else:
                # Wait for more data (simple polling)
                import time
                time.sleep(0.01)
    
    async def __aiter__(self) -> AsyncIterator[AudioChunk]:
        """Asynchronous iteration."""
        if self._async_queue is None:
            self._async_queue = asyncio.Queue()
            # Add existing chunks
            for chunk in self._chunks:
                await self._async_queue.put(chunk)
        
        while True:
            item = await self._async_queue.get()
            if item is None:  # Sentinel for close
                break
            yield item
    
    def get_all(self) -> List[AudioChunk]:
        """Get all chunks."""
        return list(self._chunks)
    
    def get_audio_bytes(self) -> bytes:
        """Concatenate all audio data."""
        return b"".join(c.data for c in self._chunks)
    
    @property
    def total_duration_ms(self) -> float:
        """Get total duration of all chunks."""
        return sum(c.duration_ms for c in self._chunks)
    
    def __len__(self) -> int:
        return len(self._chunks)


class DuplexAudioStream:
    """Full-duplex audio stream for bidirectional communication.
    
    Supports simultaneous input and output streams.
    
    Example:
        duplex = DuplexAudioStream()
        
        # Send audio
        duplex.send(audio_chunk)
        
        # Receive audio
        async for chunk in duplex.receive():
            play(chunk)
    """
    
    def __init__(
        self,
        format: AudioFormat = AudioFormat.PCM16,
        sample_rate: int = 16000,
        channels: int = 1,
    ):
        self.input_stream = AudioStream(format, sample_rate, channels)
        self.output_stream = AudioStream(format, sample_rate, channels)
    
    def send(self, chunk: AudioChunk):
        """Send audio chunk (input)."""
        self.input_stream.add(chunk)
    
    def receive_chunk(self, chunk: AudioChunk):
        """Add received audio chunk (output)."""
        self.output_stream.add(chunk)
    
    async def receive(self) -> AsyncIterator[AudioChunk]:
        """Iterate over received audio."""
        async for chunk in self.output_stream:
            yield chunk
    
    def close_input(self):
        """Close input stream."""
        self.input_stream.close()
    
    def close_output(self):
        """Close output stream."""
        self.output_stream.close()
    
    def close(self):
        """Close both streams."""
        self.close_input()
        self.close_output()
