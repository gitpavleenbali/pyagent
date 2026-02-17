# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Audio handling for multimodal agents.

Supports audio input/output for voice-enabled agents.
"""

import base64
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union


class AudioFormat(Enum):
    """Supported audio formats."""

    WAV = "wav"
    MP3 = "mp3"
    M4A = "m4a"
    OGG = "ogg"
    FLAC = "flac"
    WEBM = "webm"
    PCM = "pcm"


@dataclass
class Audio:
    """Audio data for multimodal input/output.

    Example:
        # From file
        audio = Audio.from_file("recording.mp3")

        # From base64
        audio = Audio.from_base64(data, AudioFormat.WAV)

        # Get bytes
        raw_data = audio.get_bytes()
    """

    data: str  # Base64 encoded audio data
    format: AudioFormat = AudioFormat.WAV
    sample_rate: Optional[int] = None
    channels: int = 1
    duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(
        cls, path: Union[str, Path], format: Optional[AudioFormat] = None, **metadata
    ) -> "Audio":
        """Create audio from file.

        Args:
            path: Path to audio file
            format: Audio format (auto-detected if not provided)
            **metadata: Additional metadata

        Returns:
            Audio instance
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        # Auto-detect format
        if format is None:
            suffix = path.suffix.lower().lstrip(".")
            try:
                format = AudioFormat(suffix)
            except ValueError:
                format = AudioFormat.WAV

        # Read and encode
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")

        return cls(
            data=data,
            format=format,
            metadata={"original_path": str(path), **metadata},
        )

    @classmethod
    def from_base64(
        cls,
        data: str,
        format: AudioFormat = AudioFormat.WAV,
        sample_rate: Optional[int] = None,
        channels: int = 1,
        **metadata,
    ) -> "Audio":
        """Create audio from base64 data.

        Args:
            data: Base64 encoded audio data
            format: Audio format
            sample_rate: Sample rate in Hz
            channels: Number of audio channels
            **metadata: Additional metadata

        Returns:
            Audio instance
        """
        return cls(
            data=data,
            format=format,
            sample_rate=sample_rate,
            channels=channels,
            metadata=metadata,
        )

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        format: AudioFormat = AudioFormat.WAV,
        sample_rate: Optional[int] = None,
        channels: int = 1,
        **metadata,
    ) -> "Audio":
        """Create audio from raw bytes.

        Args:
            data: Raw audio bytes
            format: Audio format
            sample_rate: Sample rate in Hz
            channels: Number of channels
            **metadata: Additional metadata

        Returns:
            Audio instance
        """
        encoded = base64.b64encode(data).decode("utf-8")
        return cls.from_base64(
            encoded,
            format=format,
            sample_rate=sample_rate,
            channels=channels,
            **metadata,
        )

    def get_bytes(self) -> bytes:
        """Get raw audio bytes."""
        return base64.b64decode(self.data)

    def save(self, path: Union[str, Path]) -> None:
        """Save audio to file.

        Args:
            path: Output file path
        """
        path = Path(path)
        with open(path, "wb") as f:
            f.write(self.get_bytes())

    @property
    def media_type(self) -> str:
        """Get MIME type for the audio."""
        format_to_mime = {
            AudioFormat.WAV: "audio/wav",
            AudioFormat.MP3: "audio/mpeg",
            AudioFormat.M4A: "audio/mp4",
            AudioFormat.OGG: "audio/ogg",
            AudioFormat.FLAC: "audio/flac",
            AudioFormat.WEBM: "audio/webm",
            AudioFormat.PCM: "audio/pcm",
        }
        return format_to_mime.get(self.format, "audio/wav")

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI API format for audio input."""
        return {
            "type": "input_audio",
            "input_audio": {
                "data": self.data,
                "format": self.format.value,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "data": self.data,
            "format": self.format.value,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Audio":
        """Create from dictionary."""
        format_val = data.get("format", "wav")
        if isinstance(format_val, str):
            format_val = AudioFormat(format_val)

        return cls(
            data=data["data"],
            format=format_val,
            sample_rate=data.get("sample_rate"),
            channels=data.get("channels", 1),
            duration_seconds=data.get("duration_seconds"),
            metadata=data.get("metadata", {}),
        )


def load_audio(
    source: Union[str, Path, bytes], format: Optional[AudioFormat] = None, **metadata
) -> Audio:
    """Load audio from various sources.

    Args:
        source: File path or bytes
        format: Audio format (auto-detected for files)
        **metadata: Additional metadata

    Returns:
        Audio instance
    """
    if isinstance(source, bytes):
        return Audio.from_bytes(source, format=format or AudioFormat.WAV, **metadata)

    return Audio.from_file(source, format=format, **metadata)
