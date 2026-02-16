# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Video handling for multimodal agents.

Basic video support for frame extraction and video input.
"""

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class VideoFormat(Enum):
    """Supported video formats."""
    MP4 = "mp4"
    WEBM = "webm"
    MOV = "mov"
    AVI = "avi"
    MKV = "mkv"


@dataclass
class Video:
    """Video data for multimodal input.
    
    Videos can be processed by extracting frames as images.
    
    Example:
        # From file
        video = Video.from_file("demo.mp4")
        
        # Extract frames
        frames = video.extract_frames(count=5)
    """
    data: str  # Base64 encoded video data or URL
    format: VideoFormat = VideoFormat.MP4
    is_url: bool = False
    duration_seconds: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        format: Optional[VideoFormat] = None,
        **metadata
    ) -> "Video":
        """Create video from file.
        
        Args:
            path: Path to video file
            format: Video format (auto-detected if not provided)
            **metadata: Additional metadata
            
        Returns:
            Video instance
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        
        # Auto-detect format
        if format is None:
            suffix = path.suffix.lower().lstrip(".")
            try:
                format = VideoFormat(suffix)
            except ValueError:
                format = VideoFormat.MP4
        
        # Read and encode
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        
        return cls(
            data=data,
            format=format,
            is_url=False,
            metadata={"original_path": str(path), **metadata},
        )
    
    @classmethod
    def from_url(cls, url: str, **metadata) -> "Video":
        """Create video from URL.
        
        Args:
            url: Video URL
            **metadata: Additional metadata
            
        Returns:
            Video instance
        """
        return cls(
            data=url,
            is_url=True,
            metadata=metadata,
        )
    
    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        format: VideoFormat = VideoFormat.MP4,
        **metadata
    ) -> "Video":
        """Create video from raw bytes.
        
        Args:
            data: Raw video bytes
            format: Video format
            **metadata: Additional metadata
            
        Returns:
            Video instance
        """
        encoded = base64.b64encode(data).decode("utf-8")
        return cls(
            data=encoded,
            format=format,
            is_url=False,
            metadata=metadata,
        )
    
    def get_bytes(self) -> bytes:
        """Get raw video bytes.
        
        Note: Only works for non-URL videos.
        """
        if self.is_url:
            raise ValueError("Cannot get bytes from URL video")
        return base64.b64decode(self.data)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save video to file."""
        if self.is_url:
            raise ValueError("Cannot save URL video directly")
        
        path = Path(path)
        with open(path, "wb") as f:
            f.write(self.get_bytes())
    
    @property
    def media_type(self) -> str:
        """Get MIME type for the video."""
        format_to_mime = {
            VideoFormat.MP4: "video/mp4",
            VideoFormat.WEBM: "video/webm",
            VideoFormat.MOV: "video/quicktime",
            VideoFormat.AVI: "video/x-msvideo",
            VideoFormat.MKV: "video/x-matroska",
        }
        return format_to_mime.get(self.format, "video/mp4")
    
    def extract_frames(
        self,
        count: int = 5,
        uniform: bool = True
    ) -> List["Image"]:
        """Extract frames from video as images.
        
        Note: Requires opencv-python or moviepy installed.
        
        Args:
            count: Number of frames to extract
            uniform: Extract uniformly spaced frames
            
        Returns:
            List of Image objects
        """
        # Lazy import to avoid dependency
        try:
            import cv2
            import numpy as np
        except ImportError:
            raise ImportError(
                "opencv-python required for frame extraction. "
                "Install with: pip install opencv-python"
            )
        
        from .image import Image
        
        # Write to temp file if needed
        if self.is_url:
            raise ValueError("Cannot extract frames from URL video directly")
        
        import tempfile
        import os
        
        frames = []
        
        with tempfile.NamedTemporaryFile(
            suffix=f".{self.format.value}",
            delete=False
        ) as tmp:
            tmp.write(self.get_bytes())
            tmp_path = tmp.name
        
        try:
            cap = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                return []
            
            # Calculate frame indices
            if uniform and total_frames > count:
                indices = np.linspace(0, total_frames - 1, count, dtype=int)
            else:
                indices = list(range(min(count, total_frames)))
            
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Encode as PNG
                    _, buffer = cv2.imencode('.png', frame_rgb)
                    img_data = base64.b64encode(buffer).decode('utf-8')
                    frames.append(Image.from_base64(
                        img_data,
                        media_type="image/png",
                        frame_index=int(i),
                    ))
            
            cap.release()
        finally:
            os.unlink(tmp_path)
        
        return frames
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "data": self.data,
            "format": self.format.value,
            "is_url": self.is_url,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Video":
        """Create from dictionary."""
        format_val = data.get("format", "mp4")
        if isinstance(format_val, str):
            format_val = VideoFormat(format_val)
        
        return cls(
            data=data["data"],
            format=format_val,
            is_url=data.get("is_url", False),
            duration_seconds=data.get("duration_seconds"),
            fps=data.get("fps"),
            width=data.get("width"),
            height=data.get("height"),
            metadata=data.get("metadata", {}),
        )


def load_video(
    source: Union[str, Path, bytes],
    format: Optional[VideoFormat] = None,
    **metadata
) -> Video:
    """Load video from various sources.
    
    Args:
        source: File path, URL, or bytes
        format: Video format (auto-detected for files)
        **metadata: Additional metadata
        
    Returns:
        Video instance
    """
    if isinstance(source, bytes):
        return Video.from_bytes(source, format=format or VideoFormat.MP4, **metadata)
    
    source_str = str(source)
    
    # Check if it's a URL
    if source_str.startswith(("http://", "https://")):
        return Video.from_url(source_str, **metadata)
    
    return Video.from_file(source_str, format=format, **metadata)
