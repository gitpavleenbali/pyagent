# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Image handling for multimodal agents.

Like OpenAI's vision API support.
"""

import base64
import io
import mimetypes
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.parse import urlparse


class ImageSource(Enum):
    """Source type for an image."""
    BASE64 = "base64"
    URL = "url"
    FILE = "file"


@dataclass
class Image:
    """An image for multimodal input.
    
    Supports images from:
    - URLs (direct linking)
    - Base64 encoded data
    - Local files
    
    Example:
        # From URL
        img = Image.from_url("https://example.com/photo.jpg")
        
        # From file
        img = Image.from_file("./image.png")
        
        # From base64
        img = Image.from_base64(data, "image/png")
        
        # Convert to API format
        api_format = img.to_openai_format()
    """
    source: ImageSource
    data: str  # URL, base64 data, or file path
    media_type: str = "image/png"
    detail: str = "auto"  # OpenAI detail level: "low", "high", "auto"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_url(
        cls,
        url: str,
        detail: str = "auto",
        **metadata
    ) -> "Image":
        """Create image from URL.
        
        Args:
            url: Image URL
            detail: Detail level for processing
            **metadata: Additional metadata
            
        Returns:
            Image instance
        """
        # Infer media type from URL
        media_type = cls._infer_media_type(url)
        
        return cls(
            source=ImageSource.URL,
            data=url,
            media_type=media_type,
            detail=detail,
            metadata=metadata,
        )
    
    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        detail: str = "auto",
        **metadata
    ) -> "Image":
        """Create image from local file.
        
        Args:
            path: Path to image file
            detail: Detail level for processing
            **metadata: Additional metadata
            
        Returns:
            Image instance
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        
        # Read and encode
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        
        # Infer media type
        media_type = cls._infer_media_type(str(path))
        
        return cls(
            source=ImageSource.BASE64,
            data=data,
            media_type=media_type,
            detail=detail,
            metadata={"original_path": str(path), **metadata},
        )
    
    @classmethod
    def from_base64(
        cls,
        data: str,
        media_type: str = "image/png",
        detail: str = "auto",
        **metadata
    ) -> "Image":
        """Create image from base64 data.
        
        Args:
            data: Base64 encoded image data
            media_type: MIME type of the image
            detail: Detail level for processing
            **metadata: Additional metadata
            
        Returns:
            Image instance
        """
        return cls(
            source=ImageSource.BASE64,
            data=data,
            media_type=media_type,
            detail=detail,
            metadata=metadata,
        )
    
    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        media_type: str = "image/png",
        detail: str = "auto",
        **metadata
    ) -> "Image":
        """Create image from bytes.
        
        Args:
            data: Raw image bytes
            media_type: MIME type of the image
            detail: Detail level
            **metadata: Additional metadata
            
        Returns:
            Image instance
        """
        encoded = base64.b64encode(data).decode("utf-8")
        return cls.from_base64(encoded, media_type, detail, **metadata)
    
    @classmethod
    def from_pil(
        cls,
        image: Any,
        format: str = "PNG",
        detail: str = "auto",
        **metadata
    ) -> "Image":
        """Create from PIL Image.
        
        Args:
            image: PIL Image object
            format: Output format (PNG, JPEG, etc.)
            detail: Detail level
            **metadata: Additional metadata
            
        Returns:
            Image instance
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        media_type = f"image/{format.lower()}"
        
        return cls.from_base64(data, media_type, detail, **metadata)
    
    @staticmethod
    def _infer_media_type(path_or_url: str) -> str:
        """Infer media type from path or URL."""
        # Try to get from path extension
        parsed = urlparse(path_or_url)
        path = parsed.path if parsed.scheme else path_or_url
        
        mime_type, _ = mimetypes.guess_type(path)
        
        if mime_type and mime_type.startswith("image/"):
            return mime_type
        
        # Default to PNG
        return "image/png"
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI API format.
        
        Returns:
            Dict for use in OpenAI messages
        """
        if self.source == ImageSource.URL:
            return {
                "type": "image_url",
                "image_url": {
                    "url": self.data,
                    "detail": self.detail,
                }
            }
        else:
            # Base64 format
            data_url = f"data:{self.media_type};base64,{self.data}"
            return {
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                    "detail": self.detail,
                }
            }
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic API format.
        
        Returns:
            Dict for use in Anthropic messages
        """
        if self.source == ImageSource.URL:
            return {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": self.data,
                }
            }
        else:
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": self.media_type,
                    "data": self.data,
                }
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source.value,
            "data": self.data,
            "media_type": self.media_type,
            "detail": self.detail,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Image":
        """Create from dictionary."""
        source = data.get("source", "base64")
        if isinstance(source, str):
            source = ImageSource(source)
        
        return cls(
            source=source,
            data=data["data"],
            media_type=data.get("media_type", "image/png"),
            detail=data.get("detail", "auto"),
            metadata=data.get("metadata", {}),
        )
    
    def get_bytes(self) -> bytes:
        """Get raw image bytes.
        
        Note: Only works for base64 images.
        
        Returns:
            Raw bytes of the image
        """
        if self.source != ImageSource.BASE64:
            raise ValueError("Can only get bytes from base64 images")
        return base64.b64decode(self.data)


def image_to_base64(path: Union[str, Path]) -> str:
    """Convert an image file to base64 string.
    
    Args:
        path: Path to image file
        
    Returns:
        Base64 encoded string
    """
    path = Path(path)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_image(
    source: Union[str, Path, bytes],
    detail: str = "auto",
    **metadata
) -> Image:
    """Load an image from various sources.
    
    Args:
        source: URL, file path, or bytes
        detail: Detail level for processing
        **metadata: Additional metadata
        
    Returns:
        Image instance
    """
    if isinstance(source, bytes):
        return Image.from_bytes(source, detail=detail, **metadata)
    
    source_str = str(source)
    
    # Check if it's a URL
    if source_str.startswith(("http://", "https://", "data:")):
        return Image.from_url(source_str, detail=detail, **metadata)
    
    # Treat as file path
    return Image.from_file(source_str, detail=detail, **metadata)
