# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Multimodal content composition.

Combine text, images, audio, and video into unified content.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from .image import Image
from .audio import Audio
from .video import Video


@dataclass
class ContentPart:
    """A single part of multimodal content.
    
    Attributes:
        type: Part type (text, image, audio, video)
        content: The actual content
        metadata: Additional metadata
    """
    type: Literal["text", "image", "audio", "video"]
    content: Union[str, Image, Audio, Video]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI API format."""
        if self.type == "text":
            return {"type": "text", "text": self.content}
        elif self.type == "image":
            return self.content.to_openai_format()
        elif self.type == "audio":
            return self.content.to_openai_format()
        else:
            # Video: extract frames and return as images
            raise ValueError("Video must be converted to frames first")
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic API format."""
        if self.type == "text":
            return {"type": "text", "text": self.content}
        elif self.type == "image":
            return self.content.to_anthropic_format()
        else:
            raise ValueError(f"Anthropic doesn't support {self.type} content directly")


@dataclass
class MultimodalContent:
    """Multimodal content container.
    
    Combines multiple content parts for LLM input.
    
    Example:
        content = MultimodalContent()
        content.add_text("Describe this image:")
        content.add_image(Image.from_file("photo.png"))
        
        # Convert to API format
        messages = content.to_openai_messages()
    """
    parts: List[ContentPart] = field(default_factory=list)
    
    def add_text(self, text: str, **metadata) -> "MultimodalContent":
        """Add text content.
        
        Args:
            text: Text content
            **metadata: Additional metadata
            
        Returns:
            Self for chaining
        """
        self.parts.append(ContentPart(
            type="text",
            content=text,
            metadata=metadata,
        ))
        return self
    
    def add_image(
        self,
        image: Union[Image, str],
        detail: str = "auto",
        **metadata
    ) -> "MultimodalContent":
        """Add image content.
        
        Args:
            image: Image object or path/URL
            detail: Detail level for OpenAI
            **metadata: Additional metadata
            
        Returns:
            Self for chaining
        """
        if isinstance(image, str):
            from .image import load_image
            image = load_image(image, detail=detail)
        
        self.parts.append(ContentPart(
            type="image",
            content=image,
            metadata=metadata,
        ))
        return self
    
    def add_audio(
        self,
        audio: Union[Audio, str],
        **metadata
    ) -> "MultimodalContent":
        """Add audio content.
        
        Args:
            audio: Audio object or file path
            **metadata: Additional metadata
            
        Returns:
            Self for chaining
        """
        if isinstance(audio, str):
            from .audio import load_audio
            audio = load_audio(audio)
        
        self.parts.append(ContentPart(
            type="audio",
            content=audio,
            metadata=metadata,
        ))
        return self
    
    def add_video(
        self,
        video: Union[Video, str],
        extract_frames: int = 0,
        **metadata
    ) -> "MultimodalContent":
        """Add video content.
        
        Args:
            video: Video object or file path
            extract_frames: Number of frames to extract as images
            **metadata: Additional metadata
            
        Returns:
            Self for chaining
        """
        if isinstance(video, str):
            from .video import load_video
            video = load_video(video)
        
        if extract_frames > 0:
            # Extract frames and add as images
            frames = video.extract_frames(count=extract_frames)
            for i, frame in enumerate(frames):
                self.parts.append(ContentPart(
                    type="image",
                    content=frame,
                    metadata={"from_video": True, "frame_index": i, **metadata},
                ))
        else:
            self.parts.append(ContentPart(
                type="video",
                content=video,
                metadata=metadata,
            ))
        return self
    
    def to_openai_content(self) -> List[Dict[str, Any]]:
        """Convert all parts to OpenAI content array.
        
        Returns:
            List of content parts for OpenAI API
        """
        return [part.to_openai_format() for part in self.parts]
    
    def to_openai_message(self, role: str = "user") -> Dict[str, Any]:
        """Convert to a complete OpenAI message.
        
        Args:
            role: Message role
            
        Returns:
            OpenAI message dict
        """
        return {
            "role": role,
            "content": self.to_openai_content(),
        }
    
    def to_anthropic_content(self) -> List[Dict[str, Any]]:
        """Convert all parts to Anthropic content array.
        
        Returns:
            List of content parts for Anthropic API
        """
        return [part.to_anthropic_format() for part in self.parts]
    
    def to_anthropic_message(self, role: str = "user") -> Dict[str, Any]:
        """Convert to a complete Anthropic message.
        
        Args:
            role: Message role
            
        Returns:
            Anthropic message dict
        """
        return {
            "role": role,
            "content": self.to_anthropic_content(),
        }
    
    def get_text(self) -> str:
        """Get concatenated text content."""
        texts = [
            part.content
            for part in self.parts
            if part.type == "text"
        ]
        return " ".join(texts)
    
    def get_images(self) -> List[Image]:
        """Get all image parts."""
        return [
            part.content
            for part in self.parts
            if part.type == "image"
        ]
    
    def __len__(self) -> int:
        return len(self.parts)


def create_content(
    text: Optional[str] = None,
    images: Optional[List[Union[Image, str]]] = None,
    audio: Optional[Union[Audio, str]] = None,
) -> MultimodalContent:
    """Create multimodal content from components.
    
    Args:
        text: Text content
        images: List of images or paths/URLs
        audio: Audio object or file path
        
    Returns:
        MultimodalContent instance
        
    Example:
        content = create_content(
            text="Describe these images:",
            images=["img1.png", "https://example.com/img2.jpg"]
        )
    """
    content = MultimodalContent()
    
    if text:
        content.add_text(text)
    
    if images:
        for img in images:
            content.add_image(img)
    
    if audio:
        content.add_audio(audio)
    
    return content
