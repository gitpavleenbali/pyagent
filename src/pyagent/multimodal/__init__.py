# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
PyAgent Multimodal Module

Support for images, audio, and video in agent interactions.
Like OpenAI's multimodal API and Google ADK's multimodal support.

Example:
    from pyagent.multimodal import Image, image_to_base64
    from pyagent import Agent
    
    # Create image from URL
    img = Image.from_url("https://example.com/image.jpg")
    
    # Or from file
    img = Image.from_file("./photo.png")
    
    # Use in agent
    agent = Agent(model="gpt-4o")
    result = agent.run("Describe this image", images=[img])
"""

from .image import (
    Image,
    ImageSource,
    image_to_base64,
    load_image,
)

from .audio import (
    Audio,
    AudioFormat,
    load_audio,
)

from .video import (
    Video,
    load_video,
)

from .content import (
    ContentPart,
    MultimodalContent,
    create_content,
)

__all__ = [
    # Image
    "Image",
    "ImageSource",
    "image_to_base64",
    "load_image",
    # Audio
    "Audio",
    "AudioFormat",
    "load_audio",
    # Video
    "Video",
    "load_video",
    # Content
    "ContentPart",
    "MultimodalContent",
    "create_content",
]
