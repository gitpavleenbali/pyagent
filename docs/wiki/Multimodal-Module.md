# Multimodal Module

The Multimodal module enables AI agents to process and generate content across multiple modalities: images, audio, and video.

## Overview

```python
from pyagent.multimodal import Image, Audio, Video, MultimodalContent
```

## Key Components

| Component | Description |
|-----------|-------------|
| [ImageContent](ImageContent) | Image processing and analysis |
| [AudioContent](AudioContent) | Audio file handling |
| [VideoContent](VideoContent) | Video processing |
| MultimodalContent | Mixed content container |

## Quick Start

### Image Analysis

```python
from pyagent import ask
from pyagent.multimodal import Image

# Analyze an image
image = Image.from_file("photo.jpg")
response = ask("What's in this image?", images=[image])
print(response)
```

### Multiple Images

```python
images = [
    Image.from_file("before.jpg"),
    Image.from_file("after.jpg")
]

response = ask(
    "Compare these two images and describe the differences",
    images=images
)
```

### From URL

```python
image = Image.from_url("https://example.com/image.jpg")
response = ask("Describe this image", images=[image])
```

### Base64 Encoded

```python
import base64

with open("image.png", "rb") as f:
    data = base64.b64encode(f.read()).decode()

image = Image.from_base64(data, media_type="image/png")
```

## MultimodalContent

Combine multiple types of content:

```python
from pyagent.multimodal import MultimodalContent, Image, Audio

content = MultimodalContent()
content.add_text("Please analyze this meeting recording and slides:")
content.add_image(Image.from_file("slides.png"))
content.add_audio(Audio.from_file("meeting.mp3"))

response = agent.run(content)
```

## With Agents

```python
from pyagent import Agent
from pyagent.multimodal import Image

agent = Agent(
    name="ImageAnalyzer",
    instructions="You are an expert at analyzing images.",
    model="gpt-4o"  # Vision-capable model
)

image = Image.from_file("diagram.png")
result = agent.run("Explain this diagram", images=[image])
```

## Supported Formats

### Images
- PNG, JPEG, GIF, WebP
- Max size varies by model (typically 20MB)
- Auto-resizing available

### Audio
- MP3, WAV, M4A, FLAC, OGG
- Transcription integration

### Video
- MP4, MOV, WebM
- Frame extraction for analysis

## Image Processing

```python
from pyagent.multimodal import Image

image = Image.from_file("large_photo.jpg")

# Resize for API limits
image = image.resize(max_width=1024, max_height=1024)

# Convert format
image = image.convert(format="jpeg", quality=85)

# Get dimensions
print(f"Size: {image.width}x{image.height}")
```

## Provider Support

| Provider | Images | Audio | Video |
|----------|--------|-------|-------|
| OpenAI GPT-4o | ✅ | ✅ | ✅ |
| Anthropic Claude 3 | ✅ | ❌ | ❌ |
| Google Gemini | ✅ | ✅ | ✅ |

## See Also

- [ImageContent](ImageContent) - Image handling
- [AudioContent](AudioContent) - Audio handling
- [VideoContent](VideoContent) - Video handling
