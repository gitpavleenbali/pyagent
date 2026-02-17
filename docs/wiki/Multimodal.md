# Multimodal

Process images, audio, and video with AI agents.

> **See [[Multimodal-Module]] for full documentation.**

## Quick Start

```python
from pyai.multimodal import ImageContent, AudioContent

# Image analysis
image = ImageContent.from_file("photo.jpg")
description = image.describe()

# Audio transcription
audio = AudioContent.from_file("recording.mp3")
text = audio.transcribe()
```

## Features

- Image understanding and analysis
- Audio transcription
- Video frame analysis
- Multi-modal conversations
- Format conversion

## Supported Formats

| Type | Formats |
|------|---------|
| Image | PNG, JPG, GIF, WebP |
| Audio | MP3, WAV, M4A, FLAC |
| Video | MP4, MOV, AVI |

## Related Pages

- [[Multimodal-Module]] - Full module documentation
- [[ImageContent]] - Image processing
- [[AudioContent]] - Audio processing
- [[VideoContent]] - Video processing
