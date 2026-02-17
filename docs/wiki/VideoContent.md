# VideoContent

The `Video` class handles video data for multimodal AI interactions.

## Import

```python
from pyai.multimodal import Video
```

## Creating Video

### From File

```python
video = Video.from_file("recording.mp4")
```

### From URL

```python
video = Video.from_url("https://example.com/video.mp4")
```

### From Bytes

```python
with open("video.mp4", "rb") as f:
    video = Video.from_bytes(f.read(), format="mp4")
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `duration` | float | Duration in seconds |
| `width` | int | Frame width in pixels |
| `height` | int | Frame height in pixels |
| `fps` | float | Frames per second |
| `format` | str | Video format |
| `size_bytes` | int | File size |
| `frame_count` | int | Total number of frames |

## Methods

### extract_frames()

Extract frames from video:

```python
# Extract frames at intervals
frames = video.extract_frames(interval=1.0)  # Every 1 second

# Extract specific number of frames
frames = video.extract_frames(count=10)  # 10 evenly spaced frames

# Extract at specific timestamps
frames = video.extract_frames(timestamps=[0.0, 5.0, 10.0])
```

### extract_audio()

Extract audio track:

```python
audio = video.extract_audio()
audio.save("audio.mp3")
```

### trim()

Trim video:

```python
# Trim to segment
trimmed = video.trim(start=10.0, end=30.0)

# First 60 seconds
trimmed = video.trim(end=60.0)
```

### resize()

Resize video:

```python
resized = video.resize(width=640, height=480)
```

### save()

Save to file:

```python
video.save("output.mp4")
video.save("output.webm", format="webm")
```

## Using with Agents

### Video Analysis

```python
from pyai import ask
from pyai.multimodal import Video

video = Video.from_file("presentation.mp4")

# Extract key frames for analysis
frames = video.extract_frames(count=5)

response = ask(
    "Describe what's happening in this video",
    images=frames
)
```

### With MultimodalContent

```python
from pyai.multimodal import MultimodalContent, Video

content = MultimodalContent()
content.add_text("Summarize this video lecture:")
content.add_video(Video.from_file("lecture.mp4"))

response = agent.run(content)
```

### Frame-by-Frame Analysis

```python
video = Video.from_file("surveillance.mp4")

for frame in video.extract_frames(interval=5.0):
    analysis = ask("What do you see?", images=[frame])
    print(f"Frame {frame.timestamp}s: {analysis}")
```

## Format Support

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| MP4 | ✅ | ✅ | Most common |
| MOV | ✅ | ✅ | QuickTime |
| WebM | ✅ | ✅ | Web optimized |
| AVI | ✅ | ✅ | Legacy format |
| MKV | ✅ | ❌ | Read only |
| GIF | ✅ | ✅ | Animated |

## Video Processing

### Get Thumbnail

```python
thumbnail = video.get_thumbnail(time=5.0)
thumbnail.save("thumbnail.jpg")
```

### Get Metadata

```python
metadata = video.get_metadata()
print(f"Duration: {metadata['duration']}")
print(f"Codec: {metadata['codec']}")
print(f"Bitrate: {metadata['bitrate']}")
```

### Convert Format

```python
# Convert to web-friendly format
web_video = video.convert(
    format="mp4",
    codec="h264",
    quality="medium"
)
```

## Provider Support

| Provider | Video Input | Notes |
|----------|-------------|-------|
| OpenAI GPT-4o | ✅ | Via frame extraction |
| Google Gemini | ✅ | Native video support |
| Anthropic Claude | ⚠️ | Via frame extraction |

## See Also

- [Multimodal-Module](Multimodal-Module) - Module overview
- [ImageContent](ImageContent) - Image handling
- [AudioContent](AudioContent) - Audio handling
