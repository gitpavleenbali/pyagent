# AudioContent

The `Audio` class handles audio data for multimodal AI interactions.

## Import

```python
from pyai.multimodal import Audio
```

## Creating Audio

### From File

```python
audio = Audio.from_file("recording.mp3")
```

### From URL

```python
audio = Audio.from_url("https://example.com/audio.wav")
```

### From Bytes

```python
with open("audio.wav", "rb") as f:
    audio = Audio.from_bytes(f.read(), format="wav")
```

### From NumPy Array

```python
import numpy as np

# Generate audio data
samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
audio = Audio.from_numpy(samples, sample_rate=44100)
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `duration` | float | Duration in seconds |
| `sample_rate` | int | Samples per second |
| `channels` | int | Number of channels |
| `format` | str | Audio format |
| `size_bytes` | int | File size |

## Methods

### convert()

Convert to different format:

```python
# Convert to MP3
mp3_audio = audio.convert(format="mp3", bitrate=192)

# Convert to WAV
wav_audio = audio.convert(format="wav")

# Convert sample rate
resampled = audio.convert(sample_rate=16000)
```

### trim()

Trim audio:

```python
# Trim to specific duration
trimmed = audio.trim(start=0.0, end=30.0)  # First 30 seconds

# Trim from start
trimmed = audio.trim(start=5.0)  # Skip first 5 seconds
```

### split()

Split into segments:

```python
# Split into 30-second chunks
segments = audio.split(segment_duration=30.0)
```

### save()

Save to file:

```python
audio.save("output.mp3")
audio.save("output.wav", format="wav")
```

### transcribe()

Transcribe audio to text:

```python
text = audio.transcribe()
print(text)

# With options
result = audio.transcribe(
    language="en",
    timestamps=True
)
```

## Using with Agents

### With Multimodal Content

```python
from pyai.multimodal import MultimodalContent, Audio

content = MultimodalContent()
content.add_text("Please transcribe and summarize this audio:")
content.add_audio(Audio.from_file("meeting.mp3"))

response = agent.run(content)
```

### Audio Analysis

```python
from pyai import ask
from pyai.multimodal import Audio

audio = Audio.from_file("speech.wav")
response = ask(
    "What language is being spoken and what is the topic?",
    audio=[audio]
)
```

## Format Support

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| MP3 | ✅ | ✅ | Most common |
| WAV | ✅ | ✅ | Uncompressed |
| M4A | ✅ | ✅ | AAC encoded |
| FLAC | ✅ | ✅ | Lossless |
| OGG | ✅ | ✅ | Open format |
| WebM | ✅ | ✅ | Web optimized |

## Audio Processing

### Volume Adjustment

```python
# Increase volume
louder = audio.adjust_volume(gain_db=3.0)

# Decrease volume
quieter = audio.adjust_volume(gain_db=-3.0)

# Normalize
normalized = audio.normalize()
```

### Channel Operations

```python
# Convert to mono
mono = audio.to_mono()

# Convert to stereo
stereo = audio.to_stereo()

# Extract channel
left_channel = audio.get_channel(0)
```

## See Also

- [Multimodal-Module](Multimodal-Module) - Module overview
- [ImageContent](ImageContent) - Image handling
- [VideoContent](VideoContent) - Video handling
- [Voice-Module](Voice-Module) - Real-time voice
