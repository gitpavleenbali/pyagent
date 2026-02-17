# Transcription

The transcription module converts speech audio to text using AI models.

## Import

```python
from pyai.voice import Transcriber
from pyai.voice.transcription import TranscriptionResult
```

## Transcriber Class

### Constructor

```python
Transcriber(
    model: str = "whisper-1",    # Transcription model
    language: str = None,         # Language hint (auto-detect if None)
    response_format: str = "json" # Output format
)
```

## Basic Usage

### Transcribe File

```python
transcriber = Transcriber()

# Simple transcription
result = transcriber.transcribe("audio.wav")
print(result.text)
```

### With Options

```python
result = transcriber.transcribe(
    "meeting.mp3",
    language="en",
    timestamps=True,
    word_timestamps=True
)

# Access segments
for segment in result.segments:
    print(f"[{segment.start:.2f}s] {segment.text}")
```

### Transcribe Bytes

```python
with open("audio.wav", "rb") as f:
    audio_data = f.read()

result = transcriber.transcribe_bytes(
    audio_data,
    format="wav"
)
```

## TranscriptionResult

The result object contains:

```python
result.text          # Full transcription text
result.language      # Detected language
result.confidence    # Overall confidence score
result.duration      # Audio duration in seconds
result.segments      # List of segments with timestamps
result.words         # Word-level timestamps (if requested)
```

### Segment Structure

```python
segment.id        # Segment index
segment.start     # Start time (seconds)
segment.end       # End time (seconds)
segment.text      # Segment text
segment.confidence # Segment confidence
```

## Streaming Transcription

For real-time transcription:

```python
async def stream_transcribe(audio_stream):
    transcriber = Transcriber()
    
    async for result in transcriber.stream(audio_stream):
        print(f"Partial: {result.text}")
        
        if result.is_final:
            print(f"Final: {result.text}")
```

## Language Support

Supported languages include:
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Chinese (zh)
- Japanese (ja)
- Korean (ko)
- And 50+ more...

```python
# Force language
result = transcriber.transcribe(
    "audio.wav",
    language="es"  # Spanish
)

# Auto-detect
result = transcriber.transcribe("audio.wav")
print(f"Detected: {result.language}")
```

## Translation

Translate audio to English:

```python
# Transcribe + translate
result = transcriber.translate("french_audio.wav")
# Output is in English regardless of source language
```

## Batch Processing

```python
files = ["audio1.wav", "audio2.wav", "audio3.wav"]
results = transcriber.batch_transcribe(files)

for file, result in zip(files, results):
    print(f"{file}: {result.text}")
```

## Output Formats

```python
# JSON (default)
result = transcriber.transcribe("audio.wav", response_format="json")

# Plain text
text = transcriber.transcribe("audio.wav", response_format="text")

# SRT subtitles
srt = transcriber.transcribe("audio.wav", response_format="srt")

# VTT subtitles
vtt = transcriber.transcribe("audio.wav", response_format="vtt")
```

## See Also

- [Voice-Module](Voice-Module) - Module overview
- [VoiceSession](VoiceSession) - Real-time voice
- [Synthesis](Synthesis) - Text-to-speech
