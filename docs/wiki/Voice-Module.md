# Voice Module

The Voice module enables real-time voice interactions with AI agents, including speech-to-text, text-to-speech, and streaming audio processing.

## Overview

```python
from pyai.voice import VoiceSession, Transcriber, Synthesizer
from pyai.voice.stream import AudioStream, AudioChunk
```

## Key Components

| Component | Description |
|-----------|-------------|
| [VoiceSession](VoiceSession) | Manages voice conversation sessions |
| [Transcription](Transcription) | Speech-to-text conversion |
| [Synthesis](Synthesis) | Text-to-speech generation |
| AudioStream | Real-time audio streaming |

## Quick Start

### Basic Voice Interaction

```python
from pyai.voice import VoiceSession

# Create voice session
session = VoiceSession(
    model="gpt-4o-realtime",
    voice="alloy"
)

# Start conversation
async with session.connect() as voice:
    # Send audio
    await voice.send_audio(audio_data)
    
    # Receive response
    async for chunk in voice.receive():
        play_audio(chunk)
```

### Transcription Only

```python
from pyai.voice import Transcriber

transcriber = Transcriber(model="whisper-1")

# Transcribe audio file
result = transcriber.transcribe("recording.wav")
print(result.text)

# Transcribe with timestamps
result = transcriber.transcribe(
    "meeting.mp3",
    timestamps=True,
    language="en"
)
```

### Text-to-Speech Only

```python
from pyai.voice import Synthesizer

synth = Synthesizer(voice="nova")

# Generate speech
audio = synth.speak("Hello, how can I help you today?")
audio.save("greeting.mp3")

# Stream speech
for chunk in synth.stream("This is a longer message..."):
    play_audio(chunk)
```

## Audio Formats

Supported formats:
- **PCM16**: Raw PCM audio (16-bit)
- **WAV**: Waveform Audio
- **MP3**: MPEG Audio Layer III
- **OGG**: Ogg Vorbis

```python
from pyai.voice.stream import AudioFormat

chunk = AudioChunk(
    data=audio_bytes,
    format=AudioFormat.PCM16,
    sample_rate=24000
)
```

## Voice Options

Available voices:
- `alloy` - Neutral, balanced
- `echo` - Warm, conversational
- `fable` - Expressive, narrative
- `onyx` - Deep, authoritative
- `nova` - Friendly, upbeat
- `shimmer` - Clear, professional

## Real-time Streaming

```python
from pyai.voice import VoiceSession

async def voice_assistant():
    session = VoiceSession()
    
    async with session.connect() as voice:
        # Enable turn detection
        voice.enable_turn_detection(
            threshold=0.5,
            silence_duration=0.8
        )
        
        # Continuous conversation
        while True:
            # User speaks
            user_audio = await voice.listen()
            
            # Agent responds
            await voice.respond(user_audio)
```

## Integration with Agents

```python
from pyai import Agent
from pyai.voice import VoiceSession

agent = Agent(
    name="VoiceAssistant",
    instructions="You are a helpful voice assistant."
)

# Attach voice capabilities
session = VoiceSession(agent=agent)
```

## See Also

- [VoiceSession](VoiceSession) - Session management
- [Transcription](Transcription) - Speech-to-text
- [Synthesis](Synthesis) - Text-to-speech
