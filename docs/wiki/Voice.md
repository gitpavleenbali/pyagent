# Voice

Real-time voice interactions with AI agents.

> **See [[Voice-Module]] for full documentation.**

## Quick Start

```python
from pyai.voice import VoiceSession

# Create voice session
session = VoiceSession()

# Start conversation
await session.start()

# Speak and get response
response = await session.speak("Hello, how are you?")
print(response.text)

# End session
await session.stop()
```

## Features

- Real-time speech-to-text
- Text-to-speech synthesis
- Voice activity detection
- Multiple voice providers
- Streaming responses

## Related Pages

- [[Voice-Module]] - Full module documentation
- [[VoiceSession]] - Session class
- [[Transcription]] - Speech-to-text
- [[Synthesis]] - Text-to-speech
