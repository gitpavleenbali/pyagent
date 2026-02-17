# VoiceSession

The `VoiceSession` class manages real-time voice conversations with AI agents.

## Import

```python
from pyagent.voice import VoiceSession
from pyagent.voice.session import SessionState
```

## Constructor

```python
VoiceSession(
    model: str = "gpt-4o-realtime",   # Voice model
    voice: str = "alloy",              # Voice selection
    agent: Agent = None,               # Associated agent
    language: str = "en",              # Language code
    sample_rate: int = 24000,          # Audio sample rate
    turn_detection: bool = True        # Auto turn detection
)
```

## Session States

| State | Description |
|-------|-------------|
| `IDLE` | Session created but not connected |
| `CONNECTING` | Establishing connection |
| `CONNECTED` | Active session |
| `LISTENING` | Receiving user audio |
| `PROCESSING` | AI processing input |
| `SPEAKING` | AI generating response |
| `DISCONNECTED` | Session ended |

## Basic Usage

### Async Context Manager

```python
session = VoiceSession()

async with session.connect() as voice:
    # Session is active here
    await voice.send_audio(audio_data)
    response = await voice.receive()
```

### Manual Connection

```python
session = VoiceSession()

await session.connect()
try:
    # Use session...
    pass
finally:
    await session.disconnect()
```

## Methods

### send_audio()

Send audio data to the session:

```python
await session.send_audio(
    audio_data: bytes,
    commit: bool = True  # End of utterance
)
```

### receive()

Receive audio response:

```python
# Get complete response
response = await session.receive()

# Stream response
async for chunk in session.stream_receive():
    play_audio(chunk.data)
```

### listen()

Listen for user input:

```python
# Auto turn detection
user_audio = await session.listen()

# With timeout
user_audio = await session.listen(timeout=10.0)
```

### respond()

Send audio and get response:

```python
response = await session.respond(user_audio)
```

### interrupt()

Interrupt current response:

```python
await session.interrupt()
```

## Configuration

### Turn Detection

```python
session.configure_turn_detection(
    enabled=True,
    threshold=0.5,        # Sensitivity
    prefix_padding=300,   # ms before speech
    silence_duration=800  # ms of silence to end turn
)
```

### Audio Settings

```python
session.configure_audio(
    sample_rate=24000,
    channels=1,
    format="pcm16"
)
```

## Events

Register event handlers:

```python
@session.on("speech_started")
async def on_speech_started():
    print("User started speaking")

@session.on("speech_ended")
async def on_speech_ended():
    print("User stopped speaking")

@session.on("response_started")
async def on_response_started():
    print("AI started responding")

@session.on("transcript_available")
async def on_transcript(text: str):
    print(f"Transcript: {text}")
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `state` | SessionState | Current state |
| `model` | str | Model being used |
| `voice` | str | Voice selection |
| `is_connected` | bool | Connection status |
| `session_id` | str | Unique session ID |

## See Also

- [Voice-Module](Voice-Module) - Module overview
- [Transcription](Transcription) - Speech-to-text
- [Synthesis](Synthesis) - Text-to-speech
