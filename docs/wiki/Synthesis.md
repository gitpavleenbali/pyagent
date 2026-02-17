# Synthesis

The synthesis module converts text to speech using AI voice models.

## Import

```python
from pyai.voice import Synthesizer
from pyai.voice.synthesis import SynthesisResult
```

## Synthesizer Class

### Constructor

```python
Synthesizer(
    model: str = "tts-1",        # TTS model
    voice: str = "alloy",         # Voice selection
    speed: float = 1.0,           # Speech speed (0.25-4.0)
    response_format: str = "mp3"  # Output format
)
```

## Available Voices

| Voice | Description |
|-------|-------------|
| `alloy` | Neutral, balanced |
| `echo` | Warm, conversational |
| `fable` | Expressive, British accent |
| `onyx` | Deep, authoritative |
| `nova` | Friendly, energetic |
| `shimmer` | Clear, professional |

## Basic Usage

### Generate Speech

```python
synthesizer = Synthesizer(voice="nova")

# Generate audio
result = synthesizer.speak("Hello, how can I help you today?")

# Save to file
result.save("greeting.mp3")

# Get bytes
audio_bytes = result.audio_data
```

### Stream Speech

```python
# For longer text, stream to reduce latency
for chunk in synthesizer.stream("This is a longer message that will be streamed..."):
    play_audio(chunk.data)
```

## SynthesisResult

The result object contains:

```python
result.audio_data    # Raw audio bytes
result.format        # Audio format (mp3, wav, etc.)
result.duration      # Duration in seconds
result.sample_rate   # Sample rate
result.voice         # Voice used
```

### Save Methods

```python
# Save with format
result.save("output.mp3")
result.save("output.wav", format="wav")
result.save("output.ogg", format="opus")
```

## Output Formats

```python
# MP3 (default, smallest)
synth = Synthesizer(response_format="mp3")

# Opus (low latency streaming)
synth = Synthesizer(response_format="opus")

# AAC (high quality)
synth = Synthesizer(response_format="aac")

# FLAC (lossless)
synth = Synthesizer(response_format="flac")

# WAV (uncompressed)
synth = Synthesizer(response_format="wav")

# PCM (raw audio)
synth = Synthesizer(response_format="pcm")
```

## Quality Models

```python
# Standard quality (faster, cheaper)
synth = Synthesizer(model="tts-1")

# HD quality (higher fidelity)
synth = Synthesizer(model="tts-1-hd")
```

## Speed Control

```python
# Slower speech
synth = Synthesizer(speed=0.75)

# Faster speech
synth = Synthesizer(speed=1.5)

# Range: 0.25 to 4.0
```

## Batch Processing

```python
texts = [
    "Welcome to our service.",
    "How can I assist you today?",
    "Thank you for your patience."
]

results = synthesizer.batch_speak(texts)

for i, result in enumerate(results):
    result.save(f"audio_{i}.mp3")
```

## SSML Support

For advanced control (when supported):

```python
ssml_text = """
<speak>
    <emphasis level="strong">Welcome</emphasis> to our service.
    <break time="500ms"/>
    How may I <prosody rate="slow">assist you</prosody> today?
</speak>
"""

result = synthesizer.speak(ssml_text, ssml=True)
```

## Async Usage

```python
async def generate_speech_async():
    synth = Synthesizer()
    
    # Async generation
    result = await synth.speak_async("Hello, world!")
    
    # Async streaming
    async for chunk in synth.stream_async("Long text here..."):
        await play_audio_async(chunk)
```

## See Also

- [Voice-Module](Voice-Module) - Module overview
- [VoiceSession](VoiceSession) - Real-time voice
- [Transcription](Transcription) - Speech-to-text
