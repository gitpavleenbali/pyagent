# translate

The `translate` function provides AI-powered language translation.

## Import

```python
from pyai import translate
```

## Basic Usage

```python
# Auto-detect source language
result = translate("Hello, world!", to="es")
# "¡Hola, mundo!"

# Specify source language
result = translate("Bonjour", from_lang="fr", to="en")
# "Hello"
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | required | Text to translate |
| `to` | str | required | Target language code |
| `from_lang` | str | "auto" | Source language (auto-detect) |
| `preserve_format` | bool | True | Keep formatting |

## Language Codes

| Code | Language |
|------|----------|
| `en` | English |
| `es` | Spanish |
| `fr` | French |
| `de` | German |
| `it` | Italian |
| `pt` | Portuguese |
| `zh` | Chinese |
| `ja` | Japanese |
| `ko` | Korean |
| `ar` | Arabic |

## Examples

### Basic Translation

```python
from pyai import translate

# English to Spanish
spanish = translate("The quick brown fox", to="es")
print(spanish)

# French to English
english = translate("Je suis un développeur", to="en")
print(english)
```

### Batch Translation

```python
texts = [
    "Hello",
    "How are you?",
    "Goodbye"
]

translations = [translate(t, to="de") for t in texts]
```

### Preserve Formatting

```python
markdown = """
# Title

- Point 1
- Point 2
"""

translated = translate(markdown, to="fr", preserve_format=True)
# Preserves markdown structure
```

### Async Usage

```python
import asyncio
from pyai import translate

async def main():
    result = await translate.async_("Hello world", to="ja")
    print(result)

asyncio.run(main())
```

## See Also

- [[ask]] - Question answering
- [[generate]] - Content generation
- [[summarize]] - Summarization
