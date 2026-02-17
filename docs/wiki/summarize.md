# summarize

The `summarize` function condenses text, files, or URLs into concise summaries.

## Import

```python
from pyai import summarize
```

## Basic Usage

```python
# Summarize text
summary = summarize("Long article text here...")

# Summarize URL
summary = summarize("https://example.com/article")

# Summarize file
summary = summarize("document.pdf")
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | str | required | Text, URL, or file path |
| `length` | str | "medium" | Output length: "short", "medium", "long" |
| `format` | str | "paragraph" | Format: "paragraph", "bullets", "numbered" |
| `focus` | str | None | Specific aspect to focus on |

## Return Value

Returns a string containing the summarized content.

## Examples

### Summarize Text

```python
from pyai import summarize

article = """
PYAI is a comprehensive Python SDK for building AI agents...
[long article text]
"""

summary = summarize(article)
print(summary)
```

### Summarize URL

```python
# Fetch and summarize web content
summary = summarize("https://en.wikipedia.org/wiki/Artificial_intelligence")
```

### Summarize File

```python
# Supports PDF, TXT, DOCX, MD
summary = summarize("report.pdf")
summary = summarize("notes.txt")
```

### Custom Options

```python
# Short bullet-point summary
summary = summarize(
    article,
    length="short",
    format="bullets"
)

# Focus on specific topic
summary = summarize(
    article,
    focus="technical implementation"
)
```

### Async Usage

```python
import asyncio
from pyai import summarize

async def main():
    summary = await summarize.async_("https://example.com/article")
    print(summary)

asyncio.run(main())
```

## Supported File Types

| Extension | Description |
|-----------|-------------|
| `.txt` | Plain text files |
| `.md` | Markdown files |
| `.pdf` | PDF documents |
| `.docx` | Word documents |
| `.html` | HTML files |

## See Also

- [[ask]] - Question answering
- [[research]] - Deep research
- [[extract]] - Data extraction
