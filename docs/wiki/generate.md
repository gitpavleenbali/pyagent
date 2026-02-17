# generate

The `generate` function creates various types of content using AI.

## Import

```python
from pyai import generate
```

## Basic Usage

```python
# Generate content
content = generate("blog post about Python")

# With type specification
email = generate("thank you email", type="email")

# Async version
content = await generate.async_("product description")
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | required | Description of content to generate |
| `type` | str | None | Content type: "email", "blog", "code", "story" |
| `tone` | str | "professional" | Tone: "formal", "casual", "professional" |
| `length` | str | "medium" | Length: "short", "medium", "long" |
| `format` | str | None | Output format: "markdown", "html", "plain" |

## Content Types

| Type | Description |
|------|-------------|
| `email` | Professional emails |
| `blog` | Blog posts and articles |
| `code` | Code snippets |
| `story` | Creative stories |
| `report` | Business reports |
| `documentation` | Technical docs |

## Examples

### Generate Email

```python
from pyai import generate

email = generate(
    "thank customer for purchase",
    type="email",
    tone="friendly"
)
print(email)
```

### Generate Blog Post

```python
blog = generate(
    "introduction to machine learning",
    type="blog",
    length="long",
    format="markdown"
)
```

### Generate Documentation

```python
docs = generate(
    "API documentation for user authentication",
    type="documentation"
)
```

### Generate Code

```python
code = generate(
    "Python function to validate email addresses",
    type="code"
)
```

### Async Usage

```python
import asyncio
from pyai import generate

async def main():
    content = await generate.async_(
        "product launch announcement",
        type="email"
    )
    print(content)

asyncio.run(main())
```

## Templates

```python
# Use predefined templates
content = generate.from_template(
    "newsletter",
    topic="Monthly Update",
    highlights=["Feature A", "Feature B"]
)
```

## See Also

- [[ask]] - Question answering
- [[code]] - Code operations
- [[translate]] - Translation
