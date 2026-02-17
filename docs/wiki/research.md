# research

The `research` function performs deep topic research with source citations.

## Import

```python
from pyai import research
```

## Basic Usage

```python
# Simple research
results = research("machine learning trends 2024")

# With specific depth
results = research("quantum computing", depth="deep")

# Async version  
results = await research.async_("AI ethics")
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `topic` | str | required | Topic to research |
| `depth` | str | "normal" | Research depth: "quick", "normal", "deep" |
| `sources` | int | 5 | Number of sources to include |
| `format` | str | "markdown" | Output format |

## Return Value

Returns a structured research report with:
- Summary
- Key findings
- Source citations
- Further reading suggestions

## Examples

### Basic Research

```python
from pyai import research

# Quick research
report = research("Python async programming")
print(report)
```

### Deep Research

```python
# Comprehensive research with more sources
report = research(
    "artificial general intelligence",
    depth="deep",
    sources=10
)
```

### Async Research

```python
import asyncio
from pyai import research

async def main():
    report = await research.async_("blockchain technology")
    print(report)

asyncio.run(main())
```

## Output Format

```markdown
# Research: Machine Learning Trends 2024

## Summary
Machine learning continues to evolve with focus on...

## Key Findings
1. Large Language Models dominate...
2. Edge ML gaining traction...

## Sources
- [Source 1](https://example.com)
- [Source 2](https://example.com)

## Further Reading
- Topic A
- Topic B
```

## See Also

- [[ask]] - Simple question answering
- [[summarize]] - Text summarization
- [[analyze]] - Data analysis
