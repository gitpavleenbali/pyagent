# analyze

The `analyze` module provides AI-powered data analysis capabilities.

## Import

```python
from pyai.easy import analyze
```

## Quick Start

```python
from pyai.easy import analyze

# Analyze data
insights = analyze.data(data, question="What are the trends?")

# Analyze sentiment
sentiment = analyze.sentiment("Great product, loved it!")

# Analyze text
analysis = analyze.text(document)
```

## Functions

### Data Analysis

```python
from pyai.easy import analyze

data = [
    {"month": "Jan", "sales": 1000},
    {"month": "Feb", "sales": 1200},
    {"month": "Mar", "sales": 1100},
]

# General analysis
insights = analyze.data(data)
print(insights)

# Specific question
answer = analyze.data(data, question="What is the average sales?")
```

### Sentiment Analysis

```python
# Single text
sentiment = analyze.sentiment("This product is amazing!")
print(sentiment)
# {"sentiment": "positive", "score": 0.95}

# Multiple texts
texts = [
    "Great service!",
    "Could be better",
    "Terrible experience"
]
results = analyze.sentiment_batch(texts)
```

### Text Analysis

```python
# Analyze document
analysis = analyze.text(document)
print(analysis)
# {
#   "word_count": 500,
#   "reading_time": "2 min",
#   "complexity": "intermediate",
#   "key_topics": ["AI", "machine learning"],
#   "tone": "informative"
# }
```

### CSV Analysis

```python
# Analyze CSV file
insights = analyze.csv("sales_data.csv")

# With specific questions
insights = analyze.csv(
    "data.csv",
    questions=[
        "What is the total revenue?",
        "Which product sells most?",
        "What are the trends?"
    ]
)
```

### DataFrame Analysis

```python
import pandas as pd
from pyai.easy import analyze

df = pd.read_csv("data.csv")
insights = analyze.dataframe(df, "What patterns do you see?")
```

## Examples

### Sales Analysis

```python
from pyai.easy import analyze

sales_data = [
    {"product": "A", "revenue": 10000, "units": 100},
    {"product": "B", "revenue": 8000, "units": 200},
    {"product": "C", "revenue": 15000, "units": 75},
]

insights = analyze.data(
    sales_data,
    question="Which product has the best revenue per unit?"
)
print(insights)
```

### Customer Feedback

```python
from pyai.easy import analyze

reviews = [
    "Love this product!",
    "Works as expected",
    "Not worth the price",
    "Excellent quality"
]

# Batch sentiment analysis
results = analyze.sentiment_batch(reviews)
for review, result in zip(reviews, results):
    print(f"{result['sentiment']}: {review}")
```

### Document Analysis

```python
from pyai.easy import analyze

with open("report.txt") as f:
    content = f.read()

analysis = analyze.text(content)
print(f"Key topics: {analysis['key_topics']}")
print(f"Reading time: {analysis['reading_time']}")
```

## Async Usage

```python
import asyncio
from pyai.easy import analyze

async def main():
    insights = await analyze.data_async(data, "What are the trends?")
    print(insights)

asyncio.run(main())
```

## See Also

- [[ask]] - Question answering
- [[extract]] - Data extraction
- [[summarize]] - Summarization
