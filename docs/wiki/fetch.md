# fetch

The `fetch` module retrieves real-time data from various sources.

## Import

```python
from pyai.easy import fetch
```

## Quick Start

```python
from pyai.easy import fetch

# Get weather
weather = fetch.weather("New York")

# Get news
news = fetch.news("technology")

# Get stock price
stock = fetch.stock("AAPL")
```

## Functions

### Weather

```python
weather = fetch.weather("San Francisco")
print(weather)
# {
#   "temperature": 72,
#   "condition": "Sunny",
#   "humidity": 45,
#   "wind": "10 mph"
# }

# With units
weather = fetch.weather("London", units="metric")
```

### News

```python
# Get news by topic
news = fetch.news("artificial intelligence")

# Get news by category
news = fetch.news(category="technology")

# Limit results
news = fetch.news("climate", limit=5)
```

### Stocks

```python
# Get stock price
stock = fetch.stock("AAPL")
print(stock)
# {
#   "symbol": "AAPL",
#   "price": 175.50,
#   "change": +2.30,
#   "change_percent": +1.33
# }

# Multiple stocks
stocks = fetch.stocks(["AAPL", "GOOGL", "MSFT"])
```

### Crypto

```python
# Get crypto price
btc = fetch.crypto("BTC")
print(btc)
# {
#   "symbol": "BTC",
#   "price": 45000.00,
#   "change_24h": +1200.00
# }
```

### URL Content

```python
# Fetch and parse web content
content = fetch.url("https://example.com/article")
print(content.text)
print(content.title)
```

## Examples

### Weather Dashboard

```python
from pyai.easy import fetch

cities = ["New York", "London", "Tokyo"]
for city in cities:
    w = fetch.weather(city)
    print(f"{city}: {w['temperature']}°F - {w['condition']}")
```

### News Aggregator

```python
from pyai.easy import fetch

topics = ["AI", "climate", "space"]
for topic in topics:
    articles = fetch.news(topic, limit=3)
    print(f"\n{topic.upper()}:")
    for article in articles:
        print(f"  - {article['title']}")
```

### Stock Tracker

```python
from pyai.easy import fetch

portfolio = ["AAPL", "GOOGL", "MSFT", "AMZN"]
stocks = fetch.stocks(portfolio)

for stock in stocks:
    sign = "+" if stock['change'] > 0 else ""
    print(f"{stock['symbol']}: ${stock['price']:.2f} ({sign}{stock['change_percent']:.2f}%)")
```

## Async Usage

```python
import asyncio
from pyai.easy import fetch

async def main():
    weather = await fetch.weather_async("New York")
    news = await fetch.news_async("tech")
    print(weather, news)

asyncio.run(main())
```

## See Also

- [[ask]] - Question answering
- [[research]] - Deep research
- [[analyze]] - Data analysis
