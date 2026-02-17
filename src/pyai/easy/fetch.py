"""
fetch Module - Real-time Data Fetching Agents

Get weather, news, stocks, and more in one line.

Examples:
    >>> from pyai import fetch
    >>>
    >>> # Weather
    >>> weather = fetch.weather("New York")
    >>> print(weather.temperature, weather.conditions)

    >>> # News
    >>> news = fetch.news("AI technology")
    >>> for article in news:
    ...     print(article.title)

    >>> # Stocks
    >>> stock = fetch.stock("AAPL")
    >>> print(stock.price, stock.change)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from pyai.easy.llm_interface import get_llm

# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class WeatherInfo:
    """Weather information for a location."""

    location: str
    temperature: float
    unit: str = "celsius"
    conditions: str = ""
    humidity: float = None
    wind_speed: float = None
    forecast: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __str__(self) -> str:
        return f"{self.location}: {self.temperature}°{self.unit[0].upper()}, {self.conditions}"

    @property
    def fahrenheit(self) -> float:
        if self.unit.lower() == "celsius":
            return self.temperature * 9 / 5 + 32
        return self.temperature

    @property
    def celsius(self) -> float:
        if self.unit.lower() == "fahrenheit":
            return (self.temperature - 32) * 5 / 9
        return self.temperature


@dataclass
class NewsArticle:
    """A news article."""

    title: str
    summary: str
    source: str = ""
    url: str = ""
    published: str = ""

    def __str__(self) -> str:
        return f"{self.title} ({self.source})"


@dataclass
class NewsResults:
    """Collection of news articles."""

    query: str
    articles: List[NewsArticle] = field(default_factory=list)

    def __iter__(self):
        return iter(self.articles)

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        return self.articles[idx]

    def __str__(self) -> str:
        return "\n".join(f"• {a.title}" for a in self.articles[:5])


@dataclass
class StockInfo:
    """Stock market information."""

    symbol: str
    name: str = ""
    price: float = 0.0
    change: float = 0.0
    change_percent: float = 0.0
    volume: int = 0
    market_cap: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __str__(self) -> str:
        sign = "+" if self.change >= 0 else ""
        return f"{self.symbol}: ${self.price:.2f} ({sign}{self.change_percent:.2f}%)"


@dataclass
class CryptoInfo:
    """Cryptocurrency information."""

    symbol: str
    name: str = ""
    price: float = 0.0
    change_24h: float = 0.0
    market_cap: str = ""
    volume_24h: str = ""

    def __str__(self) -> str:
        sign = "+" if self.change_24h >= 0 else ""
        return f"{self.symbol}: ${self.price:.2f} ({sign}{self.change_24h:.2f}%)"


# =============================================================================
# FETCH FUNCTIONS
# =============================================================================


def weather(
    location: str,
    *,
    unit: str = "celsius",
    include_forecast: bool = False,
    model: str = None,
    **kwargs,
) -> WeatherInfo:
    """
    Get weather information for a location.

    Args:
        location: City name or location
        unit: Temperature unit ("celsius" or "fahrenheit")
        include_forecast: Include weather forecast
        model: Override default model
        **kwargs: Additional parameters

    Returns:
        WeatherInfo: Weather data

    Examples:
        >>> weather("Paris")
        WeatherInfo(location='Paris', temperature=22.0, conditions='Partly Cloudy')

        >>> weather("Tokyo", unit="fahrenheit", include_forecast=True)

    Note:
        This uses AI to simulate weather data. For production,
        integrate with a real weather API like OpenWeatherMap.
    """
    llm_kwargs = {"model": model} if model else {}
    llm = get_llm(**llm_kwargs)

    prompt = f"""Provide realistic current weather data for {location}.

Return JSON with:
- temperature (number, in {unit})
- conditions (string, e.g., "Sunny", "Partly Cloudy", "Rainy")
- humidity (number, percentage)
- wind_speed (number, km/h)
{"- forecast (string, brief 3-day forecast)" if include_forecast else ""}

Make the data realistic for the current season and this location."""

    data = llm.json(
        prompt,
        system="You are a weather data provider. Return realistic weather data.",
        temperature=0.5,
    )

    return WeatherInfo(
        location=location,
        temperature=float(data.get("temperature", 20)),
        unit=unit,
        conditions=data.get("conditions", "Unknown"),
        humidity=data.get("humidity"),
        wind_speed=data.get("wind_speed"),
        forecast=data.get("forecast", "") if include_forecast else "",
    )


def news(
    topic: str = None,
    *,
    category: str = None,  # "technology", "business", "sports", etc.
    count: int = 5,
    model: str = None,
    **kwargs,
) -> NewsResults:
    """
    Get news articles on a topic.

    Args:
        topic: Topic to search for
        category: News category
        count: Number of articles
        model: Override default model
        **kwargs: Additional parameters

    Returns:
        NewsResults: Collection of news articles

    Examples:
        >>> news("artificial intelligence")
        NewsResults with 5 articles

        >>> news(category="technology", count=10)

    Note:
        This uses AI to generate representative news. For production,
        integrate with NewsAPI, Google News, or similar services.
    """
    llm_kwargs = {"model": model} if model else {}
    llm = get_llm(**llm_kwargs)

    search_term = topic or category or "general"

    prompt = f"""Generate {count} realistic news headlines and summaries about: {search_term}

Return JSON array with each article having:
- title (string)
- summary (string, 1-2 sentences)
- source (string, realistic news source name)
- published (string, relative time like "2 hours ago")

Make them realistic and current-sounding."""

    data = llm.json(
        prompt,
        system="You are a news aggregator. Generate realistic news article metadata.",
        temperature=0.7,
    )

    articles = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "articles" in data:
        items = data["articles"]
    else:
        items = []

    for item in items[:count]:
        articles.append(
            NewsArticle(
                title=item.get("title", ""),
                summary=item.get("summary", ""),
                source=item.get("source", ""),
                published=item.get("published", ""),
            )
        )

    return NewsResults(query=search_term, articles=articles)


def stock(symbol: str, *, model: str = None, **kwargs) -> StockInfo:
    """
    Get stock information.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "GOOGL")
        model: Override default model
        **kwargs: Additional parameters

    Returns:
        StockInfo: Stock market data

    Examples:
        >>> stock("AAPL")
        StockInfo(symbol='AAPL', price=178.50, change=+2.35%)

    Note:
        This uses AI to generate representative data. For production,
        integrate with Alpha Vantage, Yahoo Finance, or similar APIs.
    """
    llm_kwargs = {"model": model} if model else {}
    llm = get_llm(**llm_kwargs)

    prompt = f"""Provide realistic current stock market data for {symbol}.

Return JSON with:
- name (string, company name)
- price (number, current stock price in USD)
- change (number, price change today)
- change_percent (number, percentage change)
- volume (number, trading volume)
- market_cap (string, e.g., "2.8T")

Make the data realistic for this company."""

    data = llm.json(
        prompt,
        system="You are a stock market data provider. Return realistic stock data.",
        temperature=0.5,
    )

    return StockInfo(
        symbol=symbol.upper(),
        name=data.get("name", symbol),
        price=float(data.get("price", 0)),
        change=float(data.get("change", 0)),
        change_percent=float(data.get("change_percent", 0)),
        volume=int(data.get("volume", 0)),
        market_cap=data.get("market_cap", ""),
    )


def crypto(symbol: str, *, model: str = None, **kwargs) -> CryptoInfo:
    """
    Get cryptocurrency information.

    Args:
        symbol: Crypto symbol (e.g., "BTC", "ETH")
        model: Override default model
        **kwargs: Additional parameters

    Returns:
        CryptoInfo: Cryptocurrency data

    Examples:
        >>> crypto("BTC")
        CryptoInfo(symbol='BTC', price=43500.00, change_24h=+2.5%)
    """
    llm_kwargs = {"model": model} if model else {}
    llm = get_llm(**llm_kwargs)

    prompt = f"""Provide realistic current cryptocurrency data for {symbol}.

Return JSON with:
- name (string, full name)
- price (number, current price in USD)
- change_24h (number, 24-hour percentage change)
- market_cap (string, e.g., "850B")
- volume_24h (string, e.g., "25B")"""

    data = llm.json(
        prompt,
        system="You are a crypto market data provider. Return realistic data.",
        temperature=0.5,
    )

    return CryptoInfo(
        symbol=symbol.upper(),
        name=data.get("name", symbol),
        price=float(data.get("price", 0)),
        change_24h=float(data.get("change_24h", 0)),
        market_cap=data.get("market_cap", ""),
        volume_24h=data.get("volume_24h", ""),
    )


def url(url: str, **kwargs) -> str:
    """
    Fetch and extract content from a URL.

    Args:
        url: The URL to fetch
        **kwargs: Additional parameters

    Returns:
        str: Extracted text content
    """
    from pyai.easy.summarize import _fetch_url

    return _fetch_url(url)


def facts(topic: str, count: int = 5, **kwargs) -> List[str]:
    """
    Get interesting facts about a topic.

    Args:
        topic: Topic to get facts about
        count: Number of facts
        **kwargs: Additional parameters

    Returns:
        List of facts
    """
    llm = get_llm(**kwargs)

    result = llm.json(
        f"Provide {count} interesting, accurate facts about: {topic}",
        system="You are a knowledge expert. Return accurate, interesting facts.",
        schema={"facts": ["list of fact strings"]},
    )

    return result.get("facts", [])


def definition(term: str, **kwargs) -> str:
    """
    Get a definition for a term.

    Args:
        term: The term to define
        **kwargs: Additional parameters

    Returns:
        str: The definition
    """
    llm = get_llm(**kwargs)

    response = llm.complete(
        f"Define '{term}' clearly and concisely.",
        system="You are a dictionary. Provide clear, accurate definitions.",
        temperature=0.3,
    )

    return response.content
