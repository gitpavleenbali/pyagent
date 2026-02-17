# Creating Tools

Tools extend agent capabilities with custom functions.

## The @tool Decorator

```python
from pyai import tool

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.
    
    Args:
        expression: Math expression to evaluate
        
    Returns:
        The calculated result
    """
    return str(eval(expression))
```

## Basic Usage

```python
from pyai import Agent, tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city"""
    # API call here
    return f"Weather in {city}: Sunny, 72°F"

@tool
def search_web(query: str) -> str:
    """Search the web for information"""
    return f"Search results for: {query}"

# Add tools to agent
agent = Agent(
    name="Assistant",
    instructions="Use tools to help users",
    tools=[get_weather, search_web]
)
```

## Type Hints

Tools use type hints for parameter validation:

```python
from typing import List, Optional
from pyai import tool

@tool
def process_data(
    items: List[str],
    limit: int = 10,
    filter_empty: bool = True,
    prefix: Optional[str] = None
) -> str:
    """Process a list of items.
    
    Args:
        items: List of items to process
        limit: Maximum items to return
        filter_empty: Whether to remove empty items
        prefix: Optional prefix to add
    """
    result = items[:limit]
    if filter_empty:
        result = [i for i in result if i]
    if prefix:
        result = [f"{prefix}{i}" for i in result]
    return str(result)
```

## Async Tools

```python
from pyai import tool
import aiohttp

@tool
async def fetch_url(url: str) -> str:
    """Fetch content from a URL asynchronously"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

## Tool with Context

```python
from pyai import tool, ToolContext

@tool
def get_user_info(ctx: ToolContext) -> str:
    """Get information about current user"""
    return f"User: {ctx.user_id}, Session: {ctx.session_id}"
```

## Complex Return Types

```python
from pydantic import BaseModel
from pyai import tool

class WeatherResponse(BaseModel):
    temperature: float
    condition: str
    humidity: int

@tool
def get_detailed_weather(city: str) -> WeatherResponse:
    """Get detailed weather information"""
    return WeatherResponse(
        temperature=72.5,
        condition="Sunny",
        humidity=45
    )
```

## Tool Groups

```python
from pyai import ToolGroup

class MathTools(ToolGroup):
    """Mathematical operation tools"""
    
    @tool
    def add(self, a: float, b: float) -> float:
        """Add two numbers"""
        return a + b
    
    @tool
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers"""
        return a * b
    
    @tool
    def divide(self, a: float, b: float) -> float:
        """Divide two numbers"""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

# Use with agent
agent = Agent(
    name="Math Helper",
    tools=[MathTools()]
)
```

## Error Handling

```python
from pyai import tool, ToolError

@tool
def safe_divide(a: float, b: float) -> float:
    """Safely divide two numbers"""
    if b == 0:
        raise ToolError("Cannot divide by zero")
    return a / b
```

## Tool Metadata

```python
@tool(
    name="search",
    description="Search the knowledge base",
    tags=["search", "retrieval"],
    requires_confirmation=True
)
def search_knowledge(query: str) -> str:
    """Search internal knowledge base"""
    pass
```

## See Also

- [[Built-in-Skills]] - Pre-built tools
- [[OpenAPI-Tools]] - Auto-generated tools
- [[Agent]] - Agent class
