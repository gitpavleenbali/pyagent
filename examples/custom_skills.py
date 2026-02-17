"""# pyright: reportMissingImports=false, reportUnusedImport=falseExample: Custom Skills

This example demonstrates how to create custom skills for agents.

NOTE: This is a REFERENCE example showing the skill system architecture.
      For simpler one-liner usage, see: comprehensive_examples.py

Authentication (if running with LLM):
    # Option 1: OpenAI API Key
    export OPENAI_API_KEY=sk-your-key
    
    # Option 2: Azure OpenAI with Azure AD (recommended - no key needed)
    export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
    export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
"""

import os
import sys

# Add paths for local development (works from any directory, including PyCharm)
_examples_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_examples_dir)
sys.path.insert(0, _project_dir)  # For pyai imports
sys.path.insert(0, _examples_dir)  # For config_helper import

# Optional: Configure pyai for LLM-powered features
# This example demonstrates skill definitions without requiring LLM calls
try:
    from config_helper import setup_pyai
    setup_pyai(verbose=False)  # Silent - this example doesn't require LLM
except Exception:
    pass  # Skills work without LLM configuration

from typing import Any, Dict, List
from pyai.skills import Skill, SkillResult, SkillParameter
from pyai.skills.tool_skill import tool
from pyai.skills.action_skill import ActionSkill, action, ActionType


# =============================================================================
# Method 1: Using @tool decorator (simplest)
# =============================================================================

@tool(description="Calculate the factorial of a number")
async def factorial(n: int) -> int:
    """
    Calculate factorial of n.
    
    Args:
        n: The number to calculate factorial for
    """
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


@tool(description="Convert temperature between units")
async def convert_temperature(
    value: float,
    from_unit: str = "celsius",
    to_unit: str = "fahrenheit"
) -> float:
    """
    Convert temperature between Celsius and Fahrenheit.
    
    Args:
        value: The temperature value to convert
        from_unit: Source unit (celsius or fahrenheit)
        to_unit: Target unit (celsius or fahrenheit)
    """
    if from_unit == to_unit:
        return value
    
    if from_unit == "celsius" and to_unit == "fahrenheit":
        return (value * 9/5) + 32
    elif from_unit == "fahrenheit" and to_unit == "celsius":
        return (value - 32) * 5/9
    else:
        raise ValueError(f"Unknown units: {from_unit} -> {to_unit}")


# =============================================================================
# Method 2: Subclassing Skill (more control)
# =============================================================================

class WeatherSkill(Skill):
    """
    A skill for getting weather information.
    
    This demonstrates how to create a full skill class with
    parameter definitions and custom logic.
    """
    
    name = "weather"
    description = "Get current weather for a location"
    
    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        
        self.parameters = [
            SkillParameter(
                name="location",
                description="City name or coordinates",
                type="string",
                required=True,
            ),
            SkillParameter(
                name="units",
                description="Temperature units",
                type="string",
                required=False,
                default="metric",
                enum=["metric", "imperial"],
            ),
        ]
    
    async def execute(self, location: str, units: str = "metric", **kwargs) -> SkillResult:
        """Get weather for a location"""
        try:
            # In a real implementation, you would call a weather API
            # This is a mock response
            weather_data = {
                "location": location,
                "temperature": 22 if units == "metric" else 72,
                "units": "C" if units == "metric" else "F",
                "condition": "Partly Cloudy",
                "humidity": 65,
            }
            
            return SkillResult.ok(
                data=weather_data,
                message=f"Weather for {location}: {weather_data['temperature']}{weather_data['units']}"
            )
        except Exception as e:
            return SkillResult.fail(f"Failed to get weather: {e}")


# =============================================================================
# Method 3: ActionSkill (group related actions)
# =============================================================================

class DatabaseSkill(ActionSkill):
    """
    A skill for database operations.
    
    Uses the @action decorator to define multiple related actions
    within a single skill.
    """
    
    name = "database"
    description = "Perform database operations"
    
    def __init__(self, connection_string: str = None, **kwargs):
        super().__init__(**kwargs)
        self.connection_string = connection_string
        self._data = {}  # Mock in-memory storage
    
    @action("insert", description="Insert a record", action_type=ActionType.WRITE)
    async def insert(self, table: str, record: Dict[str, Any]) -> str:
        """Insert a record into a table"""
        if table not in self._data:
            self._data[table] = []
        
        record_id = f"{table}_{len(self._data[table])}"
        record["id"] = record_id
        self._data[table].append(record)
        
        return record_id
    
    @action("select", description="Query records", action_type=ActionType.READ)
    async def select(self, table: str, filters: Dict[str, Any] = None) -> List[Dict]:
        """Query records from a table"""
        if table not in self._data:
            return []
        
        records = self._data[table]
        
        if filters:
            filtered = []
            for record in records:
                match = all(
                    record.get(k) == v
                    for k, v in filters.items()
                )
                if match:
                    filtered.append(record)
            return filtered
        
        return records
    
    @action("update", description="Update a record", action_type=ActionType.WRITE)
    async def update(self, table: str, record_id: str, updates: Dict[str, Any]) -> bool:
        """Update a record by ID"""
        if table not in self._data:
            return False
        
        for record in self._data[table]:
            if record.get("id") == record_id:
                record.update(updates)
                return True
        
        return False
    
    @action("delete", description="Delete a record", action_type=ActionType.WRITE)
    async def delete(self, table: str, record_id: str) -> bool:
        """Delete a record by ID"""
        if table not in self._data:
            return False
        
        original_len = len(self._data[table])
        self._data[table] = [
            r for r in self._data[table]
            if r.get("id") != record_id
        ]
        
        return len(self._data[table]) < original_len


# =============================================================================
# Usage Examples
# =============================================================================

async def main():
    # Using @tool decorated function
    result = await factorial(n=5)
    print(f"Factorial: {result}")
    
    # Using Skill class
    weather = WeatherSkill()
    weather_result = await weather.execute(location="New York")
    print(f"Weather: {weather_result}")
    
    # Using ActionSkill
    db = DatabaseSkill()
    
    # Insert
    record_id = await db.execute(
        action="insert",
        table="users",
        record={"name": "Alice", "email": "alice@example.com"}
    )
    print(f"Inserted: {record_id}")
    
    # Select
    users = await db.execute(action="select", table="users")
    print(f"Users: {users}")
    
    # Get tool definitions for LLM
    print("\nTool Definitions:")
    print(weather.to_tool_definition())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
