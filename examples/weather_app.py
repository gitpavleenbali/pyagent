# pyright: reportMissingImports=false, reportUnusedImport=false
"""
pyai Weather App - Real-Time Weather Assistant
==================================================

This example demonstrates how to build a complete weather application
using pyai's simple one-liner functions.

Features:
- Get current weather for any city
- Multi-city comparison
- Weather-based recommendations
- Natural language weather queries

Usage:
    # Option 1: OpenAI
    export OPENAI_API_KEY=sk-your-key
    
    # Option 2: Azure OpenAI with Azure AD
    export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
    export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
    
    python weather_app.py

Requirements:
    - pyai library
    - Azure OpenAI or OpenAI API key
"""

import os
import sys

# =============================================================================
# SETUP: Add pyai to path and configure credentials
# =============================================================================

# Add paths for local development (works from any directory, including PyCharm)
_examples_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_examples_dir)
sys.path.insert(0, _project_dir)  # For pyai imports
sys.path.insert(0, _examples_dir)  # For config_helper import

# Configure pyai with available credentials (supports OpenAI, Azure API Key, or Azure AD)
from config_helper import setup_pyai
if not setup_pyai():
    print("Please configure credentials - see instructions above")
    sys.exit(1)

# import pyai functions
from pyai import ask, fetch, agent


# =============================================================================
# FEATURE 1: Simple Weather Query
# pyai makes getting weather data a one-liner!
# =============================================================================

def get_weather(city: str) -> None:
    """
    Get current weather for a city.
    
    This demonstrates pyai's fetch.weather() function which provides
    structured weather data in a single call.
    """
    print(f"\n[Temp] Weather for {city}")
    print("-" * 40)
    
    # One-liner to get weather! [Done]
    weather = fetch.weather(city)
    
    # Display the results
    print(f"  Temperature: {weather.temperature}C")
    print(f"  Conditions:  {weather.conditions}")
    print(f"  Humidity:    {weather.humidity}%")
    print(f"  Wind Speed:  {weather.wind_speed} km/h")


# =============================================================================
# FEATURE 2: Multi-City Weather Comparison
# Compare weather across multiple cities at once
# =============================================================================

def compare_weather(cities: list) -> None:
    """
    Compare weather across multiple cities.
    
    This shows how easy it is to aggregate data from multiple sources
    using pyai's simple API.
    """
    print(f"\n[Globe] Weather Comparison: {', '.join(cities)}")
    print("-" * 60)
    
    # Get weather for all cities
    weather_data = []
    for city in cities:
        weather = fetch.weather(city)
        weather_data.append({
            "city": city,
            "temp": weather.temperature,
            "conditions": weather.conditions
        })
    
    # Display comparison table
    print(f"{'City':<15} {'Temp (C)':<12} {'Conditions':<20}")
    print("-" * 47)
    for w in weather_data:
        print(f"{w['city']:<15} {w['temp']:<12} {w['conditions']:<20}")
    
    # Find the warmest city
    warmest = max(weather_data, key=lambda x: x['temp'])
    print(f"\n[Sun]  Warmest: {warmest['city']} at {warmest['temp']}C")


# =============================================================================
# FEATURE 3: AI Weather Assistant
# Use pyai's agent() to create a smart weather assistant
# =============================================================================

def weather_assistant():
    """
    Create an AI-powered weather assistant.
    
    This demonstrates pyai's agent() function to create a custom
    assistant that can answer weather-related questions intelligently.
    """
    print("\n[Bot] Weather Assistant")
    print("-" * 40)
    
    # Create a weather expert agent - just ONE LINE! 
    assistant = agent(
        "You are a helpful weather expert. Provide concise weather advice "
        "and recommendations based on weather conditions. Be friendly and brief.",
        name="WeatherBot"
    )
    
    # Sample questions the assistant can handle
    questions = [
        "It's 25C and sunny in Paris. Should I bring an umbrella?",
        "What should I wear for hiking if it's 10C with light rain?",
    ]
    
    for question in questions:
        print(f"\n[?] Question: {question}")
        response = assistant(question)
        print(f"[Chat] WeatherBot: {response}")


# =============================================================================
# FEATURE 4: Natural Language Weather Queries
# Use ask() for natural language weather understanding
# =============================================================================

def natural_weather_query(query: str) -> None:
    """
    Answer any weather-related question using natural language.
    
    This shows how pyai's ask() function can handle any question,
    including weather-related queries.
    """
    print("\n[Magic] Natural Language Query")
    print("-" * 40)
    print(f"Query: {query}")
    
    # Ask any question - pyai handles it! [Target]
    answer = ask(query, concise=True)
    
    print(f"Answer: {answer}")


# =============================================================================
# FEATURE 5: Weather-Based Activity Recommender
# Combine weather data with AI to recommend activities
# =============================================================================

def recommend_activities(city: str) -> None:
    """
    Recommend activities based on current weather.
    
    This combines pyai's fetch and ask capabilities to create
    intelligent recommendations.
    """
    print(f"\n[Target] Activity Recommendations for {city}")
    print("-" * 40)
    
    # Step 1: Get current weather
    weather = fetch.weather(city)
    print(f"Current weather: {weather.temperature}C, {weather.conditions}")
    
    # Step 2: Use AI to recommend activities based on weather
    prompt = f"""
    The weather in {city} is currently:
    - Temperature: {weather.temperature}C
    - Conditions: {weather.conditions}
    - Humidity: {weather.humidity}%
    
    Suggest 3 activities that would be perfect for this weather. Be brief.
    """
    
    recommendations = ask(prompt, format="bullet")
    print(f"\nRecommended activities:\n{recommendations}")


# =============================================================================
# MAIN: Run the Weather App Demo
# =============================================================================

def main():
    """Run all weather app features."""
    print("=" * 60)
    print("[Weather] pyai Weather App Demo")
    print("=" * 60)
    
    # Feature 1: Simple weather query
    get_weather("New York")
    
    # Feature 2: Multi-city comparison
    compare_weather(["London", "Tokyo", "Sydney"])
    
    # Feature 3: AI Weather Assistant
    weather_assistant()
    
    # Feature 4: Natural language query
    natural_weather_query(
        "What's the best time of year to visit Iceland for the Northern Lights?"
    )
    
    # Feature 5: Activity recommendations
    recommend_activities("San Francisco")
    
    print("\n" + "=" * 60)
    print("[Done] Weather App Demo Complete!")
    print("=" * 60)
    print("\nBuilt with pyai - AI development made simple! [pyai]")


if __name__ == "__main__":
    main()
