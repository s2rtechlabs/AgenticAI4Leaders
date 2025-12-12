"""
Weather Agent Implementation

This agent provides mock weather data for various cities.
In a real scenario, this would connect to a weather API.
"""

import random
from typing import Any
from uuid import uuid4


class WeatherAgent:
    """
    A weather agent that provides weather information for cities.
    
    Demonstrates:
    - External data simulation
    - City recognition
    - Structured responses
    """
    
    def __init__(self):
        self.name = "Weather Agent"
        self.version = "1.0.0"
        self.description = "Provides weather information for cities worldwide"
        
        # Mock weather data for various cities
        self.city_weather = {
            "tokyo": {
                "temp_range": (15, 28),
                "conditions": ["Sunny", "Partly Cloudy", "Clear", "Light Rain"],
                "humidity_range": (50, 75)
            },
            "new york": {
                "temp_range": (10, 25),
                "conditions": ["Sunny", "Cloudy", "Rainy", "Clear"],
                "humidity_range": (45, 70)
            },
            "london": {
                "temp_range": (8, 18),
                "conditions": ["Cloudy", "Rainy", "Overcast", "Foggy", "Drizzle"],
                "humidity_range": (65, 85)
            },
            "paris": {
                "temp_range": (10, 22),
                "conditions": ["Sunny", "Partly Cloudy", "Clear", "Light Rain"],
                "humidity_range": (55, 75)
            },
            "sydney": {
                "temp_range": (18, 30),
                "conditions": ["Sunny", "Clear", "Hot", "Partly Cloudy"],
                "humidity_range": (40, 65)
            },
            "mumbai": {
                "temp_range": (25, 35),
                "conditions": ["Sunny", "Hot", "Humid", "Monsoon Rain", "Partly Cloudy"],
                "humidity_range": (65, 90)
            },
            "berlin": {
                "temp_range": (5, 20),
                "conditions": ["Cloudy", "Clear", "Light Rain", "Overcast"],
                "humidity_range": (50, 75)
            },
            "dubai": {
                "temp_range": (28, 42),
                "conditions": ["Sunny", "Hot", "Clear", "Very Hot"],
                "humidity_range": (30, 55)
            },
            "singapore": {
                "temp_range": (26, 33),
                "conditions": ["Humid", "Thunderstorm", "Partly Cloudy", "Sunny"],
                "humidity_range": (70, 90)
            },
            "san francisco": {
                "temp_range": (12, 22),
                "conditions": ["Foggy", "Clear", "Sunny", "Mild"],
                "humidity_range": (60, 80)
            },
            "moscow": {
                "temp_range": (-10, 15),
                "conditions": ["Snowy", "Cold", "Overcast", "Freezing", "Clear"],
                "humidity_range": (60, 85)
            },
            "beijing": {
                "temp_range": (5, 30),
                "conditions": ["Sunny", "Hazy", "Clear", "Windy"],
                "humidity_range": (35, 65)
            }
        }
        
        # Default weather for unknown cities
        self.default_weather = {
            "temp_range": (15, 25),
            "conditions": ["Partly Cloudy", "Clear", "Sunny"],
            "humidity_range": (50, 70)
        }
    
    def get_agent_card(self) -> dict[str, Any]:
        """Returns the Agent Card for this agent."""
        return {
            "name": self.name,
            "description": self.description,
            "url": "http://localhost:10003/",
            "version": self.version,
            "defaultInputModes": ["text"],
            "defaultOutputModes": ["text"],
            "capabilities": {
                "streaming": False,
                "pushNotifications": False
            },
            "skills": [
                {
                    "id": "get_weather",
                    "name": "Get Weather",
                    "description": "Gets current weather for a specified city",
                    "tags": ["weather", "temperature", "forecast", "conditions"],
                    "examples": [
                        "What's the weather in Tokyo?",
                        "Weather in New York",
                        "How's the weather in London?",
                        "Temperature in Paris"
                    ]
                },
                {
                    "id": "list_cities",
                    "name": "List Cities",
                    "description": "Lists all cities with available weather data",
                    "tags": ["cities", "list", "available"],
                    "examples": [
                        "What cities do you have weather for?",
                        "List available cities"
                    ]
                },
                {
                    "id": "compare_weather",
                    "name": "Compare Weather",
                    "description": "Compares weather between two cities",
                    "tags": ["compare", "difference", "versus"],
                    "examples": [
                        "Compare weather in Tokyo and London",
                        "Is it warmer in Dubai or Singapore?"
                    ]
                }
            ]
        }
    
    async def handle_message(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle an incoming A2A message."""
        message = params.get("message", {})
        parts = message.get("parts", [])
        
        text = ""
        for part in parts:
            if part.get("kind") == "text":
                text += part.get("text", "")
        
        result = self._process_request(text)
        
        return {
            "message": {
                "role": "agent",
                "parts": [
                    {
                        "kind": "text",
                        "text": result
                    }
                ],
                "messageId": str(uuid4())
            }
        }
    
    def _find_city(self, text: str) -> str | None:
        """Find a city name in the text."""
        text_lower = text.lower()
        
        for city in self.city_weather.keys():
            if city in text_lower:
                return city
        
        return None
    
    def _get_weather(self, city: str) -> dict[str, Any]:
        """Generate weather data for a city."""
        weather_config = self.city_weather.get(city, self.default_weather)
        
        temp = random.randint(*weather_config["temp_range"])
        condition = random.choice(weather_config["conditions"])
        humidity = random.randint(*weather_config["humidity_range"])
        
        return {
            "city": city.title(),
            "temperature_c": temp,
            "temperature_f": round(temp * 9/5 + 32),
            "condition": condition,
            "humidity": humidity
        }
    
    def _process_request(self, text: str) -> str:
        """Process the weather request."""
        text_lower = text.lower()
        
        # Check for list cities request
        if any(word in text_lower for word in ["list", "cities", "available"]):
            cities = [city.title() for city in self.city_weather.keys()]
            return f"I have weather data for: {', '.join(cities)}"
        
        # Check for comparison request
        if "compare" in text_lower or "versus" in text_lower or " vs " in text_lower:
            cities_found = []
            for city in self.city_weather.keys():
                if city in text_lower:
                    cities_found.append(city)
            
            if len(cities_found) >= 2:
                weather1 = self._get_weather(cities_found[0])
                weather2 = self._get_weather(cities_found[1])
                
                return (
                    f"Weather Comparison:\n"
                    f"📍 {weather1['city']}: {weather1['temperature_c']}°C, {weather1['condition']}, {weather1['humidity']}% humidity\n"
                    f"📍 {weather2['city']}: {weather2['temperature_c']}°C, {weather2['condition']}, {weather2['humidity']}% humidity\n"
                    f"Temperature difference: {abs(weather1['temperature_c'] - weather2['temperature_c'])}°C"
                )
            else:
                return "Please mention two cities to compare. Example: 'Compare weather in Tokyo and London'"
        
        # Get weather for a single city
        city = self._find_city(text)
        
        if city:
            weather = self._get_weather(city)
            return (
                f"🌤️ Weather in {weather['city']}:\n"
                f"   Temperature: {weather['temperature_c']}°C ({weather['temperature_f']}°F)\n"
                f"   Condition: {weather['condition']}\n"
                f"   Humidity: {weather['humidity']}%"
            )
        else:
            cities = [city.title() for city in self.city_weather.keys()]
            return (
                f"I couldn't find that city. Try one of these: {', '.join(cities[:5])}..."
            )

