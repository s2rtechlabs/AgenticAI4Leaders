# AI Generated Code by Deloitte + Cursor (BEGIN)
# Exercise 1: Build a Weather MCP Server with FastMCP

**Difficulty**: Beginner  
**Time**: 20-30 minutes  
**Prerequisites**: Python basics, completed `01-basic-calculator` example

## 🎯 Objective

Build your first MCP server from scratch using FastMCP! You'll create a weather information server that provides current weather, forecasts, and temperature conversions.

## 📚 Learning Goals

By completing this exercise, you will:
- ✅ Understand the `@mcp.tool()` decorator
- ✅ Design tools with proper type hints and docstrings
- ✅ Handle errors gracefully
- ✅ Test your MCP server
- ✅ Use FastMCP CLI tools

## 📦 Setup

### 1. Create Project

```bash
mkdir weather-mcp-server
cd weather-mcp-server
```

### 2. Install FastMCP

```bash
pip install fastmcp
```

### 3. Verify Installation

```bash
python -c "from fastmcp import FastMCP; print('FastMCP ready!')"
```

## 🔧 Tools to Implement

### 1. `get_current_weather`

Get current weather for a city.

```python
@mcp.tool()
def get_current_weather(city: str, units: str = "celsius") -> str:
    """Get current weather for a city.
    
    Args:
        city: City name (e.g., 'London', 'New York')
        units: Temperature units - 'celsius' or 'fahrenheit'
    """
    # Your implementation here
```

**Expected Output:**
```
Weather in London: 15°C, Partly Cloudy, Humidity: 65%
```

### 2. `get_forecast`

Get weather forecast for upcoming days.

```python
@mcp.tool()
def get_forecast(city: str, days: int = 3) -> str:
    """Get weather forecast for upcoming days.
    
    Args:
        city: City name
        days: Number of days (1-7, default: 3)
    """
    # Your implementation here
```

### 3. `convert_temperature`

Convert temperature between units.

```python
@mcp.tool()
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """Convert temperature between Celsius, Fahrenheit, and Kelvin.
    
    Args:
        value: Temperature value to convert
        from_unit: Source unit (C, F, or K)
        to_unit: Target unit (C, F, or K)
    """
    # Your implementation here
```

## 📝 Step-by-Step Guide

### Step 1: Create `server.py`

```python
from fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP(
    name="Weather MCP Server",
    version="1.0.0",
    description="A weather information server"
)

# Mock weather data (no API needed!)
WEATHER_DATA = {
    "london": {"temp": 15, "condition": "Partly Cloudy", "humidity": 65},
    "new york": {"temp": 22, "condition": "Sunny", "humidity": 45},
    "tokyo": {"temp": 18, "condition": "Rainy", "humidity": 80},
    "paris": {"temp": 12, "condition": "Cloudy", "humidity": 70},
    "sydney": {"temp": 25, "condition": "Clear", "humidity": 50}
}

# TODO: Implement your tools here!

@mcp.tool()
def get_current_weather(city: str, units: str = "celsius") -> str:
    """Get current weather for a city.
    
    Args:
        city: City name (e.g., 'London', 'New York')
        units: Temperature units - 'celsius' or 'fahrenheit'
    """
    # Your code here
    pass

@mcp.tool()
def get_forecast(city: str, days: int = 3) -> str:
    """Get weather forecast for upcoming days.
    
    Args:
        city: City name
        days: Number of days (1-7, default: 3)
    """
    # Your code here
    pass

@mcp.tool()
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """Convert temperature between Celsius, Fahrenheit, and Kelvin.
    
    Args:
        value: Temperature value to convert
        from_unit: Source unit (C, F, or K)
        to_unit: Target unit (C, F, or K)
    """
    # Your code here
    pass

if __name__ == "__main__":
    print("Weather MCP Server")
    print("==================")
    mcp.run()
```

### Step 2: Implement `get_current_weather`

<details>
<summary>💡 Click for hints</summary>

```python
@mcp.tool()
def get_current_weather(city: str, units: str = "celsius") -> str:
    city_lower = city.lower()
    
    if city_lower not in WEATHER_DATA:
        return f"Error: Weather data not available for '{city}'"
    
    data = WEATHER_DATA[city_lower]
    temp = data["temp"]
    
    if units.lower() == "fahrenheit":
        temp = (temp * 9/5) + 32
        unit_symbol = "°F"
    else:
        unit_symbol = "°C"
    
    return f"Weather in {city.title()}: {temp}{unit_symbol}, {data['condition']}, Humidity: {data['humidity']}%"
```

</details>

### Step 3: Implement `get_forecast`

<details>
<summary>💡 Click for hints</summary>

```python
import random

@mcp.tool()
def get_forecast(city: str, days: int = 3) -> str:
    city_lower = city.lower()
    
    if city_lower not in WEATHER_DATA:
        return f"Error: Forecast not available for '{city}'"
    
    if days < 1 or days > 7:
        return "Error: Days must be between 1 and 7"
    
    base_temp = WEATHER_DATA[city_lower]["temp"]
    conditions = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy", "Clear"]
    
    result = f"Forecast for {city.title()}:\n"
    for day in range(1, days + 1):
        temp = base_temp + random.randint(-5, 5)
        condition = random.choice(conditions)
        result += f"  Day {day}: {temp}°C, {condition}\n"
    
    return result
```

</details>

### Step 4: Implement `convert_temperature`

<details>
<summary>💡 Click for hints</summary>

```python
@mcp.tool()
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    from_unit = from_unit.upper()
    to_unit = to_unit.upper()
    
    # Convert to Celsius first
    if from_unit == "F":
        celsius = (value - 32) * 5/9
    elif from_unit == "K":
        celsius = value - 273.15
    elif from_unit == "C":
        celsius = value
    else:
        return f"Error: Unknown unit '{from_unit}'. Use C, F, or K."
    
    # Convert from Celsius to target
    if to_unit == "F":
        result = (celsius * 9/5) + 32
    elif to_unit == "K":
        result = celsius + 273.15
    elif to_unit == "C":
        result = celsius
    else:
        return f"Error: Unknown unit '{to_unit}'. Use C, F, or K."
    
    return f"{value}°{from_unit} = {result:.2f}°{to_unit}"
```

</details>

### Step 5: Test Your Server

```bash
# Run with MCP Inspector (recommended for visual testing)
fastmcp dev server.py

# Or run directly
python server.py
```

## ✅ Validation Checklist

Before submitting, ensure:

- [ ] All 3 tools are implemented
- [ ] Type hints are correct
- [ ] Docstrings describe the tools
- [ ] Error handling works
- [ ] Server starts without errors
- [ ] Tools work in MCP Inspector

## 🏆 Bonus Challenges

### Level 1: More Features
- [ ] Add wind speed and direction
- [ ] Add UV index
- [ ] Add sunrise/sunset times

### Level 2: Real API (Optional)
- [ ] Integrate with OpenWeatherMap API
- [ ] Add API key handling
- [ ] Implement caching

### Level 3: Advanced
- [ ] Add weather alerts
- [ ] Support coordinates (lat/lon)
- [ ] Historical weather data

## 📤 Expected Output

When running your server:

```
Weather MCP Server
==================
Starting server...
```

In MCP Inspector, you should see 3 tools available to test!

## 📁 Final Project Structure

```
weather-mcp-server/
├── server.py          # Your implementation
└── README.md          # (optional) documentation
```

## 🎓 What You Learned

1. **FastMCP Basics** - Creating an MCP server with minimal code
2. **Tool Decorators** - Using `@mcp.tool()` to expose functions
3. **Type Hints** - How they become JSON schemas automatically
4. **Docstrings** - How they become tool descriptions
5. **Error Handling** - Returning user-friendly error messages
6. **Testing** - Using `fastmcp dev` for interactive testing

## 🔗 Resources

- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Calculator Example](../mcp-servers/01-basic-calculator/)
- [MCP Specification](https://modelcontextprotocol.io)

## ➡️ Next Steps

1. ✅ Complete this exercise
2. ✅ Move to Exercise 2 (Database Integration)
3. ✅ Build your own MCP server idea!

---

**Good luck!** Remember: With FastMCP, building MCP servers is just writing Python functions! 🐍✨
# AI Generated Code by Deloitte + Cursor (END)
