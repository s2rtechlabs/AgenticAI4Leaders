"""
Basic Calculator MCP Server using FastMCP
==========================================

A simple MCP server demonstrating core MCP concepts through mathematical operations.
This uses FastMCP - the easiest way to build MCP servers in Python!

Features:
- Basic arithmetic (add, subtract, multiply, divide)
- Advanced math (power, square root, factorial)
- Expression evaluation
- Minimal code, maximum functionality!

Run with:
    python server.py

Or with specific transport:
    fastmcp run server.py --transport stdio
    fastmcp run server.py --transport sse --port 8000
"""

from fastmcp import FastMCP
import math

# Create the MCP server - it's this easy!
mcp = FastMCP(
    name="Calculator MCP Server",
    version="1.0.0",
)


# ============================================================
# BASIC ARITHMETIC TOOLS
# ============================================================

@mcp.tool()
def add(a: float, b: float) -> str:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        The sum of a and b
    """
    result = a + b
    return f"Result: {a} + {b} = {result}"


@mcp.tool()
def subtract(a: float, b: float) -> str:
    """Subtract the second number from the first.
    
    Args:
        a: First number (minuend)
        b: Second number (subtrahend)
    
    Returns:
        The difference of a minus b
    """
    result = a - b
    return f"Result: {a} - {b} = {result}"


@mcp.tool()
def multiply(a: float, b: float) -> str:
    """Multiply two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        The product of a and b
    """
    result = a * b
    return f"Result: {a} × {b} = {result}"


@mcp.tool()
def divide(a: float, b: float) -> str:
    """Divide the first number by the second.
    
    Args:
        a: Dividend
        b: Divisor (cannot be zero)
    
    Returns:
        The quotient of a divided by b
    """
    if b == 0:
        return "Error: Division by zero is not allowed"
    result = a / b
    return f"Result: {a} ÷ {b} = {result}"


# ============================================================
# ADVANCED MATH TOOLS
# ============================================================

@mcp.tool()
def power(base: float, exponent: float) -> str:
    """Raise a number to a power.
    
    Args:
        base: The base number
        exponent: The exponent
    
    Returns:
        base raised to the power of exponent
    """
    try:
        result = math.pow(base, exponent)
        return f"Result: {base}^{exponent} = {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def square_root(number: float) -> str:
    """Calculate the square root of a number.
    
    Args:
        number: The number to find the square root of (must be non-negative)
    
    Returns:
        The square root of the number
    """
    if number < 0:
        return "Error: Cannot calculate square root of negative number"
    result = math.sqrt(number)
    return f"Result: √{number} = {result}"


@mcp.tool()
def factorial(n: int) -> str:
    """Calculate the factorial of a non-negative integer.
    
    Args:
        n: A non-negative integer
    
    Returns:
        The factorial of n (n!)
    """
    if n < 0:
        return "Error: Factorial is only defined for non-negative integers"
    if n > 170:
        return "Error: Number too large (max 170)"
    result = math.factorial(n)
    return f"Result: {n}! = {result}"


@mcp.tool()
def evaluate(expression: str) -> str:
    """Safely evaluate a mathematical expression.
    
    Supports: +, -, *, /, sqrt(), pow(), sin(), cos(), tan(), log(), exp(), pi, e
    
    Args:
        expression: Mathematical expression (e.g., 'sqrt(16) + pow(2, 3)')
    
    Returns:
        The result of evaluating the expression
    """
    try:
        # Safe evaluation - only allow math operations
        allowed_names = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'pow': pow, 'sum': sum,
            'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
            'tan': math.tan, 'log': math.log, 'exp': math.exp,
            'pi': math.pi, 'e': math.e
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {expression} = {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


# ============================================================
# RUN THE SERVER
# ============================================================

if __name__ == "__main__":
    # Print server info
    print("=" * 60)
    print("Calculator MCP Server (FastMCP)")
    print("=" * 60)
    print(f"Server: {mcp.name}")
    print(f"Version: {mcp.version}")
    print(f"Tools: 8")
    print("\nAvailable tools:")
    print("  - add")
    print("  - subtract")
    print("  - multiply")
    print("  - divide")
    print("  - power")
    print("  - square_root")
    print("  - factorial")
    print("  - evaluate")
    print("\nStarting server...")
    print("=" * 60)
    
    # Run the server (default: stdio transport)
    # For HTTP/SSE: fastmcp run server.py --transport sse --port 8000
    mcp.run()
