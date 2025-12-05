# AI Generated Code by Deloitte + Cursor (BEGIN)
# Basic Calculator MCP Server

A simple MCP server demonstrating core MCP concepts using **FastMCP** - the easiest way to build MCP servers in Python!

## 🚀 Why FastMCP?

| Old Way (FastAPI + SSE) | New Way (FastMCP) |
|------------------------|-------------------|
| ~550 lines of code | ~100 lines of code |
| Manual protocol handling | Automatic |
| Complex setup | One decorator |
| Hard to debug | Built-in inspector |

## ✨ Features

- ✅ Basic arithmetic (add, subtract, multiply, divide)
- ✅ Advanced math (power, square root, factorial)
- ✅ Safe expression evaluation
- ✅ Clean, readable code
- ✅ Works with Claude Desktop, MCP Inspector, and more!

## 📦 Quick Start

### 1. Install FastMCP

```bash
pip install fastmcp
```

### 2. Run the Server

```bash
# Default (stdio transport - for Claude Desktop)
python server.py

# Or use FastMCP CLI for development
fastmcp dev server.py

# Or run with HTTP/SSE transport
fastmcp run server.py --transport sse --port 8000
```

### 3. Test the Tools

```bash
python client_test.py
```

## 🔧 Available Tools

| Tool | Description | Example |
|------|-------------|---------|
| `add` | Add two numbers | `add(5, 3)` → 8 |
| `subtract` | Subtract numbers | `subtract(10, 4)` → 6 |
| `multiply` | Multiply numbers | `multiply(7, 6)` → 42 |
| `divide` | Divide numbers | `divide(20, 4)` → 5 |
| `power` | Raise to power | `power(2, 8)` → 256 |
| `square_root` | Square root | `square_root(144)` → 12 |
| `factorial` | Factorial | `factorial(5)` → 120 |
| `evaluate` | Evaluate expression | `evaluate("sqrt(16) + 2")` → 6 |

## 🖥️ Using with Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "calculator": {
      "command": "python",
      "args": ["/path/to/server.py"]
    }
  }
}
```

Or with FastMCP CLI:

```json
{
  "mcpServers": {
    "calculator": {
      "command": "fastmcp",
      "args": ["run", "/path/to/server.py"]
    }
  }
}
```

## 🔍 Development with MCP Inspector

FastMCP includes a built-in inspector for testing:

```bash
fastmcp dev server.py
```

This opens a web UI where you can:
- See all available tools
- Test tools with different inputs
- View responses in real-time
- Debug issues easily

## 📝 Code Walkthrough

The entire server is just ~100 lines! Here's the key parts:

### 1. Create the Server

```python
from fastmcp import FastMCP

mcp = FastMCP(
    name="Calculator MCP Server",
    version="1.0.0"
)
```

### 2. Define Tools with Decorators

```python
@mcp.tool()
def add(a: float, b: float) -> str:
    """Add two numbers together."""
    return f"Result: {a} + {b} = {a + b}"
```

That's it! FastMCP automatically:
- Generates the JSON schema from type hints
- Handles protocol messages
- Manages connections
- Provides error handling

### 3. Run the Server

```python
if __name__ == "__main__":
    mcp.run()
```

## 🧪 Testing

### Direct Function Testing

```bash
python client_test.py
```

### MCP Inspector Testing

```bash
fastmcp dev server.py
```

### With Claude Desktop

After configuring, just ask Claude:
- "What's 25 times 17?"
- "Calculate the square root of 256"
- "Evaluate sin(pi/2) + cos(0)"

## 🎓 Learning Objectives

This example teaches:

1. **FastMCP Basics** - Creating servers with minimal code
2. **Tool Decorators** - The `@mcp.tool()` pattern
3. **Type Hints** - How they become JSON schemas
4. **Docstrings** - How they become tool descriptions
5. **Error Handling** - Returning error messages gracefully
6. **Testing** - Direct and MCP Inspector testing

## 📚 Next Steps

1. ✅ Try modifying the tools
2. ✅ Add a new tool (e.g., percentage, average)
3. ✅ Move to `02-database-connector`
4. ✅ Build your own MCP server!

## 🔗 Resources

- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [MCP Specification](https://modelcontextprotocol.io)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)

---

**Remember**: With FastMCP, building MCP servers is as easy as writing Python functions! 🐍✨
# AI Generated Code by Deloitte + Cursor (END)
