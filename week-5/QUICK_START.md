# Week 5: MCP Quick Start Guide

Get up and running with MCP using **FastMCP** in 5 minutes! 🚀

## ⚡ Quick Setup

### 1. Install FastMCP (30 seconds)

```bash
pip install fastmcp
```

### 2. Create Your First Server (2 minutes)

Create `server.py`:

```python
from fastmcp import FastMCP

mcp = FastMCP("My First MCP Server")

@mcp.tool()
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}! Welcome to MCP!"

@mcp.tool()
def add(a: float, b: float) -> str:
    """Add two numbers."""
    return f"{a} + {b} = {a + b}"

if __name__ == "__main__":
    mcp.run()
```

### 3. Run Your Server (30 seconds)

```bash
python server.py
```

### 4. Test with MCP Inspector (1 minute)

```bash
fastmcp dev server.py
```

This opens a web UI where you can test your tools!

---

## 📚 What You'll Learn

### Session Outline (4 hours)

**Part 1: MCP Fundamentals (45 min)**
- What is MCP and why it matters
- FastMCP basics
- Building your first server

**Part 2: Building MCP Servers (90 min)**
- Calculator server (beginner)
- Database connector (intermediate)
- CRM integration (intermediate)

**Part 3: Enterprise Patterns (60 min)**
- Authentication & authorization
- Multi-tool servers
- Production deployment

**Part 4: Case Study & Design (75 min)**
- Financial services case study
- Design your own architecture
- Group discussion

---

## 🎯 Learning Path

### Beginner Track (2 hours)
1. ✅ Read this Quick Start
2. ✅ Run `01-basic-calculator` example
3. ✅ Complete Exercise 1 (Weather server)
4. ✅ Connect to Claude Desktop

### Intermediate Track (4 hours)
1. ✅ Complete Beginner Track
2. ✅ Study `02-database-connector`
3. ✅ Study `03-crm-integration`
4. ✅ Complete Exercise 2

### Advanced Track (6+ hours)
1. ✅ Complete Intermediate Track
2. ✅ Study `04-multi-tool-server`
3. ✅ Study `05-financial-services`
4. ✅ Complete Exercise 3
5. ✅ Build your own MCP server

---

## 🔧 Key Commands

```bash
# Install FastMCP
pip install fastmcp

# Run server (stdio - for Claude Desktop)
python server.py

# Run with MCP Inspector (development)
fastmcp dev server.py

# Run with HTTP transport
fastmcp run server.py --transport sse --port 8000

# Get help
fastmcp --help
```

---

## 💻 Using with Claude Desktop

Add to `claude_desktop_config.json`:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "calculator": {
      "command": "python",
      "args": ["/full/path/to/server.py"]
    }
  }
}
```

Then restart Claude Desktop!

---

## 📝 FastMCP Basics

### Tool Decorator

```python
@mcp.tool()
def my_tool(param1: str, param2: int = 10) -> str:
    """Description becomes the tool's description.
    
    Args:
        param1: This becomes the parameter description
        param2: Optional parameter with default
    """
    return f"Result: {param1}, {param2}"
```

### Type Hints → JSON Schema

| Python Type | MCP Schema |
|-------------|-----------|
| `str` | `string` |
| `int` | `integer` |
| `float` | `number` |
| `bool` | `boolean` |
| `list` | `array` |
| `dict` | `object` |

### Error Handling

```python
@mcp.tool()
def safe_divide(a: float, b: float) -> str:
    """Divide two numbers safely."""
    if b == 0:
        return "Error: Cannot divide by zero"
    return f"Result: {a / b}"
```

---

## 📊 Example: Complete Server

```python
from fastmcp import FastMCP
import math

mcp = FastMCP("Calculator", version="1.0.0")

@mcp.tool()
def add(a: float, b: float) -> str:
    """Add two numbers."""
    return f"{a} + {b} = {a + b}"

@mcp.tool()
def multiply(a: float, b: float) -> str:
    """Multiply two numbers."""
    return f"{a} × {b} = {a * b}"

@mcp.tool()
def sqrt(n: float) -> str:
    """Calculate square root."""
    if n < 0:
        return "Error: Cannot calculate square root of negative number"
    return f"√{n} = {math.sqrt(n)}"

if __name__ == "__main__":
    mcp.run()
```

**That's only 25 lines of code for a working MCP server!** 🎉

---

## 🐛 Troubleshooting

### FastMCP not found
```bash
pip install fastmcp
```

### Server won't start
- Check Python version: `python --version` (need 3.10+)
- Check for syntax errors
- Try: `python -c "from fastmcp import FastMCP; print('OK')"`

### Tool not appearing in Claude
- Restart Claude Desktop after config changes
- Check file path in config is correct
- Check Claude's MCP logs

### Inspector not opening
```bash
# Try with explicit host
fastmcp dev server.py --host 127.0.0.1 --port 8000
```

---

## 🎓 Resources

### Documentation
- [FastMCP GitHub](https://github.com/jlowin/fastmcp)
- [MCP Specification](https://modelcontextprotocol.io)
- [Anthropic MCP Docs](https://docs.anthropic.com/mcp)

### Examples
- `mcp-servers/01-basic-calculator/` - Start here!
- `mcp-servers/02-database-connector/` - Database integration
- `mcp-servers/03-crm-integration/` - External APIs
- `mcp-servers/04-multi-tool-server/` - Auth & RBAC
- `mcp-servers/05-financial-services/` - Production patterns

### Exercises
- `exercises/exercise-1-basic-server.md` - Build weather server
- `exercises/exercise-2-database-tools.md` - Database integration
- `exercises/exercise-3-enterprise-design.md` - Architecture design

---

## 💡 Pro Tips

1. **Start Simple** - Begin with 1-2 tools
2. **Use the Inspector** - `fastmcp dev` is your best friend
3. **Type Hints Matter** - They define your schema
4. **Docstrings Are Important** - They become descriptions
5. **Test Incrementally** - Add one tool at a time
6. **Handle Errors** - Return friendly messages

---

## 🚀 Next Steps

1. ✅ Run the calculator example
2. ✅ Complete Exercise 1
3. ✅ Connect to Claude Desktop
4. ✅ Build your own MCP server!

**Questions?** Check the README, explore the examples, or ask!

---

**Ready to build?** Start with `01-basic-calculator`! 🎉

```bash
cd mcp-servers/01-basic-calculator
python server.py
```
