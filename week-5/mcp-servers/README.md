# AI Generated Code by Deloitte + Cursor (BEGIN)
# MCP Server Examples using FastMCP

This directory contains **beginner-friendly** MCP server examples using **FastMCP** - the easiest way to build MCP servers in Python!

## 🎯 Why FastMCP?

| Before (FastAPI + SSE) | After (FastMCP) |
|------------------------|-----------------|
| 500+ lines per server | 50-100 lines |
| Manual protocol handling | Automatic |
| Complex SSE setup | One decorator |
| Confusing for beginners | Intuitive & clean |

## 📁 Server Overview

### 01-basic-calculator ⭐ START HERE
**Difficulty**: Beginner  
**Time**: 15-30 minutes  
**Lines of Code**: ~100

A simple calculator demonstrating core MCP concepts:
- 8 tools (add, subtract, multiply, divide, power, sqrt, factorial, evaluate)
- Clean `@mcp.tool()` decorator pattern
- Automatic schema generation from type hints

```bash
cd 01-basic-calculator
pip install fastmcp
python server.py
```

---

### 02-database-connector
**Difficulty**: Intermediate  
**Time**: 45 minutes  
**Concepts**: Database integration, async operations

PostgreSQL integration with FastMCP:
- Async database queries
- Connection management
- Safe parameterized queries

---

### 03-crm-integration
**Difficulty**: Intermediate  
**Time**: 60 minutes  
**Concepts**: External APIs, authentication

CRM (Salesforce-style) integration:
- OAuth-style authentication
- API integration patterns
- Data transformation

---

### 04-multi-tool-server
**Difficulty**: Advanced  
**Time**: 90 minutes  
**Concepts**: Authentication, RBAC, multiple tool categories

Enterprise-grade multi-tool server:
- API key authentication
- Role-based access
- Multiple tool categories (data, communication, analytics)

---

### 05-financial-services
**Difficulty**: Expert  
**Time**: 120 minutes  
**Concepts**: Case study implementation

Real-world financial services example:
- Trading, risk, customer tools
- Compliance patterns
- Production deployment

---

## 🚀 Getting Started

### 1. Install FastMCP

```bash
pip install fastmcp
```

### 2. Run Your First Server

```bash
cd 01-basic-calculator
python server.py
```

### 3. Test with MCP Inspector

```bash
fastmcp dev server.py
```

This opens a web UI to test your tools!

### 4. Use with Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "calculator": {
      "command": "python",
      "args": ["/full/path/to/01-basic-calculator/server.py"]
    }
  }
}
```

## 📊 Comparison Matrix

| Feature | 01-Calc | 02-DB | 03-CRM | 04-Multi | 05-Finance |
|---------|---------|-------|--------|----------|------------|
| FastMCP | ✅ | ✅ | ✅ | ✅ | ✅ |
| Beginner-Friendly | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐ | ⭐ |
| Lines of Code | ~100 | ~150 | ~200 | ~300 | ~400 |
| Authentication | ❌ | Basic | OAuth | API Key | Multi-tier |
| Database | ❌ | ✅ | ❌ | ✅ | ✅ |
| External APIs | ❌ | ❌ | ✅ | ✅ | ✅ |

## 🎓 Learning Path

### Week 1: Foundations
1. ✅ Complete `01-basic-calculator`
2. ✅ Understand `@mcp.tool()` decorator
3. ✅ Use MCP Inspector for testing
4. ✅ Connect to Claude Desktop

### Week 2: Integration
1. ✅ Complete `02-database-connector`
2. ✅ Complete `03-crm-integration`
3. ✅ Learn async patterns
4. ✅ Understand authentication

### Week 3: Production
1. ✅ Complete `04-multi-tool-server`
2. ✅ Implement RBAC
3. ✅ Add monitoring
4. ✅ Deploy to staging

### Week 4: Enterprise
1. ✅ Study `05-financial-services`
2. ✅ Design your architecture
3. ✅ Build custom server
4. ✅ Deploy to production

## 📝 FastMCP Patterns

### Basic Tool

```python
from fastmcp import FastMCP

mcp = FastMCP("My Server")

@mcp.tool()
def my_tool(param: str) -> str:
    """Description of what this tool does."""
    return f"Result: {param}"

mcp.run()
```

### Tool with Multiple Parameters

```python
@mcp.tool()
def search(query: str, limit: int = 10, include_archived: bool = False) -> str:
    """Search for items.
    
    Args:
        query: Search query string
        limit: Maximum results to return
        include_archived: Include archived items
    """
    # Your implementation
    return results
```

### Async Tool

```python
@mcp.tool()
async def fetch_data(url: str) -> str:
    """Fetch data from a URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text
```

### Resource

```python
@mcp.resource("config://app")
def get_app_config() -> str:
    """Application configuration."""
    return json.dumps({"version": "1.0", "env": "production"})
```

### Prompt Template

```python
@mcp.prompt()
def analyze_data(data_type: str) -> str:
    """Generate analysis prompt."""
    return f"Please analyze the following {data_type} data and provide insights..."
```

## 🔧 Common Commands

```bash
# Install FastMCP
pip install fastmcp

# Run server (stdio - for Claude Desktop)
python server.py

# Run with MCP Inspector (development)
fastmcp dev server.py

# Run with HTTP/SSE transport
fastmcp run server.py --transport sse --port 8000

# Get help
fastmcp --help
```

## 🐛 Debugging Tips

### Check Server Status

```bash
# Run with inspector for debugging
fastmcp dev server.py
```

### Common Issues

**"FastMCP not found"**
```bash
pip install fastmcp
```

**"Server won't start"**
- Check Python version (3.10+)
- Check for syntax errors
- Verify imports

**"Tool not appearing"**
- Ensure `@mcp.tool()` decorator is present
- Check function has type hints
- Add docstring for description

## 📚 Resources

- [FastMCP GitHub](https://github.com/jlowin/fastmcp)
- [MCP Specification](https://modelcontextprotocol.io)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Anthropic MCP Docs](https://docs.anthropic.com/mcp)

## 💡 Tips for Success

1. **Start with 01-basic-calculator** - Build confidence first
2. **Use MCP Inspector** - Visual debugging is easier
3. **Read the docstrings** - They become tool descriptions
4. **Type hints matter** - They define the schema
5. **Test incrementally** - Add one tool at a time
6. **Check the examples** - Working code is the best teacher

## 🎉 Ready to Build?

Start with `01-basic-calculator` - you'll have a working MCP server in under 5 minutes!

```bash
cd 01-basic-calculator
pip install fastmcp
python server.py
```

---

**Questions?** Check individual README files or explore the MCP Inspector!
# AI Generated Code by Deloitte + Cursor (END)
