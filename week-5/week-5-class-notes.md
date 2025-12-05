# Week 5: MCP (Model Context Protocol) - Class Notes

## Table of Contents
1. [Introduction to MCP](#introduction-to-mcp)
2. [FastMCP: The Easy Way](#fastmcp-the-easy-way)
3. [Building MCP Servers](#building-mcp-servers)
4. [Protocol Deep Dive](#protocol-deep-dive)
5. [Security Framework](#security-framework)
6. [Enterprise Integration Patterns](#enterprise-integration-patterns)
7. [Case Study: Financial Services](#case-study)
8. [Best Practices](#best-practices)

---

## Introduction to MCP

### What is the Model Context Protocol (MCP)?

The Model Context Protocol (MCP) is an **open standard** that enables seamless integration between AI applications and external data sources. Think of it as a universal adapter for AI systems.

**Key Analogy**: Just as USB-C standardized device connectivity, MCP standardizes AI tool integration.

### Why MCP Matters

**Before MCP:**
- Each AI vendor required custom integrations
- Security implementations varied wildly
- No standardized governance
- High maintenance costs
- Vendor lock-in

**With MCP:**
- ✅ Single integration standard
- ✅ Consistent security framework
- ✅ Centralized governance
- ✅ 70% reduction in integration time
- ✅ Vendor-agnostic architecture

### The MCP Ecosystem

```
┌─────────────────┐     ┌─────────────────┐
│   MCP Client    │     │   MCP Client    │
│ (Claude Desktop)│     │  (Custom Agent) │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │     MCP Server        │
         │  (Your Tools/Data)    │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   Enterprise Systems  │
         │ (DBs, APIs, Services) │
         └───────────────────────┘
```

---

## FastMCP: The Easy Way

### What is FastMCP?

FastMCP is a Python framework that makes building MCP servers incredibly simple. It's the **recommended** way to build MCP servers.

### Why FastMCP?

| Traditional (FastAPI + SSE) | FastMCP |
|---------------------------|---------|
| 500+ lines of code | 50-100 lines |
| Manual protocol handling | Automatic |
| Complex SSE setup | One decorator |
| Confusing for beginners | Intuitive |
| Manual schema creation | Auto-generated from type hints |

### Your First FastMCP Server

```python
from fastmcp import FastMCP

# Create server
mcp = FastMCP("My Server")

# Define a tool
@mcp.tool()
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

# Run it
if __name__ == "__main__":
    mcp.run()
```

**That's it!** 🎉 In just 10 lines, you have a working MCP server.

### Key FastMCP Features

#### 1. The `@mcp.tool()` Decorator

```python
@mcp.tool()
def my_tool(param1: str, param2: int = 10) -> str:
    """Tool description (becomes MCP description).
    
    Args:
        param1: First parameter description
        param2: Second parameter with default
    """
    return f"Result: {param1}, {param2}"
```

FastMCP automatically:
- Generates JSON schema from type hints
- Extracts description from docstring
- Handles protocol messages
- Manages errors

#### 2. Resources

```python
@mcp.resource("config://app")
def get_config() -> str:
    """Application configuration."""
    return json.dumps({"version": "1.0"})
```

#### 3. Prompts

```python
@mcp.prompt()
def analyze_prompt(data_type: str) -> str:
    """Generate analysis prompt."""
    return f"Please analyze this {data_type} data..."
```

### Type Hints → JSON Schema

| Python Type | MCP Schema Type |
|-------------|----------------|
| `str` | `string` |
| `int` | `integer` |
| `float` | `number` |
| `bool` | `boolean` |
| `list` | `array` |
| `dict` | `object` |
| `Optional[str]` | `string` (optional) |

### Running FastMCP Servers

```bash
# stdio transport (for Claude Desktop)
python server.py

# MCP Inspector (development/testing)
fastmcp dev server.py

# HTTP/SSE transport (for web deployment)
fastmcp run server.py --transport sse --port 8000
```

---

## Building MCP Servers

### Example 1: Calculator Server

```python
from fastmcp import FastMCP
import math

mcp = FastMCP("Calculator", version="1.0.0")

@mcp.tool()
def add(a: float, b: float) -> str:
    """Add two numbers together."""
    return f"Result: {a} + {b} = {a + b}"

@mcp.tool()
def divide(a: float, b: float) -> str:
    """Divide first number by second."""
    if b == 0:
        return "Error: Cannot divide by zero"
    return f"Result: {a} / {b} = {a / b}"

@mcp.tool()
def sqrt(n: float) -> str:
    """Calculate square root."""
    if n < 0:
        return "Error: Cannot calculate sqrt of negative number"
    return f"√{n} = {math.sqrt(n)}"

if __name__ == "__main__":
    mcp.run()
```

### Example 2: Database Connector

```python
from fastmcp import FastMCP

mcp = FastMCP("Database Connector")

# Mock database
CUSTOMERS = [
    {"id": 1, "name": "John", "country": "USA"},
    {"id": 2, "name": "Jane", "country": "UK"},
]

@mcp.tool()
def query_customers(country: str = None) -> str:
    """Query customers, optionally filtered by country."""
    results = CUSTOMERS
    if country:
        results = [c for c in results if c["country"] == country]
    return str(results)

@mcp.tool()
def get_customer(customer_id: int) -> str:
    """Get a specific customer by ID."""
    customer = next((c for c in CUSTOMERS if c["id"] == customer_id), None)
    if not customer:
        return f"Error: Customer {customer_id} not found"
    return str(customer)
```

### Example 3: With Authentication

```python
from fastmcp import FastMCP

mcp = FastMCP("Secure Server")

VALID_API_KEYS = {"admin-key-123": "admin", "user-key-456": "user"}

def check_auth(api_key: str) -> tuple:
    """Validate API key and return role."""
    if api_key not in VALID_API_KEYS:
        return None, "Invalid API key"
    return VALID_API_KEYS[api_key], None

@mcp.tool()
def admin_action(api_key: str, action: str) -> str:
    """Perform admin action (requires admin API key).
    
    Args:
        api_key: Your API key
        action: Action to perform
    """
    role, error = check_auth(api_key)
    if error:
        return f"Error: {error}"
    if role != "admin":
        return "Error: Admin access required"
    return f"Admin action '{action}' completed"
```

---

## Protocol Deep Dive

### MCP Message Format

MCP uses JSON-RPC 2.0:

```json
// Request
{
  "jsonrpc": "2.0",
  "id": "123",
  "method": "tools/call",
  "params": {
    "name": "add",
    "arguments": {"a": 5, "b": 3}
  }
}

// Response
{
  "jsonrpc": "2.0",
  "id": "123",
  "result": {
    "content": [{"type": "text", "text": "Result: 8"}]
  }
}
```

### Transport Layers

| Transport | Use Case | FastMCP Command |
|-----------|----------|-----------------|
| stdio | Claude Desktop, CLI | `python server.py` |
| SSE/HTTP | Web deployment | `fastmcp run server.py --transport sse` |
| WebSocket | Real-time apps | Custom implementation |

### MCP Capabilities

```python
# FastMCP handles these automatically:
- initialize (server capabilities)
- tools/list (available tools)
- tools/call (execute tool)
- resources/list (available resources)
- resources/read (read resource)
- prompts/list (available prompts)
- prompts/get (get prompt)
```

---

## Security Framework

### Authentication Patterns

#### 1. API Key (Simple)

```python
@mcp.tool()
def secure_action(api_key: str, action: str) -> str:
    if api_key != "valid-key":
        return "Error: Invalid API key"
    return f"Action {action} completed"
```

#### 2. Role-Based Access Control (RBAC)

```python
ROLES = {
    "admin": ["read", "write", "delete"],
    "user": ["read"],
    "editor": ["read", "write"],
}

def has_permission(role: str, action: str) -> bool:
    return action in ROLES.get(role, [])

@mcp.tool()
def protected_action(role: str, action: str) -> str:
    if not has_permission(role, action):
        return f"Error: {role} cannot perform {action}"
    return f"Action {action} completed"
```

### Security Checklist

- [ ] Use parameterized queries (prevent SQL injection)
- [ ] Validate all inputs
- [ ] Implement authentication
- [ ] Use HTTPS/TLS in production
- [ ] Enable audit logging
- [ ] Set rate limits
- [ ] Don't expose sensitive data in errors

### SQL Injection Prevention

```python
# ❌ NEVER DO THIS
query = f"SELECT * FROM users WHERE id = {user_input}"

# ✅ ALWAYS DO THIS
query = "SELECT * FROM users WHERE id = $1"
await conn.fetch(query, user_input)
```

---

## Enterprise Integration Patterns

### Pattern 1: Database Integration

```python
from fastmcp import FastMCP
import asyncpg

mcp = FastMCP("Database MCP")

@mcp.tool()
async def query_db(sql: str, table: str) -> str:
    """Query the database safely."""
    # Only allow SELECT on approved tables
    allowed_tables = ["customers", "orders", "products"]
    if table not in allowed_tables:
        return f"Error: Table {table} not accessible"
    
    conn = await asyncpg.connect(...)
    try:
        rows = await conn.fetch(f"SELECT * FROM {table} LIMIT 100")
        return str([dict(r) for r in rows])
    finally:
        await conn.close()
```

### Pattern 2: API Integration

```python
import httpx

@mcp.tool()
async def call_external_api(endpoint: str) -> str:
    """Call an external API."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/{endpoint}")
        return response.text
```

### Pattern 3: Multi-Service Orchestration

```python
@mcp.tool()
async def get_customer_360(customer_id: str) -> str:
    """Get complete customer view from multiple sources."""
    # Fetch from multiple services in parallel
    crm_data = await get_crm_data(customer_id)
    orders = await get_orders(customer_id)
    support = await get_support_tickets(customer_id)
    
    return f"""
Customer 360 for {customer_id}:
CRM: {crm_data}
Orders: {orders}
Support: {support}
"""
```

---

## Case Study: Financial Services

### Company Profile
- **Industry**: Global Investment Bank
- **Size**: 50,000+ employees
- **Challenge**: 50+ enterprise tools, fragmented integrations

### Implementation Results

| Metric | Before MCP | After MCP | Improvement |
|--------|-----------|-----------|-------------|
| Integration Time | 4-6 weeks | 3-5 days | **70% faster** |
| Maintenance Cost | $500K/year | $300K/year | **40% reduction** |
| Security Incidents | 12/year | 0/year | **100% reduction** |
| Developer Satisfaction | 6/10 | 9/10 | **50% increase** |

### Key Learnings

1. **Start Small** - Pilot with 5 critical tools
2. **Security First** - Auth + audit from day one
3. **Monitor Everything** - Comprehensive logging
4. **Train Teams** - Developer education is critical
5. **Iterate** - Continuous improvement

---

## Best Practices

### Tool Design

1. **Single Responsibility** - One tool, one purpose
2. **Clear Names** - `get_customer` not `gc`
3. **Comprehensive Docstrings** - They become descriptions
4. **Type Hints** - Required for schema generation
5. **Error Handling** - Return friendly messages

### Code Example

   ```python
@mcp.tool()
def get_customer_orders(
    customer_id: int,
    status: str = "all",
    limit: int = 100
) -> str:
    """Get orders for a specific customer.
    
    Args:
        customer_id: The customer's unique identifier
        status: Filter by status: 'pending', 'completed', 'all'
        limit: Maximum number of orders to return (default: 100)
    
    Returns:
        A formatted list of orders or an error message
    """
    # Validate inputs
    if customer_id < 1:
        return "Error: Invalid customer ID"
    if status not in ["pending", "completed", "all"]:
        return f"Error: Invalid status '{status}'"
    if limit < 1 or limit > 1000:
        return "Error: Limit must be between 1 and 1000"
    
    # Get orders
    orders = fetch_orders(customer_id, status, limit)
    
    if not orders:
        return f"No orders found for customer {customer_id}"
    
    # Format output
    result = f"Orders for customer {customer_id}:\n"
    for order in orders:
        result += f"  - Order #{order['id']}: ${order['amount']}\n"
    return result
```

### Testing

```bash
# Use MCP Inspector for development
fastmcp dev server.py

# Write unit tests for tools
def test_add():
    result = add(5, 3)
    assert "8" in result
```

### Deployment

```bash
# Development
python server.py

# Production (with process manager)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker server:app

# Docker
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install fastmcp
CMD ["python", "server.py"]
```

---

## Summary

### Key Takeaways

1. **MCP is the Standard** - For AI tool integration
2. **FastMCP is the Way** - Build servers with minimal code
3. **Security Matters** - Multi-layer security from day one
4. **Start Simple** - Pilot, learn, scale
5. **Monitor Everything** - Logging and metrics are critical

### Next Steps

1. Complete the hands-on exercises
2. Build your first MCP server
3. Connect to Claude Desktop
4. Design your enterprise architecture
5. Deploy to production

### Resources

- [FastMCP GitHub](https://github.com/jlowin/fastmcp)
- [MCP Specification](https://modelcontextprotocol.io)
- [Anthropic MCP Docs](https://docs.anthropic.com/mcp)
- [Example Servers](./mcp-servers/)
- [Case Study](./case-study/)

---

**Remember**: With FastMCP, building MCP servers is as easy as writing Python functions! 🐍✨
# AI Generated Code by Deloitte + Cursor (END)
