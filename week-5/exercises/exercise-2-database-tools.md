# Exercise 2: Database Integration MCP Server

**Difficulty**: Intermediate  
**Time**: 45-60 minutes  
**Prerequisites**: Exercise 1 completed, basic SQL knowledge

## 🎯 Objective

Build an MCP server that safely integrates with a database using FastMCP. You'll learn parameterized queries, connection management, and security best practices.

## 📚 Learning Goals

- ✅ Async database operations
- ✅ SQL injection prevention
- ✅ Connection management
- ✅ Error handling for database operations
- ✅ Security best practices

## 📦 Setup

### 1. Install Dependencies

```bash
pip install fastmcp asyncpg python-dotenv
```

### 2. Set Up PostgreSQL (Optional)

If you have PostgreSQL:

```bash
# Docker option
docker run --name postgres-mcp \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=testdb \
  -p 5432:5432 \
  -d postgres:15-alpine
```

**OR** use the mock database approach (no PostgreSQL needed!)

## 🔧 Tools to Implement

### 1. `query_customers`

```python
@mcp.tool()
async def query_customers(country: str = None, limit: int = 100) -> str:
    """Query customers from the database.
    
    Args:
        country: Filter by country (optional)
        limit: Maximum results (default: 100)
    """
```

### 2. `get_order_history`

```python
@mcp.tool()
async def get_order_history(customer_id: int) -> str:
    """Get order history for a customer.
    
    Args:
        customer_id: The customer ID to look up
    """
```

### 3. `get_customer_stats`

```python
@mcp.tool()
async def get_customer_stats() -> str:
    """Get overall customer statistics."""
```

## 📝 Implementation Guide

### Option A: Mock Database (No PostgreSQL Needed)

```python
from fastmcp import FastMCP

mcp = FastMCP("Database Connector", version="1.0.0")

# Mock database
CUSTOMERS = [
    {"id": 1, "name": "John Doe", "email": "john@example.com", "country": "USA"},
    {"id": 2, "name": "Jane Smith", "email": "jane@example.com", "country": "UK"},
    {"id": 3, "name": "Bob Wilson", "email": "bob@example.com", "country": "USA"},
    {"id": 4, "name": "Alice Brown", "email": "alice@example.com", "country": "Canada"},
]

ORDERS = [
    {"id": 1, "customer_id": 1, "product": "Widget A", "amount": 99.99},
    {"id": 2, "customer_id": 1, "product": "Widget B", "amount": 149.99},
    {"id": 3, "customer_id": 2, "product": "Widget C", "amount": 79.99},
    {"id": 4, "customer_id": 3, "product": "Widget A", "amount": 99.99},
]

@mcp.tool()
def query_customers(country: str = None, limit: int = 100) -> str:
    """Query customers from the database.
    
    Args:
        country: Filter by country (optional)
        limit: Maximum results (default: 100)
    """
    results = CUSTOMERS
    
    if country:
        results = [c for c in results if c["country"].lower() == country.lower()]
    
    results = results[:limit]
    
    if not results:
        return "No customers found"
    
    output = f"Found {len(results)} customer(s):\n"
    for c in results:
        output += f"  [{c['id']}] {c['name']} ({c['email']}) - {c['country']}\n"
    return output

@mcp.tool()
def get_order_history(customer_id: int) -> str:
    """Get order history for a customer.
    
    Args:
        customer_id: The customer ID to look up
    """
    # Find customer
    customer = next((c for c in CUSTOMERS if c["id"] == customer_id), None)
    if not customer:
        return f"Error: Customer {customer_id} not found"
    
    # Get orders
    orders = [o for o in ORDERS if o["customer_id"] == customer_id]
    
    if not orders:
        return f"No orders found for {customer['name']}"
    
    total = sum(o["amount"] for o in orders)
    output = f"Orders for {customer['name']}:\n"
    for o in orders:
        output += f"  - {o['product']}: ${o['amount']}\n"
    output += f"\nTotal: ${total:.2f}"
    return output

@mcp.tool()
def get_customer_stats() -> str:
    """Get overall customer statistics."""
    total_customers = len(CUSTOMERS)
    total_orders = len(ORDERS)
    total_revenue = sum(o["amount"] for o in ORDERS)
    countries = set(c["country"] for c in CUSTOMERS)
    
    return f"""Customer Statistics:
  Total Customers: {total_customers}
  Total Orders: {total_orders}
  Total Revenue: ${total_revenue:.2f}
  Average Order: ${total_revenue/total_orders:.2f}
  Countries: {', '.join(countries)}"""

if __name__ == "__main__":
    mcp.run()
```

### Option B: Real PostgreSQL (Advanced)

```python
from fastmcp import FastMCP
import asyncpg
import os

mcp = FastMCP("Database Connector", version="1.0.0")

async def get_connection():
    return await asyncpg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password"),
        database=os.getenv("DB_NAME", "testdb")
    )

@mcp.tool()
async def query_customers(country: str = None, limit: int = 100) -> str:
    """Query customers from the database."""
    conn = await get_connection()
    try:
        if country:
            # SAFE: Parameterized query prevents SQL injection!
            query = "SELECT * FROM customers WHERE country = $1 LIMIT $2"
            rows = await conn.fetch(query, country, limit)
        else:
            query = "SELECT * FROM customers LIMIT $1"
            rows = await conn.fetch(query, limit)
        
        if not rows:
            return "No customers found"
        
        output = f"Found {len(rows)} customer(s):\n"
        for row in rows:
            output += f"  [{row['id']}] {row['name']} ({row['email']})\n"
        return output
    finally:
        await conn.close()

# ... more tools
```

## 🔒 Security: SQL Injection Prevention

### ❌ NEVER DO THIS

```python
# DANGEROUS! SQL injection vulnerability!
query = f"SELECT * FROM users WHERE country = '{user_input}'"
```

### ✅ ALWAYS DO THIS

```python
# SAFE! Parameters are escaped automatically
query = "SELECT * FROM users WHERE country = $1"
rows = await conn.fetch(query, user_input)
```

## 🧪 Testing

```bash
# Run with MCP Inspector
fastmcp dev server.py
```

Test scenarios:
1. Query all customers
2. Filter by country
3. Get orders for valid customer ID
4. Get orders for invalid customer ID (error handling)
5. Get statistics

## ✅ Validation Checklist

- [ ] All 3 tools implemented
- [ ] Parameterized queries used (if using real DB)
- [ ] Error handling for invalid inputs
- [ ] Type hints correct
- [ ] Docstrings describe tools
- [ ] Server runs without errors

## 🏆 Bonus Challenges

### Level 1: More Features
- [ ] Add `search_customers(query: str)` - full-text search
- [ ] Add pagination support
- [ ] Add sorting options

### Level 2: Write Operations
- [ ] Add `add_customer(name, email, country)`
- [ ] Add `update_customer(id, updates)`
- [ ] Implement permission checks

### Level 3: Advanced
- [ ] Connection pooling
- [ ] Query caching
- [ ] Audit logging

## 📁 Final Project Structure

```
database-mcp-server/
├── server.py          # Your implementation
├── .env               # Database credentials (don't commit!)
└── README.md          # Documentation
```

## 🎓 What You Learned

1. **Database Integration** - Connecting MCP to databases
2. **SQL Injection Prevention** - Parameterized queries
3. **Async Operations** - Non-blocking database calls
4. **Error Handling** - Graceful failure for missing data
5. **Security** - Protecting against common attacks

## ➡️ Next Steps

1. ✅ Complete this exercise
2. ✅ Move to Exercise 3 (Enterprise Design)
3. ✅ Study the CRM integration example

---

**Remember**: Always use parameterized queries! Never trust user input! 🔒
