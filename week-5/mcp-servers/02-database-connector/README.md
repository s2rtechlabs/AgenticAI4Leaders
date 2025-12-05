# Database Connector MCP Server

MCP server for PostgreSQL database integration using **FastMCP**.

## Overview

This server demonstrates how to safely integrate AI systems with relational databases using MCP and FastMCP. It includes connection management, parameterized queries, and security best practices.

## ✨ Features

- ✅ PostgreSQL connectivity with asyncpg
- ✅ Safe parameterized queries (SQL injection prevention)
- ✅ Connection management
- ✅ Read/write separation
- ✅ Clean FastMCP implementation

## 📦 Quick Start

### 1. Install Dependencies

```bash
pip install fastmcp asyncpg python-dotenv
```

### 2. Set Up PostgreSQL

```bash
# Option A: Docker
docker run --name postgres-mcp \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=testdb \
  -p 5432:5432 \
  -d postgres:15-alpine

# Option B: Use existing PostgreSQL
```

### 3. Configure Environment

Create `.env` file:

```env
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=password
DB_NAME=testdb
```

### 4. Run Server

```bash
python server.py
```

## 🔧 Example Implementation

```python
from fastmcp import FastMCP
import asyncpg
import os

mcp = FastMCP("Database Connector")

# Database connection
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
    """Query customers from the database.
    
    Args:
        country: Filter by country (optional)
        limit: Maximum number of results (default: 100)
    """
    conn = await get_connection()
    try:
        if country:
            # Safe parameterized query - prevents SQL injection!
            query = "SELECT * FROM customers WHERE country = $1 LIMIT $2"
            rows = await conn.fetch(query, country, limit)
        else:
            query = "SELECT * FROM customers LIMIT $1"
            rows = await conn.fetch(query, limit)
        
        return str([dict(row) for row in rows])
    finally:
        await conn.close()

@mcp.tool()
async def get_order_history(customer_id: int) -> str:
    """Get order history for a customer.
    
    Args:
        customer_id: The customer ID to look up
    """
    conn = await get_connection()
    try:
        query = """
            SELECT o.*, c.name as customer_name
            FROM orders o
            JOIN customers c ON o.customer_id = c.id
            WHERE o.customer_id = $1
            ORDER BY o.order_date DESC
        """
        rows = await conn.fetch(query, customer_id)
        return str([dict(row) for row in rows])
    finally:
        await conn.close()

@mcp.tool()
async def get_table_schema(table_name: str) -> str:
    """Get the schema of a database table.
    
    Args:
        table_name: Name of the table to inspect
    """
    conn = await get_connection()
    try:
        query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = $1
            ORDER BY ordinal_position
        """
        rows = await conn.fetch(query, table_name)
        return str([dict(row) for row in rows])
    finally:
        await conn.close()

if __name__ == "__main__":
    mcp.run()
```

## 🔒 Security Best Practices

### ✅ DO: Use Parameterized Queries

```python
# SAFE - parameters are escaped automatically
query = "SELECT * FROM users WHERE id = $1"
await conn.fetch(query, user_id)
```

### ❌ DON'T: Concatenate SQL

```python
# DANGEROUS - SQL injection vulnerability!
query = f"SELECT * FROM users WHERE id = {user_id}"
```

### Permission Separation

```python
@mcp.tool()
async def read_data(table: str) -> str:
    """Read-only access to data."""
    # Only SELECT allowed
    pass

@mcp.tool()  
async def write_data(table: str, data: dict) -> str:
    """Write access to data (requires admin)."""
    # Check permissions before INSERT/UPDATE
    pass
```

## 🧪 Testing

```bash
# Run server
python server.py

# Test with MCP Inspector
fastmcp dev server.py
```

## 📝 Sample Database Setup

```sql
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    country VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    product VARCHAR(100),
    amount DECIMAL(10, 2),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sample data
INSERT INTO customers (name, email, country) VALUES
    ('John Doe', 'john@example.com', 'USA'),
    ('Jane Smith', 'jane@example.com', 'UK');

INSERT INTO orders (customer_id, product, amount) VALUES
    (1, 'Widget A', 99.99),
    (1, 'Widget B', 149.99);
```

## 🎓 Learning Objectives

- Async database operations with FastMCP
- SQL injection prevention
- Connection management
- Query parameterization
- Error handling for database operations

## 📚 Resources

- [asyncpg Documentation](https://magicstack.github.io/asyncpg/)
- [SQL Injection Prevention](https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)

---

**Remember**: Always use parameterized queries! Never trust user input! 🔒
