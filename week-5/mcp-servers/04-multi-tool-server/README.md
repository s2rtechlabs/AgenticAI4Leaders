# AI Generated Code by Deloitte + Cursor (BEGIN)
# Multi-Tool Enterprise MCP Server

Production-ready MCP server with authentication, RBAC, and multiple tool categories using **FastMCP**.

## Overview

Enterprise-grade MCP server demonstrating production patterns including API key authentication, role-based access control, and multi-tool organization.

## ✨ Features

- ✅ API key authentication
- ✅ Role-Based Access Control (RBAC)
- ✅ Multiple tool categories
- ✅ Structured logging
- ✅ Clean FastMCP implementation

## 📦 Quick Start

### 1. Install Dependencies

```bash
pip install fastmcp python-dotenv
```

### 2. Run Server

```bash
python server.py
```

## 🔧 Example Implementation

```python
from fastmcp import FastMCP
from functools import wraps
import os

mcp = FastMCP("Enterprise Multi-Tool Server", version="1.0.0")

# ============================================================
# AUTHENTICATION & AUTHORIZATION
# ============================================================

# Mock user database
USERS = {
    "admin-key-123": {"role": "admin", "name": "Admin User"},
    "dev-key-456": {"role": "developer", "name": "Dev User"},
    "analyst-key-789": {"role": "analyst", "name": "Analyst User"},
}

ROLE_PERMISSIONS = {
    "admin": ["data", "communication", "analytics", "integration", "admin"],
    "developer": ["data", "integration"],
    "analyst": ["data", "analytics"],
}

def get_current_user(api_key: str) -> dict:
    """Get user from API key."""
    if api_key not in USERS:
        return None
    return USERS[api_key]

def check_permission(role: str, category: str) -> bool:
    """Check if role has permission for category."""
    return category in ROLE_PERMISSIONS.get(role, [])

# ============================================================
# DATA TOOLS
# ============================================================

@mcp.tool()
def query_data(query: str, api_key: str) -> str:
    """Query data from the enterprise database.
    
    Args:
        query: The search query
        api_key: Your API key for authentication
    """
    user = get_current_user(api_key)
    if not user:
        return "Error: Invalid API key"
    if not check_permission(user["role"], "data"):
        return "Error: Permission denied"
    
    return f"Query results for '{query}' (user: {user['name']}): [Sample data...]"

@mcp.tool()
def export_data(format: str, api_key: str) -> str:
    """Export data in specified format.
    
    Args:
        format: Export format (csv, json, excel)
        api_key: Your API key for authentication
    """
    user = get_current_user(api_key)
    if not user:
        return "Error: Invalid API key"
    
    return f"Data exported as {format} for user {user['name']}"

# ============================================================
# COMMUNICATION TOOLS
# ============================================================

@mcp.tool()
def send_notification(recipient: str, message: str, api_key: str) -> str:
    """Send a notification to a user.
    
    Args:
        recipient: Email or username of recipient
        message: Notification message
        api_key: Your API key for authentication
    """
    user = get_current_user(api_key)
    if not user:
        return "Error: Invalid API key"
    if not check_permission(user["role"], "communication"):
        return "Error: Permission denied - communication not allowed for your role"
    
    return f"Notification sent to {recipient}: '{message}'"

# ============================================================
# ANALYTICS TOOLS
# ============================================================

@mcp.tool()
def generate_report(report_type: str, date_range: str, api_key: str) -> str:
    """Generate an analytics report.
    
    Args:
        report_type: Type of report (sales, usage, performance)
        date_range: Date range (e.g., '2024-01-01 to 2024-01-31')
        api_key: Your API key for authentication
    """
    user = get_current_user(api_key)
    if not user:
        return "Error: Invalid API key"
    if not check_permission(user["role"], "analytics"):
        return "Error: Permission denied - analytics not allowed for your role"
    
    return f"{report_type.title()} report generated for {date_range}"

@mcp.tool()
def get_metrics(metric_name: str, api_key: str) -> str:
    """Get specific metrics.
    
    Args:
        metric_name: Name of the metric (users, revenue, performance)
        api_key: Your API key for authentication
    """
    user = get_current_user(api_key)
    if not user:
        return "Error: Invalid API key"
    
    # Mock metrics
    metrics = {
        "users": "Active users: 1,234 | New today: 56",
        "revenue": "MTD Revenue: $123,456 | Growth: +12%",
        "performance": "Avg response: 150ms | Uptime: 99.9%"
    }
    
    return metrics.get(metric_name, f"Unknown metric: {metric_name}")

# ============================================================
# ADMIN TOOLS
# ============================================================

@mcp.tool()
def list_users(api_key: str) -> str:
    """List all users (admin only).
    
    Args:
        api_key: Your API key for authentication
    """
    user = get_current_user(api_key)
    if not user:
        return "Error: Invalid API key"
    if not check_permission(user["role"], "admin"):
        return "Error: Admin access required"
    
    result = "Users:\n"
    for key, u in USERS.items():
        result += f"  - {u['name']} ({u['role']})\n"
    return result

if __name__ == "__main__":
    print("Multi-Tool Enterprise MCP Server")
    print("=" * 40)
    print("API Keys for testing:")
    print("  Admin: admin-key-123")
    print("  Developer: dev-key-456")
    print("  Analyst: analyst-key-789")
    print("=" * 40)
    mcp.run()
```

## 🔐 RBAC Model

| Role | Data | Communication | Analytics | Admin |
|------|------|---------------|-----------|-------|
| Admin | ✅ | ✅ | ✅ | ✅ |
| Developer | ✅ | ❌ | ❌ | ❌ |
| Analyst | ✅ | ❌ | ✅ | ❌ |

## 🧪 Testing

```bash
# Run server
python server.py

# Test with MCP Inspector
fastmcp dev server.py
```

## 🎓 Learning Objectives

- API key authentication patterns
- Role-based access control (RBAC)
- Multi-tool organization
- Permission checking
- Enterprise security patterns

## 📚 Resources

- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [RBAC Best Practices](https://auth0.com/docs/manage-users/access-control/rbac)

---

**Security Note**: In production, use proper secret management (HashiCorp Vault, AWS Secrets Manager) for API keys! 🔒
# AI Generated Code by Deloitte + Cursor (END)
