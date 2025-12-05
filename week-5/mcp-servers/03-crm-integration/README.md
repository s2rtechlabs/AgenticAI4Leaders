# CRM Integration MCP Server

MCP server demonstrating external API integration using **FastMCP**.

## Overview

Learn how to integrate AI systems with CRM platforms (Salesforce-style) using MCP. This example shows API authentication, data transformation, and error handling patterns.

## ✨ Features

- ✅ OAuth-style authentication flow
- ✅ REST API integration
- ✅ Data transformation
- ✅ Error handling
- ✅ Clean FastMCP implementation

## 📦 Quick Start

### 1. Install Dependencies

```bash
pip install fastmcp httpx python-dotenv
```

### 2. Configure Environment

Create `.env` file:

```env
CRM_API_URL=https://api.example-crm.com
CRM_API_KEY=your_api_key
```

### 3. Run Server

```bash
python server.py
```

## 🔧 Example Implementation

```python
from fastmcp import FastMCP
import httpx
import os

mcp = FastMCP("CRM Integration")

# Mock CRM data for demonstration
MOCK_ACCOUNTS = {
    "ACC001": {"name": "Acme Corp", "industry": "Technology", "revenue": 1000000},
    "ACC002": {"name": "Global Inc", "industry": "Finance", "revenue": 5000000},
}

MOCK_CONTACTS = {
    "ACC001": [
        {"name": "John Doe", "email": "john@acme.com", "role": "CEO"},
        {"name": "Jane Smith", "email": "jane@acme.com", "role": "CTO"},
    ],
    "ACC002": [
        {"name": "Bob Johnson", "email": "bob@global.com", "role": "CFO"},
    ],
}

@mcp.tool()
def get_account(account_id: str) -> str:
    """Retrieve account information from the CRM.
    
    Args:
        account_id: The unique account identifier (e.g., 'ACC001')
    """
    if account_id in MOCK_ACCOUNTS:
        account = MOCK_ACCOUNTS[account_id]
        return f"Account: {account['name']}\nIndustry: {account['industry']}\nRevenue: ${account['revenue']:,}"
    return f"Error: Account {account_id} not found"

@mcp.tool()
def get_contacts(account_id: str) -> str:
    """Get all contacts for an account.
    
    Args:
        account_id: The account ID to get contacts for
    """
    if account_id in MOCK_CONTACTS:
        contacts = MOCK_CONTACTS[account_id]
        result = f"Contacts for {account_id}:\n"
        for contact in contacts:
            result += f"  - {contact['name']} ({contact['role']}): {contact['email']}\n"
        return result
    return f"Error: No contacts found for {account_id}"

@mcp.tool()
def search_accounts(query: str, industry: str = None) -> str:
    """Search for accounts by name or industry.
    
    Args:
        query: Search query for account name
        industry: Filter by industry (optional)
    """
    results = []
    for acc_id, account in MOCK_ACCOUNTS.items():
        if query.lower() in account['name'].lower():
            if industry is None or account['industry'].lower() == industry.lower():
                results.append(f"{acc_id}: {account['name']} ({account['industry']})")
    
    if results:
        return "Found accounts:\n" + "\n".join(results)
    return "No accounts found matching your query"

@mcp.tool()
def create_lead(name: str, email: str, company: str, source: str = "Website") -> str:
    """Create a new lead in the CRM.
    
    Args:
        name: Lead's full name
        email: Lead's email address
        company: Lead's company name
        source: Lead source (default: 'Website')
    """
    # In a real implementation, this would call the CRM API
    lead_id = f"LEAD{hash(email) % 10000:04d}"
    return f"Lead created successfully!\nID: {lead_id}\nName: {name}\nEmail: {email}\nCompany: {company}\nSource: {source}"

if __name__ == "__main__":
    mcp.run()
```

## 🔒 Authentication Patterns

### API Key Authentication

```python
import httpx

async def call_crm_api(endpoint: str, method: str = "GET", data: dict = None):
    headers = {
        "Authorization": f"Bearer {os.getenv('CRM_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        if method == "GET":
            response = await client.get(f"{CRM_API_URL}/{endpoint}", headers=headers)
        elif method == "POST":
            response = await client.post(f"{CRM_API_URL}/{endpoint}", headers=headers, json=data)
        
        return response.json()
```

### OAuth 2.0 Flow

```python
# For production CRM integrations, implement OAuth:
# 1. Redirect user to authorization URL
# 2. Exchange authorization code for access token
# 3. Refresh tokens when expired
# 4. Store tokens securely
```

## 🧪 Testing

```bash
# Run server
python server.py

# Test with MCP Inspector
fastmcp dev server.py
```

## 🎓 Learning Objectives

- External API integration with FastMCP
- Authentication patterns (API key, OAuth)
- Data transformation between systems
- Error handling for API calls
- Mock data for development

## 📚 Resources

- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [httpx Documentation](https://www.python-httpx.org/)
- [OAuth 2.0 Guide](https://oauth.net/2/)

---

**Tip**: Start with mock data, then integrate with the real API! 🎯