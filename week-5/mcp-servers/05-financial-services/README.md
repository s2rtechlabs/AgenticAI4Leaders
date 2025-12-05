# AI Generated Code by Deloitte + Cursor (BEGIN)
# Financial Services MCP Server - Case Study Implementation

Enterprise-grade MCP server based on the Fortune 500 financial services case study using **FastMCP**.

## Overview

Real-world implementation of MCP for a global investment bank integrating enterprise tools. This example shows production patterns for trading, risk management, and customer service.

## ✨ Features

- ✅ Trading tools (market data, orders)
- ✅ Risk management tools
- ✅ Customer service tools
- ✅ Compliance logging
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
from datetime import datetime
import random

mcp = FastMCP("Financial Services MCP", version="1.0.0")

# ============================================================
# TRADING TOOLS
# ============================================================

@mcp.tool()
def get_market_data(symbol: str) -> str:
    """Get real-time market data for a stock symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    """
    # Mock market data
    price = round(random.uniform(100, 500), 2)
    change = round(random.uniform(-5, 5), 2)
    change_pct = round(change / price * 100, 2)
    
    return f"""
Market Data for {symbol.upper()}:
  Price: ${price}
  Change: ${change} ({change_pct}%)
  Volume: {random.randint(1000000, 10000000):,}
  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

@mcp.tool()
def get_portfolio_positions(account_id: str) -> str:
    """Get current portfolio positions for an account.
    
    Args:
        account_id: Trading account identifier
    """
    # Mock portfolio
    positions = [
        {"symbol": "AAPL", "shares": 100, "value": 17500},
        {"symbol": "GOOGL", "shares": 50, "value": 7000},
        {"symbol": "MSFT", "shares": 75, "value": 28000},
    ]
    
    result = f"Portfolio for {account_id}:\n"
    total = 0
    for pos in positions:
        result += f"  {pos['symbol']}: {pos['shares']} shares (${pos['value']:,})\n"
        total += pos['value']
    result += f"\nTotal Value: ${total:,}"
    return result

# ============================================================
# RISK MANAGEMENT TOOLS
# ============================================================

@mcp.tool()
def calculate_var(portfolio_id: str, confidence: float = 0.95) -> str:
    """Calculate Value at Risk (VaR) for a portfolio.
    
    Args:
        portfolio_id: Portfolio identifier
        confidence: Confidence level (default: 0.95 = 95%)
    """
    # Mock VaR calculation
    var_amount = round(random.uniform(10000, 50000), 2)
    
    return f"""
Value at Risk Analysis for {portfolio_id}:
  Confidence Level: {confidence * 100}%
  1-Day VaR: ${var_amount:,}
  10-Day VaR: ${var_amount * 3.16:,.2f}
  Method: Historical Simulation
  Date: {datetime.now().strftime('%Y-%m-%d')}
"""

@mcp.tool()
def stress_test(scenario: str, portfolio_id: str) -> str:
    """Run stress test scenario on portfolio.
    
    Args:
        scenario: Stress scenario (market_crash, rate_hike, recession)
        portfolio_id: Portfolio to test
    """
    scenarios = {
        "market_crash": {"impact": -25, "description": "2008-style market crash"},
        "rate_hike": {"impact": -8, "description": "200bp rate increase"},
        "recession": {"impact": -15, "description": "Economic recession"},
    }
    
    if scenario not in scenarios:
        return f"Unknown scenario: {scenario}. Available: {list(scenarios.keys())}"
    
    s = scenarios[scenario]
    return f"""
Stress Test Results for {portfolio_id}:
  Scenario: {s['description']}
  Portfolio Impact: {s['impact']}%
  Estimated Loss: ${abs(s['impact']) * 1000:,}
  Risk Rating: {'HIGH' if abs(s['impact']) > 20 else 'MEDIUM'}
"""

# ============================================================
# CUSTOMER SERVICE TOOLS
# ============================================================

@mcp.tool()
def get_customer_account(customer_id: str) -> str:
    """Get customer account information.
    
    Args:
        customer_id: Customer identifier
    """
    # Mock customer data
    return f"""
Customer Account: {customer_id}
  Name: John Smith
  Account Type: Premium
  Total Assets: $1,250,000
  Risk Profile: Moderate
  Advisor: Sarah Johnson
  Last Contact: 2024-01-15
"""

@mcp.tool()
def get_transaction_history(customer_id: str, days: int = 30) -> str:
    """Get recent transaction history.
    
    Args:
        customer_id: Customer identifier
        days: Number of days to look back (default: 30)
    """
    transactions = [
        {"date": "2024-01-15", "type": "BUY", "symbol": "AAPL", "amount": 5000},
        {"date": "2024-01-10", "type": "SELL", "symbol": "TSLA", "amount": 3500},
        {"date": "2024-01-05", "type": "DIVIDEND", "symbol": "MSFT", "amount": 150},
    ]
    
    result = f"Transaction History for {customer_id} (last {days} days):\n"
    for t in transactions:
        result += f"  {t['date']} | {t['type']:8} | {t['symbol']:5} | ${t['amount']:,}\n"
    return result

# ============================================================
# COMPLIANCE TOOLS
# ============================================================

@mcp.tool()
def compliance_check(action: str, amount: float, customer_id: str) -> str:
    """Run compliance check on a proposed action.
    
    Args:
        action: Proposed action (trade, withdrawal, transfer)
        amount: Amount in USD
        customer_id: Customer identifier
    """
    # Mock compliance check
    checks = [
        ("KYC Verification", "PASS"),
        ("AML Screening", "PASS"),
        ("Sanctions Check", "PASS"),
        ("Risk Limit Check", "PASS" if amount < 100000 else "REVIEW"),
    ]
    
    result = f"Compliance Check for {action.upper()} ${amount:,}:\n"
    all_pass = True
    for check, status in checks:
        result += f"  {check}: {status}\n"
        if status != "PASS":
            all_pass = False
    
    result += f"\nOverall: {'APPROVED' if all_pass else 'REQUIRES REVIEW'}"
    return result

if __name__ == "__main__":
    print("=" * 50)
    print("Financial Services MCP Server")
    print("=" * 50)
    print("Tools available:")
    print("  Trading: get_market_data, get_portfolio_positions")
    print("  Risk: calculate_var, stress_test")
    print("  Customer: get_customer_account, get_transaction_history")
    print("  Compliance: compliance_check")
    print("=" * 50)
    mcp.run()
```

## 📊 Implementation Results (Case Study)

| Metric | Before MCP | After MCP | Improvement |
|--------|-----------|-----------|-------------|
| Integration Time | 4-6 weeks | 3-5 days | **70% faster** |
| Maintenance Cost | $500K/year | $300K/year | **40% reduction** |
| Security Incidents | 12/year | 0/year | **100% reduction** |
| Developer Satisfaction | 6/10 | 9/10 | **50% increase** |

## 🔒 Compliance Features

For production deployment, add:
- SOC 2 compliant audit logging
- GDPR data protection
- Immutable log storage
- Anomaly detection

## 🧪 Testing

```bash
# Run server
python server.py

# Test with MCP Inspector
fastmcp dev server.py
```

## 🎓 Learning Objectives

- Enterprise MCP architecture
- Financial services domain tools
- Compliance and audit patterns
- Risk management integration
- Production deployment considerations

## 📚 Resources

- [Case Study Details](../../case-study/case-study.md)
- [Security Checklist](../../resources/security-checklist.md)
- [Deployment Guide](../../resources/deployment-guide.md)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)

---

**This is the culmination of all MCP concepts!** Study this to understand enterprise-grade implementation. 🏦
# AI Generated Code by Deloitte + Cursor (END)
