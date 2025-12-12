# Week 5: MCP (Model Context Protocol) - Building AI Tool Integrations

## 🎯 Overview

This week focuses on the **Model Context Protocol (MCP)** - a standard for integrating AI systems with enterprise tools and data sources. Using **FastMCP**, you'll learn to build MCP servers with minimal code!

## ⚡ Quick Start

```bash
# Install FastMCP
pip install fastmcp

# Run your first MCP server
cd mcp-servers/01-basic-calculator
python server.py

# Test with MCP Inspector
fastmcp dev server.py
```

**That's it!** You now have a working MCP server! 🎉

## 📚 Learning Objectives

By the end of this week, you will:

1. ✅ **Understand MCP Fundamentals** - What MCP is and why it matters
2. ✅ **Build MCP Servers** - Using FastMCP with minimal code
3. ✅ **Connect to AI Systems** - Claude Desktop, GPT-4, custom agents
4. ✅ **Implement Security** - Authentication, RBAC, audit logging
5. ✅ **Design Architectures** - Enterprise-scale MCP deployments

## 🗂️ Repository Structure

```
week-5/
├── README.md                       # This file
├── QUICK_START.md                  # Get started in 5 minutes
├── requirements.txt                # Python dependencies
├── week-5-class-notes.md          # Comprehensive lecture notes
│
├── mcp-servers/                    # Example servers
│   ├── 01-basic-calculator/        # ⭐ Start here!
│   ├── 02-database-connector/      # PostgreSQL integration
│   ├── 03-crm-integration/         # CRM/API integration
│   ├── 04-multi-tool-server/       # Auth & RBAC
│   └── 05-financial-services/      # Production case study
│
├── exercises/                      # Hands-on exercises
│   ├── exercise-1-basic-server.md  # Build weather server
│   ├── exercise-2-database-tools.md # Database integration
│   └── exercise-3-enterprise-design.md # Architecture design
│
├── case-study/                     # Financial services case study
│   └── case-study.md
│
└── resources/                      # Additional resources
    ├── deployment-guide.md
    └── security-checklist.md
```

## 🎓 Session Structure (4 hours)

### Part 1: MCP Fundamentals (45 minutes)
- What is MCP and why it matters
- FastMCP introduction
- Your first MCP server

### Part 2: Building MCP Servers (90 minutes)
- Calculator server walkthrough
- Database integration
- CRM/API integration

### Part 3: Enterprise Patterns (60 minutes)
- Authentication & authorization
- Multi-tool servers
- Production deployment

### Part 4: Case Study & Design (75 minutes)
- Financial services case study
- Design your own architecture
- Group discussion

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- (Optional) Claude Desktop for testing

### Installation

   ```bash
# Navigate to week-5
   cd week-5

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
   pip install -r requirements.txt

# Verify installation
python -c "from fastmcp import FastMCP; print('Ready!')"
   ```

### Run Your First Server

   ```bash
   cd mcp-servers/01-basic-calculator
   python server.py
   ```

### Test with MCP Inspector

   ```bash
fastmcp dev server.py
```

## 📖 Why FastMCP?

| Traditional Approach | FastMCP |
|---------------------|---------|
| 500+ lines per server | 50-100 lines |
| Manual protocol handling | Automatic |
| Complex SSE setup | Simple decorator |
| Confusing for beginners | Intuitive |

### Example: Complete MCP Server in 20 Lines

```python
from fastmcp import FastMCP

mcp = FastMCP("Calculator")

@mcp.tool()
def add(a: float, b: float) -> str:
    """Add two numbers."""
    return f"{a} + {b} = {a + b}"

@mcp.tool()
def multiply(a: float, b: float) -> str:
    """Multiply two numbers."""
    return f"{a} × {b} = {a * b}"

if __name__ == "__main__":
    mcp.run()
```

## 🎯 Learning Path

### Week 1: Foundations
- ✅ Complete `01-basic-calculator`
- ✅ Complete Exercise 1 (Weather server)
- ✅ Connect to Claude Desktop

### Week 2: Integration
- ✅ Complete `02-database-connector`
- ✅ Complete `03-crm-integration`
- ✅ Complete Exercise 2

### Week 3: Enterprise
- ✅ Study `04-multi-tool-server`
- ✅ Study `05-financial-services`
- ✅ Complete Exercise 3

### Week 4: Production
- ✅ Build your own MCP server
- ✅ Deploy to production
- ✅ Present your solution

## 💼 Case Study: Financial Services

A Fortune 500 financial services company implemented MCP:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Integration Time | 4-6 weeks | 3-5 days | **70% faster** |
| Maintenance Cost | $500K/year | $300K/year | **40% reduction** |
| Security Incidents | 12/year | 0/year | **100% reduction** |

[Read the full case study →](case-study/case-study.md)

## 🔗 Resources

### Documentation
- [FastMCP GitHub](https://github.com/jlowin/fastmcp)
- [MCP Specification](https://modelcontextprotocol.io)
- [Anthropic MCP Docs](https://docs.anthropic.com/mcp)

### Additional Materials
- [Deployment Guide](resources/deployment-guide.md)
- [Security Checklist](resources/security-checklist.md)
- [Class Notes](week-5-class-notes.md)

## 📝 Assignment

Design and implement an MCP architecture:

1. **Identify Tools** - List 5-10 tools to integrate
2. **Design Architecture** - Create system diagram
3. **Build Prototype** - Implement 2 MCP servers
4. **Present Solution** - Demo and cost-benefit analysis

## 🆘 Support

- Review the [Quick Start Guide](QUICK_START.md)
- Check the [Class Notes](week-5-class-notes.md)
- Explore the [Example Servers](mcp-servers/)
- Complete the [Exercises](exercises/)

---

## ➡️ Next Week Preview

**Week 6: Advanced Agentic AI Patterns**
- Multi-agent systems
- Agent orchestration
- Autonomous decision-making
- Production deployment strategies

---

**Ready to build?** Start with the [Quick Start Guide](QUICK_START.md)! 🚀
