# A2A Protocol Exercise: Multi-Agent Communication

## Overview

In this hands-on exercise, you will build a simple multi-agent system using the **Agent-to-Agent (A2A) Protocol**. You'll create three specialized agents and a client that discovers and communicates with them.

**Duration:** 1-2 hours  
**Level:** Beginner to Intermediate

---

## Learning Objectives

By completing this exercise, you will:

1. Understand how to create an **A2A-compliant agent** with an Agent Card
2. Learn how **agent discovery** works using the well-known endpoint
3. Implement **agent-to-agent communication** using the A2A protocol
4. Build a simple **orchestrator/client** that coordinates multiple agents
5. Experience the **delegation pattern** in multi-agent systems

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        A2A Client                           │
│                    (Orchestrator)                           │
└─────────────────────────┬───────────────────────────────────┘
                          │ A2A Protocol
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │Calculator│   │ Greeter  │   │ Weather  │
    │  Agent   │   │  Agent   │   │  Agent   │
    │:10001    │   │:10002    │   │:10003    │
    └──────────┘   └──────────┘   └──────────┘
```

### Agents Overview

| Agent | Port | Capabilities |
|-------|------|--------------|
| **Calculator Agent** | 10001 | add, subtract, multiply, divide |
| **Greeter Agent** | 10002 | greeting in multiple languages |
| **Weather Agent** | 10003 | get weather for cities (mock data) |

---

## Prerequisites

1. **Python 3.12+** installed
2. **uv** package manager (recommended) or pip
3. Basic understanding of async Python
4. Familiarity with REST APIs

### Install Dependencies

```bash
cd week-6/a2a-exercise
pip install -r requirements.txt
```

Or using uv:
```bash
uv pip install -r requirements.txt
```

---

## Exercise Structure

```
a2a-exercise/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── agents/
│   ├── calculator/           # Calculator Agent
│   │   ├── __main__.py       # Entry point
│   │   └── agent.py          # Agent logic
│   ├── greeter/              # Greeter Agent
│   │   ├── __main__.py       # Entry point
│   │   └── agent.py          # Agent logic
│   └── weather/              # Weather Agent
│       ├── __main__.py       # Entry point
│       └── agent.py          # Agent logic
└── client/
    ├── simple_client.py      # Basic A2A client
    └── orchestrator.py       # Multi-agent orchestrator
```

---

## Part 1: Understanding Agent Cards (10 minutes)

Every A2A agent must publish an **Agent Card** - a JSON document describing who it is and what it can do.

### Agent Card Structure

```json
{
  "name": "Calculator Agent",
  "description": "Performs basic arithmetic operations",
  "url": "http://localhost:10001/",
  "version": "1.0.0",
  "defaultInputModes": ["text"],
  "defaultOutputModes": ["text"],
  "capabilities": {
    "streaming": false
  },
  "skills": [
    {
      "id": "add",
      "name": "Addition",
      "description": "Adds two numbers together",
      "tags": ["math", "arithmetic"],
      "examples": ["Add 5 and 3", "What is 10 + 20?"]
    }
  ]
}
```

### Key Components:

| Field | Purpose |
|-------|---------|
| `name` | Human-readable agent name |
| `description` | What the agent does |
| `url` | Base URL for A2A communication |
| `version` | Agent version (semver) |
| `capabilities` | What features the agent supports (streaming, etc.) |
| `skills` | List of capabilities/actions the agent can perform |

---

## Part 2: Run the Calculator Agent (15 minutes)

### Step 1: Start the Calculator Agent

Open a terminal and run:

```bash
cd week-6/a2a-exercise/agents/calculator
python -m __main__
```

You should see:
```
Calculator Agent starting on http://localhost:10001
Agent Card available at: http://localhost:10001/.well-known/agent.json
```

### Step 2: Fetch the Agent Card

In a new terminal, test the agent discovery:

```bash
curl http://localhost:10001/.well-known/agent.json | python -m json.tool
```

This returns the Agent Card - the first step in A2A communication!

### Step 3: Send a Message to the Agent

Test the agent with a simple calculation:

```bash
curl -X POST http://localhost:10001/a2a \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Add 5 and 3"}],
        "messageId": "msg-001"
      }
    }
  }'
```

---

## Part 3: Run All Agents (10 minutes)

Start all three agents in separate terminals:

**Terminal 1 - Calculator Agent:**
```bash
cd week-6/a2a-exercise/agents/calculator
python -m __main__
```

**Terminal 2 - Greeter Agent:**
```bash
cd week-6/a2a-exercise/agents/greeter
python -m __main__
```

**Terminal 3 - Weather Agent:**
```bash
cd week-6/a2a-exercise/agents/weather
python -m __main__
```

### Verify All Agents Are Running

```bash
# Check Calculator Agent
curl -s http://localhost:10001/.well-known/agent.json | jq '.name'

# Check Greeter Agent
curl -s http://localhost:10002/.well-known/agent.json | jq '.name'

# Check Weather Agent
curl -s http://localhost:10003/.well-known/agent.json | jq '.name'
```

---

## Part 4: Build the A2A Client (20 minutes)

Now let's build a client that discovers and communicates with agents.

### Step 1: Run the Simple Client

```bash
cd week-6/a2a-exercise/client
python simple_client.py
```

This client will:
1. Discover each agent by fetching their Agent Cards
2. Display their capabilities
3. Send a test message to each agent

### Step 2: Understand the Client Code

Open `client/simple_client.py` and study how it:

1. **Discovers agents** via `/.well-known/agent.json`
2. **Parses Agent Cards** to understand capabilities
3. **Sends messages** using the A2A message format
4. **Handles responses** from agents

---

## Part 5: Build the Orchestrator (25 minutes)

The orchestrator is an agent that coordinates work across multiple agents.

### Step 1: Run the Orchestrator

```bash
cd week-6/a2a-exercise/client
python orchestrator.py
```

### Step 2: Test Multi-Agent Workflow

The orchestrator will execute a workflow that:
1. Gets a greeting from the Greeter Agent
2. Gets weather for a city from the Weather Agent
3. Performs a calculation with the Calculator Agent
4. Aggregates all results

### Example Output:

```
=== A2A Multi-Agent Orchestration Demo ===

Step 1: Discovering agents...
  ✓ Found: Calculator Agent (http://localhost:10001)
  ✓ Found: Greeter Agent (http://localhost:10002)
  ✓ Found: Weather Agent (http://localhost:10003)

Step 2: Executing workflow...
  → Requesting greeting in Spanish...
  ← Response: "¡Hola! Bienvenido!"
  
  → Requesting weather for Tokyo...
  ← Response: "Tokyo: 22°C, Partly Cloudy"
  
  → Requesting calculation: 42 * 3.14...
  ← Response: "Result: 131.88"

Step 3: Aggregating results...
═══════════════════════════════════════════
  Final Report:
  - Greeting: ¡Hola! Bienvenido!
  - Weather: Tokyo: 22°C, Partly Cloudy
  - Calculation: 131.88
═══════════════════════════════════════════

Workflow completed successfully!
```

---

## Part 6: Your Challenge (30 minutes)

Now it's your turn! Complete these challenges:

### Challenge 1: Add a New Skill

Add a "power" skill to the Calculator Agent that computes `x^y`.

1. Edit `agents/calculator/agent.py`
2. Add the new skill to the Agent Card
3. Implement the calculation logic
4. Test with the client

### Challenge 2: Create a New Agent

Create a **Joke Agent** that tells programming jokes.

1. Create `agents/joke/` directory
2. Create `__main__.py` and `agent.py`
3. Define skills: "tell_joke", "explain_joke"
4. Add it to the orchestrator

### Challenge 3: Implement Error Handling

Modify the orchestrator to:
1. Handle agent unavailability gracefully
2. Implement retry logic (3 attempts)
3. Add timeout handling (5 seconds per request)

### Challenge 4: Add Agent Authentication

Implement simple API key authentication:
1. Agents require `X-API-Key` header
2. Client must include the key in requests
3. Return 401 if key is missing/invalid

---

## Key A2A Concepts Demonstrated

### 1. Agent Discovery
```python
# Fetch Agent Card from well-known endpoint
response = await client.get("/.well-known/agent.json")
agent_card = response.json()
```

### 2. Capability Negotiation
```python
# Check if agent has the skill we need
skills = agent_card.get("skills", [])
can_calculate = any(s["id"] == "add" for s in skills)
```

### 3. Message Sending
```python
# A2A message format
message = {
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "Add 5 and 3"}],
            "messageId": str(uuid4())
        }
    }
}
```

### 4. Task Delegation
```python
# Orchestrator delegates to specialized agents
greeting = await greeter_agent.send_message("Say hello in Spanish")
weather = await weather_agent.send_message("Weather in Tokyo")
result = await calculator_agent.send_message("Multiply 42 by 3.14")
```

---

## Troubleshooting

### Agent Won't Start
- Check if port is already in use: `lsof -i :10001`
- Ensure all dependencies are installed
- Check for Python version compatibility

### Client Can't Connect
- Verify agents are running: `curl http://localhost:10001/.well-known/agent.json`
- Check firewall settings
- Ensure correct URLs in client

### Message Not Processed
- Verify JSON-RPC format is correct
- Check message ID is unique
- Look at agent console for error messages

---

## Next Steps

After completing this exercise:

1. **Explore the A2A Samples**: Check out `/a2a-samples/` for more complex examples
2. **Build with ADK**: Try Google's Agent Development Kit for production agents
3. **Add MCP Integration**: Connect your agents to tools using MCP
4. **Deploy to Cloud**: Deploy your agents to Cloud Run or Kubernetes

---

## Resources

- [A2A Protocol Specification](https://google.github.io/a2a-protocol/)
- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [A2A Samples Repository](https://github.com/google/a2a-samples)
- [MCP Documentation](https://modelcontextprotocol.io/)

---

*Happy Coding! 🚀*

