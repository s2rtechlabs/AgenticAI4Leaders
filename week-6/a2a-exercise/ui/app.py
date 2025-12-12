"""
A2A Exercise - Web UI with Capability Negotiation

A web interface demonstrating A2A concepts including:
- Agent Discovery
- Capability Negotiation
- Intelligent Routing
"""

import httpx
import asyncio
import re
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Any
import os

app = FastAPI(title="A2A Exercise UI")

# Agent configurations
AGENTS = {
    "calculator": {
        "url": "http://localhost:10001",
        "name": "Calculator Agent",
        "icon": "🧮",
        "color": "#4CAF50"
    },
    "greeter": {
        "url": "http://localhost:10002",
        "name": "Greeter Agent",
        "icon": "👋",
        "color": "#2196F3"
    },
    "weather": {
        "url": "http://localhost:10003",
        "name": "Weather Agent",
        "icon": "🌤️",
        "color": "#FF9800"
    }
}

# Capability keywords for matching
CAPABILITY_KEYWORDS = {
    "calculator": [
        "add", "subtract", "multiply", "divide", "calculate", "math", 
        "sum", "plus", "minus", "times", "product", "quotient",
        "arithmetic", "number", "compute", "+", "-", "*", "/"
    ],
    "greeter": [
        "hello", "hi", "greet", "greeting", "welcome", "bye", "goodbye",
        "farewell", "language", "spanish", "french", "german", "japanese",
        "hindi", "mandarin", "chinese", "say hello", "say hi"
    ],
    "weather": [
        "weather", "temperature", "forecast", "rain", "sunny", "cloudy",
        "humidity", "climate", "tokyo", "london", "paris", "new york",
        "dubai", "singapore", "city", "cities", "compare weather"
    ]
}


class MessageRequest(BaseModel):
    agent_id: str
    message: str


class SmartRouteRequest(BaseModel):
    message: str


class AgentResponse(BaseModel):
    agent_id: str
    agent_name: str
    response: str
    success: bool
    error: str | None = None


async def fetch_agent_card(url: str) -> dict[str, Any] | None:
    """Fetch an agent's card."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{url}/.well-known/agent.json",
                timeout=5.0
            )
            response.raise_for_status()
            return response.json()
    except Exception:
        return None


async def send_a2a_message(url: str, message: str) -> dict[str, Any]:
    """Send a message to an agent using A2A protocol."""
    request_payload = {
        "jsonrpc": "2.0",
        "id": str(uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": message}],
                "messageId": str(uuid4())
            }
        }
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{url}/a2a",
            json=request_payload,
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()


def calculate_capability_match(message: str, agent_id: str) -> dict[str, Any]:
    """
    Calculate how well a message matches an agent's capabilities.
    
    This simulates capability negotiation by:
    1. Analyzing the user's intent
    2. Matching against agent skills
    3. Returning a confidence score
    """
    message_lower = message.lower()
    keywords = CAPABILITY_KEYWORDS.get(agent_id, [])
    
    matched_keywords = []
    for keyword in keywords:
        if keyword in message_lower:
            matched_keywords.append(keyword)
    
    # Calculate match score (0-100)
    if not keywords:
        score = 0
    else:
        score = min(100, int((len(matched_keywords) / max(1, len(keywords) * 0.1)) * 100))
    
    return {
        "agent_id": agent_id,
        "score": score,
        "matched_keywords": matched_keywords,
        "confidence": "high" if score >= 70 else "medium" if score >= 30 else "low"
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI."""
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.get("/api/agents")
async def get_agents():
    """Get all agents with their full capability information."""
    agents_status = []
    
    for agent_id, config in AGENTS.items():
        card = await fetch_agent_card(config["url"])
        
        agents_status.append({
            "id": agent_id,
            "name": config["name"],
            "url": config["url"],
            "icon": config["icon"],
            "color": config["color"],
            "online": card is not None,
            "skills": card.get("skills", []) if card else [],
            "description": card.get("description", "") if card else "Agent unavailable",
            "version": card.get("version", "unknown") if card else "unknown",
            "capabilities": card.get("capabilities", {}) if card else {},
            "defaultInputModes": card.get("defaultInputModes", []) if card else [],
            "defaultOutputModes": card.get("defaultOutputModes", []) if card else [],
            # Full agent card for inspection
            "agentCard": card
        })
    
    return {"agents": agents_status}


@app.post("/api/negotiate")
async def negotiate_capabilities(request: SmartRouteRequest):
    """
    Perform capability negotiation to find the best agent.
    
    This demonstrates the A2A capability negotiation pattern:
    1. Analyze the user's request
    2. Match against each agent's capabilities
    3. Return ranked results with confidence scores
    """
    message = request.message
    results = []
    
    for agent_id, config in AGENTS.items():
        # Fetch agent card
        card = await fetch_agent_card(config["url"])
        
        if not card:
            continue
        
        # Calculate capability match
        match = calculate_capability_match(message, agent_id)
        
        # Get skill descriptions for context
        skills = card.get("skills", [])
        skill_names = [s.get("name", "") for s in skills]
        skill_descriptions = [s.get("description", "") for s in skills]
        
        results.append({
            "agent_id": agent_id,
            "agent_name": config["name"],
            "icon": config["icon"],
            "url": config["url"],
            "score": match["score"],
            "confidence": match["confidence"],
            "matched_keywords": match["matched_keywords"],
            "skills": skill_names,
            "skill_descriptions": skill_descriptions,
            "recommended": match["score"] >= 50
        })
    
    # Sort by score (highest first)
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Determine best agent
    best_agent = results[0] if results and results[0]["score"] > 0 else None
    
    return {
        "query": message,
        "negotiation_results": results,
        "best_match": best_agent,
        "auto_route": best_agent["agent_id"] if best_agent and best_agent["score"] >= 50 else None
    }


def detect_multi_agent_workflow(message: str) -> dict[str, Any] | None:
    """
    Detect if the message requires multiple agents.
    
    Patterns detected:
    - Weather + Math operations (add/subtract temperatures)
    - Multiple cities with comparison/calculation
    """
    message_lower = message.lower()
    
    # Check for weather + calculation pattern
    weather_keywords = ["weather", "temperature", "temp"]
    math_keywords = ["add", "subtract", "plus", "minus", "difference", "sum", "calculate", "+", "-"]
    
    has_weather = any(kw in message_lower for kw in weather_keywords)
    has_math = any(kw in message_lower for kw in math_keywords)
    
    # Find cities mentioned
    cities = ["tokyo", "moscow", "london", "paris", "new york", "dubai", "singapore", 
              "mumbai", "berlin", "sydney", "san francisco", "beijing"]
    found_cities = [city for city in cities if city in message_lower]
    
    if has_weather and has_math and len(found_cities) >= 2:
        # Determine operation
        if any(kw in message_lower for kw in ["subtract", "minus", "difference", "-"]):
            operation = "subtract"
            op_symbol = "-"
        else:
            operation = "add"
            op_symbol = "+"
            
        return {
            "type": "weather_calculation",
            "cities": found_cities[:2],
            "operation": operation,
            "op_symbol": op_symbol,
            "agents_needed": ["weather", "calculator"]
        }
    
    return None


async def execute_multi_agent_workflow(workflow: dict[str, Any], original_message: str) -> dict[str, Any]:
    """
    Execute a multi-agent workflow with capability negotiation.
    """
    steps = []
    
    city1, city2 = workflow["cities"][0], workflow["cities"][1]
    operation = workflow["operation"]
    op_symbol = workflow["op_symbol"]
    
    # Step 1: Capability Negotiation - Determine we need Weather Agent first
    step1_nego = {
        "step": 1,
        "type": "capability_negotiation",
        "description": f"Analyzing request: Need temperature data for {city1.title()} and {city2.title()}",
        "result": "Weather Agent has 'get_weather' capability - SELECTED",
        "agent_selected": "weather",
        "icon": "🔍"
    }
    steps.append(step1_nego)
    
    # Step 2: Get temperature for city1
    step2 = {
        "step": 2,
        "type": "agent_call",
        "agent": "Weather Agent",
        "icon": "🌤️",
        "action": f"Get temperature for {city1.title()}",
        "status": "running"
    }
    
    try:
        response1 = await send_a2a_message(
            AGENTS["weather"]["url"],
            f"What is the temperature in {city1}?"
        )
        result1 = response1.get("result", {}).get("message", {}).get("parts", [{}])[0].get("text", "")
        
        # Extract temperature
        temp1_match = re.search(r'Temperature:\s*(-?\d+)', result1)
        temp1 = int(temp1_match.group(1)) if temp1_match else None
        
        step2["status"] = "completed"
        step2["response"] = result1
        step2["extracted_value"] = temp1
    except Exception as e:
        step2["status"] = "failed"
        step2["error"] = str(e)
        temp1 = None
    
    steps.append(step2)
    
    # Step 3: Get temperature for city2
    step3 = {
        "step": 3,
        "type": "agent_call",
        "agent": "Weather Agent",
        "icon": "🌤️",
        "action": f"Get temperature for {city2.title()}",
        "status": "running"
    }
    
    try:
        response2 = await send_a2a_message(
            AGENTS["weather"]["url"],
            f"What is the temperature in {city2}?"
        )
        result2 = response2.get("result", {}).get("message", {}).get("parts", [{}])[0].get("text", "")
        
        temp2_match = re.search(r'Temperature:\s*(-?\d+)', result2)
        temp2 = int(temp2_match.group(1)) if temp2_match else None
        
        step3["status"] = "completed"
        step3["response"] = result2
        step3["extracted_value"] = temp2
    except Exception as e:
        step3["status"] = "failed"
        step3["error"] = str(e)
        temp2 = None
    
    steps.append(step3)
    
    # Step 4: Capability Negotiation - Weather Agent can't do math!
    step4_nego = {
        "step": 4,
        "type": "capability_negotiation",
        "description": f"Need to {operation} temperatures: {temp1}°C {op_symbol} {temp2}°C",
        "analysis": "Weather Agent skills: [get_weather, list_cities, compare_weather] - NO MATH CAPABILITY",
        "result": "Calculator Agent has 'subtract' capability - DELEGATING",
        "agent_selected": "calculator",
        "icon": "🔄",
        "is_delegation": True
    }
    steps.append(step4_nego)
    
    # Step 5: Delegate to Calculator Agent
    step5 = {
        "step": 5,
        "type": "agent_call",
        "agent": "Calculator Agent",
        "icon": "🧮",
        "action": f"Calculate: {temp1} {op_symbol} {temp2}",
        "status": "running",
        "delegation_reason": f"Weather Agent lacks arithmetic capability - delegating {operation} to Calculator Agent"
    }
    
    final_result = None
    if temp1 is not None and temp2 is not None:
        try:
            calc_message = f"{operation.title()} {temp1} and {temp2}" if operation == "add" else f"Subtract {temp2} from {temp1}"
            response3 = await send_a2a_message(
                AGENTS["calculator"]["url"],
                calc_message
            )
            result3 = response3.get("result", {}).get("message", {}).get("parts", [{}])[0].get("text", "")
            
            # Calculate result ourselves too for display
            if operation == "subtract":
                final_result = temp1 - temp2
            else:
                final_result = temp1 + temp2
            
            step5["status"] = "completed"
            step5["response"] = result3
        except Exception as e:
            step5["status"] = "failed"
            step5["error"] = str(e)
    else:
        step5["status"] = "skipped"
        step5["reason"] = "Could not extract temperatures"
    
    steps.append(step5)
    
    return {
        "is_multi_agent": True,
        "workflow_type": "weather_calculation",
        "original_query": original_message,
        "steps": steps,
        "summary": {
            "city1": {"name": city1.title(), "temperature": temp1},
            "city2": {"name": city2.title(), "temperature": temp2},
            "operation": operation,
            "op_symbol": op_symbol,
            "result": final_result
        },
        "agents_used": ["Weather Agent", "Calculator Agent"],
        "capability_negotiations": 2,
        "success": final_result is not None
    }


@app.post("/api/smart-send")
async def smart_send(request: SmartRouteRequest):
    """
    Smart send that automatically routes to the best agent.
    
    This combines capability negotiation with message sending.
    Now also supports multi-agent orchestration!
    """
    message = request.message
    
    # First, check if this requires multi-agent orchestration
    workflow = detect_multi_agent_workflow(message)
    
    if workflow:
        # Execute multi-agent workflow
        result = await execute_multi_agent_workflow(workflow, message)
        return result
    
    # Single agent routing
    negotiation = await negotiate_capabilities(request)
    
    if not negotiation["auto_route"]:
        return {
            "success": False,
            "is_multi_agent": False,
            "negotiation": negotiation,
            "error": "No suitable agent found for this request",
            "response": None
        }
    
    # Send to best agent
    agent_id = negotiation["auto_route"]
    agent_config = AGENTS[agent_id]
    
    try:
        response = await send_a2a_message(agent_config["url"], message)
        
        # Extract text from response
        result = response.get("result", {})
        msg = result.get("message", {})
        parts = msg.get("parts", [])
        
        response_text = ""
        for part in parts:
            if part.get("kind") == "text":
                response_text = part.get("text", "")
                break
        
        return {
            "success": True,
            "is_multi_agent": False,
            "negotiation": negotiation,
            "selected_agent": {
                "id": agent_id,
                "name": agent_config["name"],
                "icon": agent_config["icon"]
            },
            "response": response_text
        }
        
    except Exception as e:
        return {
            "success": False,
            "is_multi_agent": False,
            "negotiation": negotiation,
            "error": str(e),
            "response": None
        }


@app.post("/api/send")
async def send_message(request: MessageRequest) -> AgentResponse:
    """Send a message to a specific agent."""
    if request.agent_id not in AGENTS:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent_config = AGENTS[request.agent_id]
    
    try:
        response = await send_a2a_message(agent_config["url"], request.message)
        
        # Extract text from response
        result = response.get("result", {})
        message = result.get("message", {})
        parts = message.get("parts", [])
        
        response_text = ""
        for part in parts:
            if part.get("kind") == "text":
                response_text = part.get("text", "")
                break
        
        return AgentResponse(
            agent_id=request.agent_id,
            agent_name=agent_config["name"],
            response=response_text,
            success=True
        )
        
    except Exception as e:
        return AgentResponse(
            agent_id=request.agent_id,
            agent_name=agent_config["name"],
            response="",
            success=False,
            error=str(e)
        )


@app.post("/api/broadcast")
async def broadcast_message(request: MessageRequest):
    """Send a message to all agents."""
    results = []
    
    async def send_to_agent(agent_id: str, config: dict):
        try:
            response = await send_a2a_message(config["url"], request.message)
            result = response.get("result", {})
            message = result.get("message", {})
            parts = message.get("parts", [])
            
            response_text = ""
            for part in parts:
                if part.get("kind") == "text":
                    response_text = part.get("text", "")
                    break
            
            return {
                "agent_id": agent_id,
                "agent_name": config["name"],
                "icon": config["icon"],
                "response": response_text,
                "success": True
            }
        except Exception as e:
            return {
                "agent_id": agent_id,
                "agent_name": config["name"],
                "icon": config["icon"],
                "response": "",
                "success": False,
                "error": str(e)
            }
    
    # Send to all agents in parallel
    tasks = [send_to_agent(aid, cfg) for aid, cfg in AGENTS.items()]
    results = await asyncio.gather(*tasks)
    
    return {"results": results}


class MultiAgentRequest(BaseModel):
    city1: str
    city2: str


@app.post("/api/orchestrate/temperature-sum")
async def orchestrate_temperature_sum(request: MultiAgentRequest):
    """
    Multi-Agent Orchestration Demo: Temperature Sum
    
    This demonstrates agent-to-agent delegation:
    1. Weather Agent gets temperature for city1
    2. Weather Agent gets temperature for city2  
    3. Calculator Agent adds the temperatures
    4. Results are aggregated and returned
    
    This shows the A2A delegation pattern in action!
    """
    workflow_steps = []
    
    # Step 1: Get weather for city1 from Weather Agent
    step1 = {
        "step": 1,
        "agent": "Weather Agent",
        "icon": "🌤️",
        "action": f"Get temperature for {request.city1}",
        "status": "running"
    }
    workflow_steps.append(step1)
    
    try:
        response1 = await send_a2a_message(
            AGENTS["weather"]["url"], 
            f"What is the temperature in {request.city1}?"
        )
        result1 = response1.get("result", {}).get("message", {}).get("parts", [{}])[0].get("text", "")
        
        # Extract temperature from response
        import re
        temp1_match = re.search(r'Temperature:\s*(-?\d+)', result1)
        temp1 = int(temp1_match.group(1)) if temp1_match else None
        
        step1["status"] = "completed"
        step1["response"] = result1
        step1["extracted_value"] = temp1
    except Exception as e:
        step1["status"] = "failed"
        step1["error"] = str(e)
        temp1 = None
    
    # Step 2: Get weather for city2 from Weather Agent
    step2 = {
        "step": 2,
        "agent": "Weather Agent",
        "icon": "🌤️",
        "action": f"Get temperature for {request.city2}",
        "status": "running"
    }
    workflow_steps.append(step2)
    
    try:
        response2 = await send_a2a_message(
            AGENTS["weather"]["url"], 
            f"What is the temperature in {request.city2}?"
        )
        result2 = response2.get("result", {}).get("message", {}).get("parts", [{}])[0].get("text", "")
        
        # Extract temperature from response
        temp2_match = re.search(r'Temperature:\s*(-?\d+)', result2)
        temp2 = int(temp2_match.group(1)) if temp2_match else None
        
        step2["status"] = "completed"
        step2["response"] = result2
        step2["extracted_value"] = temp2
    except Exception as e:
        step2["status"] = "failed"
        step2["error"] = str(e)
        temp2 = None
    
    # Step 3: Delegate to Calculator Agent to add temperatures
    step3 = {
        "step": 3,
        "agent": "Calculator Agent",
        "icon": "🧮",
        "action": f"Add temperatures: {temp1}°C + {temp2}°C",
        "status": "running",
        "delegation_from": "Orchestrator",
        "delegation_reason": "Weather Agent cannot perform arithmetic - delegating to Calculator Agent"
    }
    workflow_steps.append(step3)
    
    if temp1 is not None and temp2 is not None:
        try:
            response3 = await send_a2a_message(
                AGENTS["calculator"]["url"], 
                f"Add {temp1} and {temp2}"
            )
            result3 = response3.get("result", {}).get("message", {}).get("parts", [{}])[0].get("text", "")
            
            step3["status"] = "completed"
            step3["response"] = result3
        except Exception as e:
            step3["status"] = "failed"
            step3["error"] = str(e)
    else:
        step3["status"] = "skipped"
        step3["reason"] = "Could not extract temperatures from previous steps"
    
    # Final aggregation
    final_result = {
        "workflow": "Temperature Sum Calculation",
        "description": "Demonstrating A2A agent delegation pattern",
        "city1": {
            "name": request.city1.title(),
            "temperature": temp1
        },
        "city2": {
            "name": request.city2.title(),
            "temperature": temp2
        },
        "sum": temp1 + temp2 if temp1 and temp2 else None,
        "steps": workflow_steps,
        "agents_involved": ["Weather Agent", "Calculator Agent"],
        "delegation_pattern": "Sequential with capability-based delegation"
    }
    
    return final_result


@app.get("/api/orchestrate/cities")
async def get_available_cities():
    """Get list of cities available for orchestration demo."""
    return {
        "cities": [
            "Tokyo", "Moscow", "London", "Paris", "New York", 
            "Dubai", "Singapore", "Mumbai", "Berlin", "Sydney",
            "San Francisco", "Beijing"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("  A2A Exercise - Web UI with Capability Negotiation")
    print("="*60)
    print("\n  Open in browser: http://localhost:8000")
    print("\n  Make sure all agents are running:")
    print("    • Calculator: http://localhost:10001")
    print("    • Greeter:    http://localhost:10002")
    print("    • Weather:    http://localhost:10003")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
