"""
Calculator Agent - A2A Protocol Exercise

This agent demonstrates basic A2A concepts:
- Agent Card publication
- Message handling
- Skill-based routing
"""

import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from agent import CalculatorAgent

# Create FastAPI app
app = FastAPI(title="Calculator Agent")

# Initialize agent
agent = CalculatorAgent()


@app.get("/.well-known/agent.json")
async def get_agent_card():
    """
    A2A Agent Discovery Endpoint
    
    This is the standard endpoint where A2A clients look for the Agent Card.
    The Agent Card describes who this agent is and what it can do.
    """
    return agent.get_agent_card()


@app.post("/a2a")
async def handle_a2a_message(request: Request):
    """
    A2A Message Handling Endpoint
    
    This endpoint receives A2A messages in JSON-RPC format.
    """
    try:
        body = await request.json()
        
        # Validate JSON-RPC format
        if body.get("jsonrpc") != "2.0":
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "error": {"code": -32600, "message": "Invalid Request"},
                    "id": body.get("id")
                }
            )
        
        method = body.get("method")
        params = body.get("params", {})
        request_id = body.get("id")
        
        # Handle different A2A methods
        if method == "message/send":
            result = await agent.handle_message(params)
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id
            })
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                    "id": request_id
                }
            )
            
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": "Parse error"},
                "id": None
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
                "id": body.get("id") if 'body' in locals() else None
            }
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "agent": agent.name}


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  {agent.name} starting...")
    print(f"  URL: http://localhost:10001")
    print(f"  Agent Card: http://localhost:10001/.well-known/agent.json")
    print(f"{'='*60}\n")
    
    uvicorn.run(app, host="0.0.0.0", port=10001, log_level="info")

