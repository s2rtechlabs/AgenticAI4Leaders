"""
A2A Orchestrator

This orchestrator demonstrates multi-agent coordination:
1. Discovers multiple agents
2. Routes tasks to appropriate agents based on capabilities
3. Aggregates results from multiple agents
4. Handles failures gracefully
"""

import asyncio
import httpx
from uuid import uuid4
from typing import Any
from dataclasses import dataclass
from enum import Enum


class AgentStatus(Enum):
    """Status of an agent connection."""
    UNKNOWN = "unknown"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


@dataclass
class AgentInfo:
    """Information about a discovered agent."""
    url: str
    name: str
    description: str
    skills: list[dict[str, Any]]
    status: AgentStatus
    

class A2AOrchestrator:
    """
    Orchestrator that coordinates multiple A2A agents.
    
    This demonstrates the A2A pattern of:
    - Agent discovery
    - Capability-based routing
    - Task delegation
    - Result aggregation
    """
    
    def __init__(self):
        self.agents: dict[str, AgentInfo] = {}
        self.agent_urls = {
            "calculator": "http://localhost:10001",
            "greeter": "http://localhost:10002",
            "weather": "http://localhost:10003"
        }
    
    async def discover_agents(self) -> dict[str, AgentInfo]:
        """
        Discover all configured agents by fetching their Agent Cards.
        
        This is the A2A discovery pattern - agents publish their capabilities
        at a well-known endpoint.
        """
        print("\n🔍 Discovering agents...")
        
        async with httpx.AsyncClient() as client:
            for agent_type, base_url in self.agent_urls.items():
                try:
                    response = await client.get(
                        f"{base_url}/.well-known/agent.json",
                        timeout=5.0
                    )
                    response.raise_for_status()
                    card = response.json()
                    
                    self.agents[agent_type] = AgentInfo(
                        url=base_url,
                        name=card.get("name", "Unknown"),
                        description=card.get("description", ""),
                        skills=card.get("skills", []),
                        status=AgentStatus.AVAILABLE
                    )
                    print(f"   ✓ Found: {card.get('name')} at {base_url}")
                    
                except Exception as e:
                    self.agents[agent_type] = AgentInfo(
                        url=base_url,
                        name=agent_type.title(),
                        description="",
                        skills=[],
                        status=AgentStatus.UNAVAILABLE
                    )
                    print(f"   ✗ {agent_type.title()} Agent unavailable: {str(e)[:50]}")
        
        return self.agents
    
    async def send_to_agent(self, agent_type: str, message: str) -> dict[str, Any] | None:
        """
        Send a message to a specific agent.
        
        Returns the response or None if the agent is unavailable.
        """
        agent = self.agents.get(agent_type)
        
        if not agent or agent.status != AgentStatus.AVAILABLE:
            return None
        
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
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{agent.url}/a2a",
                    json=request_payload,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"   ⚠ Error from {agent.name}: {str(e)[:50]}")
            return None
    
    def extract_response_text(self, response: dict[str, Any] | None) -> str:
        """Extract the text from an A2A response."""
        if not response:
            return "No response"
        
        result = response.get("result", {})
        message = result.get("message", {})
        parts = message.get("parts", [])
        
        for part in parts:
            if part.get("kind") == "text":
                return part.get("text", "")
        
        return "No text in response"
    
    async def execute_workflow(self):
        """
        Execute a multi-agent workflow.
        
        This workflow demonstrates:
        1. Parallel task execution
        2. Sequential dependencies
        3. Result aggregation
        """
        print("\n📋 Executing multi-agent workflow...")
        print("-" * 50)
        
        results = {}
        
        # Task 1: Get a greeting (Greeter Agent)
        print("\n  Step 1: Requesting greeting in Spanish...")
        response = await self.send_to_agent("greeter", "Say hello in Spanish")
        results["greeting"] = self.extract_response_text(response)
        print(f"   ← {results['greeting']}")
        
        # Task 2: Get weather for a city (Weather Agent)
        print("\n  Step 2: Requesting weather for Tokyo...")
        response = await self.send_to_agent("weather", "What's the weather in Tokyo?")
        results["weather"] = self.extract_response_text(response)
        print(f"   ← {results['weather']}")
        
        # Task 3: Perform a calculation (Calculator Agent)
        print("\n  Step 3: Performing calculation (42 × 3.14)...")
        response = await self.send_to_agent("calculator", "Multiply 42 by 3.14")
        results["calculation"] = self.extract_response_text(response)
        print(f"   ← {results['calculation']}")
        
        return results
    
    async def execute_parallel_workflow(self):
        """
        Execute tasks in parallel across multiple agents.
        
        This demonstrates the parallel delegation pattern.
        """
        print("\n📋 Executing parallel workflow...")
        print("-" * 50)
        
        # Define tasks for parallel execution
        tasks = [
            ("greeter", "Greet me in French"),
            ("weather", "Weather in London"),
            ("calculator", "Add 100 and 250"),
        ]
        
        # Execute all tasks in parallel
        print("\n  Sending requests to all agents simultaneously...")
        
        async def execute_task(agent_type: str, message: str) -> tuple[str, str]:
            response = await self.send_to_agent(agent_type, message)
            return (agent_type, self.extract_response_text(response))
        
        # Create tasks for parallel execution
        parallel_tasks = [
            execute_task(agent_type, message) 
            for agent_type, message in tasks
        ]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*parallel_tasks)
        
        # Convert to dictionary
        results_dict = {agent_type: response for agent_type, response in results}
        
        return results_dict
    
    async def capability_based_routing(self, user_request: str) -> str:
        """
        Route a user request to the appropriate agent based on capabilities.
        
        This demonstrates the capability negotiation pattern.
        """
        user_lower = user_request.lower()
        
        # Simple keyword-based routing
        if any(word in user_lower for word in ["add", "subtract", "multiply", "divide", "calculate", "math"]):
            response = await self.send_to_agent("calculator", user_request)
            return f"[Calculator] {self.extract_response_text(response)}"
        
        elif any(word in user_lower for word in ["weather", "temperature", "forecast", "rain"]):
            response = await self.send_to_agent("weather", user_request)
            return f"[Weather] {self.extract_response_text(response)}"
        
        elif any(word in user_lower for word in ["hello", "greet", "hi", "welcome", "goodbye"]):
            response = await self.send_to_agent("greeter", user_request)
            return f"[Greeter] {self.extract_response_text(response)}"
        
        else:
            return "I'm not sure which agent can handle that request."


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_results(results: dict[str, str]):
    """Print aggregated results in a nice format."""
    print("\n" + "═"*60)
    print("  📊 AGGREGATED RESULTS")
    print("═"*60)
    
    for key, value in results.items():
        # Truncate long values for display
        display_value = value if len(value) < 50 else value[:47] + "..."
        print(f"  • {key.title()}: {display_value}")
    
    print("═"*60)


async def main():
    """Run the orchestrator demo."""
    print("\n" + "="*60)
    print("      A2A Multi-Agent Orchestrator Demo")
    print("="*60)
    print("\nThis demo shows how an orchestrator coordinates")
    print("multiple agents to complete complex workflows.\n")
    print("Make sure all agents are running:")
    print("  • Calculator Agent: http://localhost:10001")
    print("  • Greeter Agent:    http://localhost:10002")
    print("  • Weather Agent:    http://localhost:10003")
    
    input("\nPress Enter to start the orchestration demo...")
    
    orchestrator = A2AOrchestrator()
    
    # Step 1: Discover all agents
    print_header("Step 1: Agent Discovery")
    await orchestrator.discover_agents()
    
    # Check if any agents are available
    available = [a for a in orchestrator.agents.values() if a.status == AgentStatus.AVAILABLE]
    if not available:
        print("\n  ❌ No agents available. Please start the agents first.")
        return
    
    # Step 2: Sequential workflow
    print_header("Step 2: Sequential Workflow")
    results = await orchestrator.execute_workflow()
    print_results(results)
    
    # Step 3: Parallel workflow
    print_header("Step 3: Parallel Workflow")
    parallel_results = await orchestrator.execute_parallel_workflow()
    print_results(parallel_results)
    
    # Step 4: Capability-based routing
    print_header("Step 4: Capability-Based Routing")
    
    test_requests = [
        "Add 50 and 75",
        "What's the weather in Paris?",
        "Say hello in German",
        "Can you help me with something random?"
    ]
    
    print("\n  Testing intelligent routing:")
    for request in test_requests:
        print(f"\n  → User: {request}")
        response = await orchestrator.capability_based_routing(request)
        print(f"  ← {response}")
    
    # Summary
    print_header("Demo Complete!")
    print("""
  ✅ You've seen A2A multi-agent orchestration in action!
  
  Key Patterns Demonstrated:
  • Agent Discovery - Finding agents via their Agent Cards
  • Sequential Workflows - Tasks that depend on each other
  • Parallel Execution - Running multiple tasks simultaneously
  • Capability Routing - Directing requests to the right agent
  • Result Aggregation - Combining results from multiple agents
  
  📚 Next Steps:
  • Try the challenges in the README
  • Add your own agents to the system
  • Implement error handling and retries
""")


if __name__ == "__main__":
    asyncio.run(main())

