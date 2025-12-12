"""
Simple A2A Client

This client demonstrates the basic A2A interaction pattern:
1. Discover agents via their Agent Cards
2. Send messages using the A2A protocol
3. Handle responses
"""

import asyncio
import httpx
from uuid import uuid4
from typing import Any


class A2AClient:
    """
    A simple A2A client that can discover and communicate with agents.
    """
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.agent_card: dict[str, Any] | None = None
    
    async def discover(self) -> dict[str, Any]:
        """
        Discover the agent by fetching its Agent Card.
        
        The Agent Card is always available at /.well-known/agent.json
        This is the standard A2A discovery mechanism.
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/.well-known/agent.json",
                timeout=10.0
            )
            response.raise_for_status()
            self.agent_card = response.json()
            return self.agent_card
    
    async def send_message(self, text: str) -> dict[str, Any]:
        """
        Send a message to the agent using the A2A protocol.
        
        Messages are sent as JSON-RPC requests with method "message/send".
        """
        request_payload = {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": text
                        }
                    ],
                    "messageId": str(uuid4())
                }
            }
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/a2a",
                json=request_payload,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
    
    def get_skills(self) -> list[dict[str, Any]]:
        """Get the list of skills from the agent card."""
        if not self.agent_card:
            return []
        return self.agent_card.get("skills", [])
    
    def get_agent_name(self) -> str:
        """Get the agent name from the agent card."""
        if not self.agent_card:
            return "Unknown"
        return self.agent_card.get("name", "Unknown")


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_agent_info(agent_card: dict[str, Any]):
    """Print agent information in a nice format."""
    print(f"\n  📋 Agent Card:")
    print(f"     Name: {agent_card.get('name')}")
    print(f"     Version: {agent_card.get('version')}")
    print(f"     Description: {agent_card.get('description')}")
    print(f"     URL: {agent_card.get('url')}")
    
    skills = agent_card.get("skills", [])
    if skills:
        print(f"\n  🔧 Skills ({len(skills)}):")
        for skill in skills:
            print(f"     • {skill.get('name')}: {skill.get('description')}")


def print_response(response: dict[str, Any]):
    """Print the response in a nice format."""
    result = response.get("result", {})
    message = result.get("message", {})
    parts = message.get("parts", [])
    
    for part in parts:
        if part.get("kind") == "text":
            print(f"  💬 Response: {part.get('text')}")


async def demo_calculator():
    """Demonstrate interaction with the Calculator Agent."""
    print_header("Calculator Agent Demo")
    
    client = A2AClient("http://localhost:10001")
    
    try:
        # Step 1: Discover the agent
        print("\n  🔍 Discovering agent...")
        agent_card = await client.discover()
        print_agent_info(agent_card)
        
        # Step 2: Send test messages
        test_messages = [
            "Add 15 and 27",
            "Multiply 6 by 8",
            "What is 100 divided by 4?",
            "Subtract 33 from 100"
        ]
        
        print("\n  📤 Sending test messages:")
        for msg in test_messages:
            print(f"\n  → Request: {msg}")
            response = await client.send_message(msg)
            print_response(response)
            
    except httpx.ConnectError:
        print("  ❌ Error: Could not connect to Calculator Agent")
        print("     Make sure the agent is running on port 10001")


async def demo_greeter():
    """Demonstrate interaction with the Greeter Agent."""
    print_header("Greeter Agent Demo")
    
    client = A2AClient("http://localhost:10002")
    
    try:
        # Step 1: Discover the agent
        print("\n  🔍 Discovering agent...")
        agent_card = await client.discover()
        print_agent_info(agent_card)
        
        # Step 2: Send test messages
        test_messages = [
            "Say hello in Spanish",
            "Greet me in Japanese",
            "Hello in French",
            "What languages do you support?"
        ]
        
        print("\n  📤 Sending test messages:")
        for msg in test_messages:
            print(f"\n  → Request: {msg}")
            response = await client.send_message(msg)
            print_response(response)
            
    except httpx.ConnectError:
        print("  ❌ Error: Could not connect to Greeter Agent")
        print("     Make sure the agent is running on port 10002")


async def demo_weather():
    """Demonstrate interaction with the Weather Agent."""
    print_header("Weather Agent Demo")
    
    client = A2AClient("http://localhost:10003")
    
    try:
        # Step 1: Discover the agent
        print("\n  🔍 Discovering agent...")
        agent_card = await client.discover()
        print_agent_info(agent_card)
        
        # Step 2: Send test messages
        test_messages = [
            "What's the weather in Tokyo?",
            "Weather in London",
            "Compare weather in Dubai and Singapore",
            "List available cities"
        ]
        
        print("\n  📤 Sending test messages:")
        for msg in test_messages:
            print(f"\n  → Request: {msg}")
            response = await client.send_message(msg)
            print_response(response)
            
    except httpx.ConnectError:
        print("  ❌ Error: Could not connect to Weather Agent")
        print("     Make sure the agent is running on port 10003")


async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("      A2A Protocol - Simple Client Demo")
    print("="*60)
    print("\nThis demo will connect to each agent, discover its")
    print("capabilities, and send test messages.\n")
    print("Make sure all agents are running:")
    print("  • Calculator Agent: http://localhost:10001")
    print("  • Greeter Agent:    http://localhost:10002")
    print("  • Weather Agent:    http://localhost:10003")
    
    input("\nPress Enter to start the demo...")
    
    # Run demos for each agent
    await demo_calculator()
    await demo_greeter()
    await demo_weather()
    
    print_header("Demo Complete!")
    print("\n  ✅ You've successfully interacted with A2A agents!")
    print("  📚 Check out orchestrator.py for multi-agent coordination\n")


if __name__ == "__main__":
    asyncio.run(main())

