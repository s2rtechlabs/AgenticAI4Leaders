"""
Calculator Agent Implementation

This module contains the core logic for the Calculator Agent,
demonstrating A2A protocol patterns.
"""

import re
from typing import Any
from uuid import uuid4


class CalculatorAgent:
    """
    A simple calculator agent that performs basic arithmetic operations.
    
    This agent demonstrates:
    - Agent Card structure
    - Skill-based message routing
    - A2A response format
    """
    
    def __init__(self):
        self.name = "Calculator Agent"
        self.version = "1.0.0"
        self.description = "Performs basic arithmetic operations: add, subtract, multiply, divide"
    
    def get_agent_card(self) -> dict[str, Any]:
        """
        Returns the Agent Card for this agent.
        
        The Agent Card is a standardized JSON document that describes:
        - Who the agent is (name, description, version)
        - Where to reach it (url)
        - What it can do (skills)
        - How it communicates (capabilities)
        """
        return {
            "name": self.name,
            "description": self.description,
            "url": "http://localhost:10001/",
            "version": self.version,
            "defaultInputModes": ["text"],
            "defaultOutputModes": ["text"],
            "capabilities": {
                "streaming": False,
                "pushNotifications": False
            },
            "skills": [
                {
                    "id": "add",
                    "name": "Addition",
                    "description": "Adds two numbers together",
                    "tags": ["math", "arithmetic", "addition"],
                    "examples": [
                        "Add 5 and 3",
                        "What is 10 + 20?",
                        "Sum of 15 and 25"
                    ]
                },
                {
                    "id": "subtract",
                    "name": "Subtraction",
                    "description": "Subtracts second number from first",
                    "tags": ["math", "arithmetic", "subtraction"],
                    "examples": [
                        "Subtract 3 from 10",
                        "What is 20 - 5?",
                        "15 minus 8"
                    ]
                },
                {
                    "id": "multiply",
                    "name": "Multiplication",
                    "description": "Multiplies two numbers",
                    "tags": ["math", "arithmetic", "multiplication"],
                    "examples": [
                        "Multiply 6 by 7",
                        "What is 8 * 9?",
                        "12 times 11"
                    ]
                },
                {
                    "id": "divide",
                    "name": "Division",
                    "description": "Divides first number by second",
                    "tags": ["math", "arithmetic", "division"],
                    "examples": [
                        "Divide 20 by 4",
                        "What is 100 / 5?",
                        "15 divided by 3"
                    ]
                }
            ]
        }
    
    async def handle_message(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Handle an incoming A2A message.
        
        Args:
            params: The message parameters containing the user's request
            
        Returns:
            An A2A-compliant response with the calculation result
        """
        message = params.get("message", {})
        parts = message.get("parts", [])
        
        # Extract text from message parts
        text = ""
        for part in parts:
            if part.get("kind") == "text":
                text += part.get("text", "")
        
        # Parse and execute the calculation
        result = self._process_calculation(text)
        
        # Return A2A-formatted response
        return {
            "message": {
                "role": "agent",
                "parts": [
                    {
                        "kind": "text",
                        "text": result
                    }
                ],
                "messageId": str(uuid4())
            }
        }
    
    def _process_calculation(self, text: str) -> str:
        """
        Parse the text and perform the calculation.
        
        This is a simple parser that handles basic arithmetic requests.
        """
        text_lower = text.lower()
        
        # Extract numbers from text
        numbers = [float(n) for n in re.findall(r'-?\d+\.?\d*', text)]
        
        if len(numbers) < 2:
            return "I need at least two numbers to perform a calculation. Try: 'Add 5 and 3'"
        
        a, b = numbers[0], numbers[1]
        
        # Determine operation
        if any(op in text_lower for op in ['add', 'plus', 'sum', '+']):
            result = a + b
            operation = "+"
        elif any(op in text_lower for op in ['subtract', 'minus', 'difference', '-']):
            result = a - b
            operation = "-"
        elif any(op in text_lower for op in ['multiply', 'times', 'product', '*', 'x']):
            result = a * b
            operation = "×"
        elif any(op in text_lower for op in ['divide', 'divided', 'quotient', '/']):
            if b == 0:
                return "Error: Cannot divide by zero!"
            result = a / b
            operation = "÷"
        else:
            return f"I found numbers {a} and {b}, but I couldn't determine the operation. Try: add, subtract, multiply, or divide."
        
        # Format result nicely
        if result == int(result):
            result = int(result)
        else:
            result = round(result, 4)
            
        return f"{a} {operation} {b} = {result}"

