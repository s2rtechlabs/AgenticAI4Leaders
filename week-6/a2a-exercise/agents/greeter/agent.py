"""
Greeter Agent Implementation

This agent provides multilingual greetings.
"""

import random
from typing import Any
from uuid import uuid4


class GreeterAgent:
    """
    A greeter agent that provides greetings in multiple languages.
    
    Demonstrates:
    - Skill-based routing
    - Multi-language support
    - Randomized responses
    """
    
    def __init__(self):
        self.name = "Greeter Agent"
        self.version = "1.0.0"
        self.description = "Provides friendly greetings in multiple languages"
        
        # Greeting templates for different languages
        self.greetings = {
            "english": [
                "Hello! Welcome!",
                "Hi there! Great to meet you!",
                "Greetings! How can I help you today?",
                "Hey! Nice to see you!"
            ],
            "spanish": [
                "¡Hola! ¡Bienvenido!",
                "¡Buenos días! ¿Cómo estás?",
                "¡Saludos! ¿En qué puedo ayudarte?",
                "¡Hola amigo! ¡Qué gusto verte!"
            ],
            "french": [
                "Bonjour! Bienvenue!",
                "Salut! Comment allez-vous?",
                "Bienvenue! Je suis ravi de vous voir!",
                "Coucou! Ça va?"
            ],
            "german": [
                "Hallo! Willkommen!",
                "Guten Tag! Wie geht es Ihnen?",
                "Grüß Gott! Schön Sie zu sehen!",
                "Hi! Freut mich!"
            ],
            "japanese": [
                "こんにちは！ようこそ！",
                "はじめまして！よろしくお願いします！",
                "いらっしゃいませ！",
                "やあ！元気？"
            ],
            "hindi": [
                "नमस्ते! स्वागत है!",
                "आपका स्वागत है!",
                "प्रणाम! कैसे हैं आप?",
                "नमस्कार! मिलकर खुशी हुई!"
            ],
            "mandarin": [
                "你好！欢迎！",
                "您好！很高兴见到你！",
                "嗨！今天过得怎么样？",
                "欢迎光临！"
            ]
        }
        
        # Farewell templates
        self.farewells = {
            "english": ["Goodbye!", "See you later!", "Take care!", "Bye bye!"],
            "spanish": ["¡Adiós!", "¡Hasta luego!", "¡Cuídate!", "¡Nos vemos!"],
            "french": ["Au revoir!", "À bientôt!", "Salut!", "Bonne journée!"],
            "german": ["Auf Wiedersehen!", "Tschüss!", "Bis bald!", "Mach's gut!"],
            "japanese": ["さようなら！", "またね！", "お元気で！", "バイバイ！"],
            "hindi": ["अलविदा!", "फिर मिलेंगे!", "ध्यान रखना!", "नमस्ते!"],
            "mandarin": ["再见！", "回头见！", "保重！", "拜拜！"]
        }
    
    def get_agent_card(self) -> dict[str, Any]:
        """Returns the Agent Card for this agent."""
        return {
            "name": self.name,
            "description": self.description,
            "url": "http://localhost:10002/",
            "version": self.version,
            "defaultInputModes": ["text"],
            "defaultOutputModes": ["text"],
            "capabilities": {
                "streaming": False,
                "pushNotifications": False
            },
            "skills": [
                {
                    "id": "greet",
                    "name": "Greeting",
                    "description": "Provides a friendly greeting in the specified language",
                    "tags": ["greeting", "hello", "welcome", "multilingual"],
                    "examples": [
                        "Say hello in Spanish",
                        "Greet me in French",
                        "Hello in Japanese",
                        "Give me a German greeting"
                    ]
                },
                {
                    "id": "farewell",
                    "name": "Farewell",
                    "description": "Says goodbye in the specified language",
                    "tags": ["goodbye", "farewell", "bye", "multilingual"],
                    "examples": [
                        "Say goodbye in French",
                        "Farewell in German",
                        "Bye in Mandarin"
                    ]
                },
                {
                    "id": "list_languages",
                    "name": "List Languages",
                    "description": "Lists all supported languages",
                    "tags": ["languages", "list", "help"],
                    "examples": [
                        "What languages do you support?",
                        "List available languages"
                    ]
                }
            ]
        }
    
    async def handle_message(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle an incoming A2A message."""
        message = params.get("message", {})
        parts = message.get("parts", [])
        
        text = ""
        for part in parts:
            if part.get("kind") == "text":
                text += part.get("text", "")
        
        result = self._process_request(text)
        
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
    
    def _detect_language(self, text: str) -> str:
        """Detect which language the user wants."""
        text_lower = text.lower()
        
        language_keywords = {
            "english": ["english", "en"],
            "spanish": ["spanish", "español", "espanol"],
            "french": ["french", "français", "francais"],
            "german": ["german", "deutsch"],
            "japanese": ["japanese", "日本語", "nihongo"],
            "hindi": ["hindi", "हिंदी"],
            "mandarin": ["mandarin", "chinese", "中文"]
        }
        
        for lang, keywords in language_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return lang
        
        return "english"  # Default to English
    
    def _process_request(self, text: str) -> str:
        """Process the greeting request."""
        text_lower = text.lower()
        
        # Check for list languages request
        if any(word in text_lower for word in ["list", "languages", "supported", "available"]):
            languages = list(self.greetings.keys())
            return f"I can greet you in these languages: {', '.join(lang.title() for lang in languages)}"
        
        # Detect language
        language = self._detect_language(text)
        
        # Check for farewell
        if any(word in text_lower for word in ["goodbye", "bye", "farewell", "adios", "ciao"]):
            farewells = self.farewells.get(language, self.farewells["english"])
            return random.choice(farewells)
        
        # Default to greeting
        greetings = self.greetings.get(language, self.greetings["english"])
        greeting = random.choice(greetings)
        
        return f"{greeting} (in {language.title()})"

