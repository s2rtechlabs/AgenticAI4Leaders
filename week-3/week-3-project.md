# Week 3 Project: Build an AI Research Assistant Chatbot

Build a smart chatbot that can search the internet, understand user intent, and provide well-researched answers using Groq LLM and DuckDuckGo search.

---

## 🎯 Project Goal

Create an intelligent research assistant that:
1. Understands what users are asking (intent detection)
2. Searches the internet when needed (DuckDuckGo integration)
3. Provides clean, summarized answers (LLM-powered)
4. Has an easy-to-use interface (Gradio UI)

---

## 🛠️ Tech Stack

- **Groq API**: Fast LLM inference (Llama 3.1 70B)
- **DuckDuckGo Search**: Web search without API keys
- **Gradio**: Quick UI prototyping
- **Python**: Core programming
- **Framework** (Choose ONE):
  - **LlamaIndex**: Best for RAG and document Q&A
  - **LangChain**: Most popular, extensive ecosystem
  - **Phidata**: Modern, simple, production-ready

## 🎨 Choose Your Framework

This project can be built with any of these frameworks. Pick based on your preference:

| Framework | Best For | Difficulty | Documentation |
|-----------|----------|------------|---------------|
| **Phidata** | Beginners, quick prototyping | ⭐ Easy | [phidata.com](https://docs.phidata.com) |
| **LangChain** | Complex workflows, many integrations | ⭐⭐ Medium | [langchain.com](https://python.langchain.com) |
| **LlamaIndex** | RAG, document indexing | ⭐⭐ Medium | [llamaindex.ai](https://docs.llamaindex.ai) |

**Recommendation for this project**: Start with **Phidata** - it's the simplest and most modern!

---

## 📦 Setup

### 1. Install Dependencies

**Choose your framework and install:**

```bash
# Option 1: Phidata (Recommended for beginners)
pip install phidata groq duckduckgo-search gradio python-dotenv

# Option 2: LangChain
pip install langchain langchain-groq duckduckgo-search gradio python-dotenv

# Option 3: LlamaIndex
pip install llama-index llama-index-llms-groq duckduckgo-search gradio python-dotenv
```

### 2. Get Groq API Key

1. Visit https://console.groq.com
2. Sign up and create an API key
3. Add to `.env` file:

```bash
GROQ_API_KEY=your_api_key_here
```

### 3. Test Your Setup

```python
from groq import Groq
from duckduckgo_search import DDGS
import os
from dotenv import load_dotenv

load_dotenv()

# Test Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="llama-3.1-70b-versatile",
    messages=[{"role": "user", "content": "Hello!"}]
)
print("Groq:", response.choices[0].message.content)

# Test DuckDuckGo
with DDGS() as ddgs:
    results = list(ddgs.text("Python programming", max_results=3))
    print("DuckDuckGo:", results[0]['title'])

print("✅ Setup complete!")
```

---

## 🚀 Project Milestones

### Milestone 1: Intent Detection System

Build a system that understands what the user wants.

**Goal**: Classify user queries into intents:
- `search`: Needs web search (e.g., "What's the weather in Tokyo?")
- `chat`: General conversation (e.g., "Tell me a joke")
- `explain`: Needs explanation (e.g., "What is quantum computing?")

**Starter Code**:

```python
from groq import Groq
import os
from dotenv import load_dotenv
import json

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def detect_intent(user_message):
    """
    Detect the intent of user's message
    
    Returns:
        dict: {"intent": "search|chat|explain", "entities": [...]}
    """
    
    system_prompt = """You are an intent classifier. Analyze the user's message and respond with JSON:
    {
        "intent": "search" | "chat" | "explain",
        "needs_search": true | false,
        "entities": ["entity1", "entity2"],
        "search_query": "optimized search query if needed"
    }
    
    Intent definitions:
    - search: User wants current/factual information from the web
    - chat: Casual conversation, jokes, greetings
    - explain: Wants explanation of concepts (may or may not need search)
    
    Examples:
    "What's the weather in Paris?" -> search
    "Tell me a joke" -> chat
    "Explain quantum computing" -> explain (needs_search: true)
    "How are you?" -> chat
    """
    
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

# Test it
if __name__ == "__main__":
    test_queries = [
        "What's the latest news about AI?",
        "Tell me a joke",
        "Explain how transformers work",
        "What's the weather in London?"
    ]
    
    for query in test_queries:
        intent = detect_intent(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {intent}")
```

**Your Task**:
1. Run the code and test with different queries
2. Add more intent types (e.g., `calculate`, `translate`)
3. Extract entities (locations, dates, topics)
4. Handle edge cases

---

### Milestone 2: Web Search Integration

Add DuckDuckGo search to fetch real-time information.

**Starter Code**:

```python
from duckduckgo_search import DDGS
import json

def search_web(query, max_results=5):
    """
    Search the web using DuckDuckGo
    
    Args:
        query: Search query
        max_results: Number of results to return
        
    Returns:
        list: Search results with title, body, and URL
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            
        # Format results
        formatted_results = []
        for r in results:
            formatted_results.append({
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "url": r.get("href", "")
            })
        
        return formatted_results
    
    except Exception as e:
        print(f"Search error: {e}")
        return []

def format_search_results(results):
    """Format search results for LLM context"""
    context = "Search Results:\n\n"
    for i, result in enumerate(results, 1):
        context += f"{i}. {result['title']}\n"
        context += f"   {result['snippet']}\n"
        context += f"   Source: {result['url']}\n\n"
    return context

# Test it
if __name__ == "__main__":
    query = "latest developments in AI 2024"
    results = search_web(query, max_results=3)
    
    print(format_search_results(results))
```

**Your Task**:
1. Test search with different queries
2. Add error handling for no results
3. Filter out low-quality results
4. Add search result caching to avoid duplicate searches

---

### Milestone 3: LLM-Powered Response Generation

Use Groq LLM to generate clean, accurate answers from search results.

**Starter Code**:

```python
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_answer(user_question, search_results=None, conversation_history=None):
    """
    Generate answer using Groq LLM
    
    Args:
        user_question: User's question
        search_results: Optional search results for context
        conversation_history: Previous messages for context
        
    Returns:
        str: Generated answer
    """
    
    # Build context
    system_prompt = """You are a helpful research assistant. 
    
    Guidelines:
    - Provide accurate, well-researched answers
    - Cite sources when using search results
    - Be concise but comprehensive
    - If you don't know, say so
    - Use markdown formatting for readability
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history
    if conversation_history:
        messages.extend(conversation_history)
    
    # Add search context if available
    user_content = user_question
    if search_results:
        context = format_search_results(search_results)
        user_content = f"{context}\n\nQuestion: {user_question}\n\nProvide a comprehensive answer based on the search results above. Cite sources."
    
    messages.append({"role": "user", "content": user_content})
    
    # Generate response
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

# Test it
if __name__ == "__main__":
    # Test without search
    answer1 = generate_answer("What is Python?")
    print("Answer 1:", answer1)
    
    # Test with search
    results = search_web("Python programming language")
    answer2 = generate_answer("What is Python?", search_results=results)
    print("\nAnswer 2 (with search):", answer2)
```

**Your Task**:
1. Test with and without search results
2. Add source citations
3. Handle long search results (summarize if needed)
4. Add conversation memory

---

### Milestone 4: Complete Chatbot System

Combine everything into a working chatbot.

**Starter Code**:

```python
from groq import Groq
from duckduckgo_search import DDGS
import os
from dotenv import load_dotenv
import json

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class ResearchChatbot:
    def __init__(self):
        self.conversation_history = []
        
    def process_message(self, user_message):
        """
        Process user message and return response
        
        Steps:
        1. Detect intent
        2. Search if needed
        3. Generate response
        4. Update history
        """
        
        # Step 1: Detect intent
        intent_data = detect_intent(user_message)
        print(f"Intent: {intent_data['intent']}")
        
        # Step 2: Search if needed
        search_results = None
        if intent_data.get('needs_search', False):
            search_query = intent_data.get('search_query', user_message)
            print(f"Searching for: {search_query}")
            search_results = search_web(search_query, max_results=5)
        
        # Step 3: Generate response
        response = generate_answer(
            user_message,
            search_results=search_results,
            conversation_history=self.conversation_history[-6:]  # Last 3 exchanges
        )
        
        # Step 4: Update history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response, intent_data
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

# Test it
if __name__ == "__main__":
    bot = ResearchChatbot()
    
    print("Research Assistant ready! Type 'quit' to exit, 'clear' to reset.\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'clear':
            bot.clear_history()
            print("History cleared!\n")
            continue
        
        response, intent = bot.process_message(user_input)
        print(f"\nBot: {response}\n")
        print(f"[Intent: {intent['intent']}]\n")
```

**Your Task**:
1. Test the complete chatbot
2. Add error handling
3. Implement conversation memory limits
4. Add typing indicators or progress messages

---

### Milestone 5: Gradio UI

Create a beautiful web interface for your chatbot.

**Starter Code**:

```python
import gradio as gr
from groq import Groq
from duckduckgo_search import DDGS
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize chatbot
bot = ResearchChatbot()

def chat_interface(message, history):
    """
    Gradio chat interface function
    
    Args:
        message: Current user message
        history: Chat history (list of [user, bot] pairs)
        
    Returns:
        str: Bot response
    """
    
    # Process message
    response, intent = bot.process_message(message)
    
    # Add metadata
    metadata = f"\n\n*[Intent: {intent['intent']}]*"
    if intent.get('needs_search'):
        metadata += f" *[Searched: {intent.get('search_query', 'N/A')}]*"
    
    return response + metadata

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_interface,
    title="🔍 AI Research Assistant",
    description="Ask me anything! I can search the web and provide researched answers.",
    examples=[
        "What's the latest news about AI?",
        "Explain how transformers work in machine learning",
        "What's the weather in Tokyo?",
        "Tell me about recent SpaceX launches"
    ],
    theme=gr.themes.Soft(),
    retry_btn="🔄 Retry",
    undo_btn="↩️ Undo",
    clear_btn="🗑️ Clear"
)

if __name__ == "__main__":
    demo.launch(share=True)  # share=True creates public link
```

**Your Task**:
1. Run the Gradio interface
2. Customize the theme and styling
3. Add additional features:
   - Show search results in sidebar
   - Display intent detection
   - Add export chat button
   - Show token usage/cost
4. Deploy to Hugging Face Spaces (optional)

---

## 🎨 Enhancement Ideas

Once you have the basic chatbot working, try these enhancements:

### 1. **Multi-Source Search**
```python
# Add news search
def search_news(query, max_results=5):
    with DDGS() as ddgs:
        return list(ddgs.news(query, max_results=max_results))

# Add image search
def search_images(query, max_results=5):
    with DDGS() as ddgs:
        return list(ddgs.images(query, max_results=max_results))
```

### 2. **Smart Caching**
```python
import hashlib
import json
from datetime import datetime, timedelta

class SearchCache:
    def __init__(self, ttl_hours=24):
        self.cache = {}
        self.ttl = timedelta(hours=ttl_hours)
    
    def get(self, query):
        key = hashlib.md5(query.encode()).hexdigest()
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return data
        return None
    
    def set(self, query, results):
        key = hashlib.md5(query.encode()).hexdigest()
        self.cache[key] = (results, datetime.now())
```

### 3. **Response Streaming**
```python
def generate_answer_stream(user_question, search_results=None):
    """Stream response token by token"""
    
    # Build messages (same as before)
    messages = [...]
    
    # Stream response
    stream = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=messages,
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

### 4. **Advanced Intent Detection**
```python
def detect_advanced_intent(user_message):
    """
    Detect more specific intents:
    - compare: "Compare X and Y"
    - summarize: "Summarize this article"
    - translate: "Translate to Spanish"
    - calculate: "What's 15% of 200?"
    - code: "Write Python code for..."
    """
    # Your implementation here
    pass
```

### 5. **Conversation Analytics**
```python
class ChatAnalytics:
    def __init__(self):
        self.total_queries = 0
        self.intent_counts = {}
        self.search_counts = 0
        self.total_tokens = 0
    
    def log_query(self, intent, tokens_used, searched):
        self.total_queries += 1
        self.intent_counts[intent] = self.intent_counts.get(intent, 0) + 1
        if searched:
            self.search_counts += 1
        self.total_tokens += tokens_used
    
    def get_stats(self):
        return {
            "total_queries": self.total_queries,
            "intents": self.intent_counts,
            "searches": self.search_counts,
            "avg_tokens": self.total_tokens / max(self.total_queries, 1)
        }
```

---

## 📊 Testing Your Chatbot

Test with these scenarios:

### Factual Questions (Should Search)
- "What's the current price of Bitcoin?"
- "Who won the latest Nobel Prize in Physics?"
- "What are the top trending topics on Twitter today?"

### Explanations (May Search)
- "Explain how neural networks work"
- "What is the difference between AI and ML?"
- "How does blockchain technology work?"

### Casual Chat (No Search)
- "Tell me a joke"
- "How are you today?"
- "What's your favorite color?"

### Complex Queries
- "Compare Python and JavaScript for web development"
- "What are the pros and cons of electric vehicles?"
- "Summarize the latest developments in quantum computing"

---

## 🚢 Deployment Options

### Option 1: Hugging Face Spaces (Free)
```bash
# Create a new Space on huggingface.co
# Upload your code
# Add requirements.txt:
groq
duckduckgo-search
gradio
python-dotenv

# Add your GROQ_API_KEY as a secret
```

### Option 2: Local Sharing
```python
# In your Gradio code:
demo.launch(share=True)  # Creates temporary public URL
```

### Option 3: Streamlit Cloud
```python
# Convert to Streamlit if preferred
import streamlit as st

st.title("🔍 AI Research Assistant")
# Your chatbot code here
```

---

## 💡 Learning Outcomes

By completing this project, you'll learn:

✅ **Intent Detection**: Understanding user queries with LLMs  
✅ **Web Search Integration**: Real-time information retrieval  
✅ **LLM Orchestration**: Combining multiple AI components  
✅ **Context Management**: Handling conversation history  
✅ **UI Development**: Building user-friendly interfaces with Gradio  
✅ **API Integration**: Working with Groq and DuckDuckGo APIs  
✅ **Error Handling**: Robust application development  
✅ **Prompt Engineering**: Crafting effective prompts for different tasks  

---

## 🎯 Success Criteria

Your chatbot should:

- ✅ Correctly identify when to search vs. chat
- ✅ Retrieve relevant search results
- ✅ Generate accurate, well-cited answers
- ✅ Maintain conversation context
- ✅ Have a clean, functional UI
- ✅ Handle errors gracefully
- ✅ Be fast and responsive

---

## 📚 Resources

### Documentation
- [Groq API Docs](https://console.groq.com/docs)
- [DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search)
- [Gradio Docs](https://www.gradio.app/docs)

### Tutorials
- [Gradio Chatbot Tutorial](https://www.gradio.app/guides/creating-a-chatbot-fast)
- [Groq Quickstart](https://console.groq.com/docs/quickstart)

### Example Projects
- Look at ChatGPT, Perplexity AI for inspiration
- Study how they present search results
- Notice how they cite sources

---

## 🤝 Getting Help

Stuck? Try these:

1. **Check error messages carefully**
2. **Print intermediate results** (intent, search results, etc.)
3. **Test each component separately**
4. **Use simple test cases first**
5. **Read the API documentation**
6. **Ask for help with specific error messages**

---

## 🎉 Bonus Challenges

Ready for more? Try these:

1. **Multi-Language Support**: Detect language and respond accordingly
2. **Voice Interface**: Add speech-to-text and text-to-speech
3. **Image Understanding**: Allow users to upload images for analysis
4. **Fact Checking**: Verify claims against multiple sources
5. **Personalization**: Remember user preferences
6. **Export Features**: Save conversations as PDF/Markdown
7. **Mobile App**: Convert to mobile using Gradio's mobile support

---

**Happy Coding! Build something amazing! 🚀**

Remember: The best way to learn is by building. Start simple, test often, and iterate!
