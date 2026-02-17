# pyright: reportMissingImports=false, reportUnusedImport=false, reportGeneralTypeIssues=false
"""
[Brain] pyai Smart Research Assistant
====================================

This example demonstrates how to build a powerful AI-powered research
assistant using pyai's revolutionary simple API.

Features:
- [Book] Document analysis and Q&A (RAG)
- [Search] Deep research on any topic
- [Code] Code generation and review
- [Persona] Role-based expert personas
- [Chat] Interactive chat with memory

This showcases the TRUE POWER of pyai:
What would take 50+ lines in other frameworks takes just 2-3 lines here!

Usage:
    $env:AZURE_OPENAI_ENDPOINT = "https://openai-pyai.openai.azure.com/"
    $env:AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"
    python smart_research_assistant.py

Requirements:
    - pyai library
    - Azure OpenAI or OpenAI API key
"""

import os
import sys

# =============================================================================
# SETUP: Configure pyai (supports OpenAI, Azure API Key, or Azure AD)
# =============================================================================

# Add paths for local development (works from any directory, including PyCharm)
_examples_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_examples_dir)
sys.path.insert(0, _project_dir)  # For pyai imports
sys.path.insert(0, _examples_dir)  # For config_helper import

# Configure pyai with available credentials
from config_helper import setup_pyai
if not setup_pyai():
    print("Please configure credentials - see instructions above")
    sys.exit(1)

# Import ALL the amazing pyai functions
from pyai import (  # noqa: F401 - Imports shown for demonstration
    ask,           # Ask any question
    research,      # Deep research on topics
    summarize,     # Summarize anything
    agent,         # Create custom AI agents
    chat,          # Interactive chat sessions
    code,          # Code operations
    rag,           # Document Q&A
    analyze,       # Data analysis
)


# =============================================================================
# FEATURE 1: Instant Research
# Deep research on any topic in ONE LINE!
# =============================================================================

def demo_research():
    """
    Demonstrate pyai's research() function.
    
    In other frameworks, this would require:
    - Setting up search APIs
    - Multiple LLM calls
    - Result aggregation
    - Custom prompt engineering
    
    In pyai? ONE LINE. 
    """
    print("=" * 60)
    print("[Book] FEATURE 1: Instant Research")
    print("=" * 60)
    
    topic = "Benefits of microservices architecture"
    print(f"\n[Search] Researching: {topic}")
    print("-" * 40)
    
    # ONE LINE to do deep research! [Done]
    result = research(topic, quick=True)
    
    print(f"\n[Summary] Summary:\n{result}")


# =============================================================================
# FEATURE 2: Document Q&A with RAG
# Index documents and query them - just 2 lines!
# =============================================================================

def demo_rag():
    """
    Demonstrate pyai's RAG (Retrieval-Augmented Generation) capabilities.
    
    Traditional RAG setup requires:
    - Document loaders (20+ lines)
    - Text splitters (10+ lines)
    - Embeddings setup (10+ lines)
    - Vector store (15+ lines)
    - Retrieval chain (15+ lines)
    Total: 70+ lines
    
    pyai? TWO LINES! !
    """
    print("\n" + "=" * 60)
    print("[RAG] FEATURE 2: Document Q&A (RAG)")
    print("=" * 60)
    
    # Sample documents (in real use, these could be files or URLs)
    documents = [
        """
        pyai is a revolutionary Python library for AI development.
        It was created in 2026 to make AI accessible to everyone.
        The main philosophy is: complex AI tasks should be ONE LINE of code.
        pyai supports OpenAI, Anthropic, and Azure OpenAI providers.
        """,
        """
        Key features of pyai include:
        1. ask() - Ask any question and get intelligent answers
        2. research() - Deep research on any topic
        3. rag - Document Q&A with vector search
        4. agent() - Create custom AI agents with personas
        5. chat() - Interactive sessions with memory
        The library is designed for developers who want results, not boilerplate.
        """,
        """
        pyai vs Other Frameworks:
        - LangChain: 50+ lines for RAG vs pyai's 2 lines
        - LlamaIndex: Complex indexing vs pyai's simple index()
        - AutoGen: Multi-agent setup vs pyai's one-liner
        The goal is to be the "pandas of AI" - simple, powerful, intuitive.
        """
    ]
    
    print("\n[Inbox] Indexing 3 documents...")
    
    # LINE 1: Index documents 
    docs = rag.index(documents)
    print(f"   [OK] Indexed: {docs}")
    
    # LINE 2: Ask questions!
    questions = [
        "What is pyai's main philosophy?",
        "How does pyai compare to LangChain?",
        "What providers does pyai support?"
    ]
    
    for q in questions:
        print(f"\n? Question: {q}")
        answer = docs.ask(q)
        print(f"[Tip] Answer: {answer}")


# =============================================================================
# FEATURE 3: Expert Agent Personas
# Create specialized AI assistants instantly
# =============================================================================

def demo_agents():
    """
    Demonstrate pyai's agent() function with different personas.
    
    pyai comes with prebuilt personas:
    - coder: Expert programmer
    - researcher: Academic researcher
    - writer: Content writer
    - analyst: Data analyst
    - teacher: Patient educator
    
    Creating a specialized agent? ONE LINE! [Target]
    """
    print("\n" + "=" * 60)
    print("[Bot] FEATURE 3: Expert Agent Personas")
    print("=" * 60)
    
    # Create different expert agents - each is ONE LINE!
    
    # 1. Code Expert
    print("\n[Code] Code Expert Agent")
    print("-" * 40)
    coder = agent(persona="coder")
    response = coder("Write a Python one-liner to reverse a string")
    print(f"Response: {response}")
    
    # 2. Teacher Agent
    print("\n[Teacher] Teacher Agent")
    print("-" * 40)
    teacher = agent(persona="teacher")
    response = teacher("Explain recursion to a beginner in 2 sentences")
    print(f"Response: {response}")
    
    # 3. Custom Agent with memory
    print("\n[Persona] Custom Agent with Memory")
    print("-" * 40)
    assistant = agent(
        "You are a friendly startup advisor. Be concise and actionable.",
        name="StartupCoach"
    )
    
    # The agent remembers previous messages!
    assistant("I'm building an AI startup")
    response = assistant("What should be my first priority?")
    print(f"Response: {response}")


# =============================================================================
# FEATURE 4: AI-Powered Code Review
# Get code reviewed instantly
# =============================================================================

def demo_code_review():
    """
    Demonstrate pyai's code module for code operations.
    
    Features:
    - code.write() - Generate code from description
    - code.review() - Review code for issues
    - code.debug() - Debug error messages
    - code.explain() - Explain complex code
    - code.refactor() - Improve code quality
    """
    print("\n" + "=" * 60)
    print("[Code] FEATURE 4: AI Code Operations")
    print("=" * 60)
    
    # Generate code
    print("\n[Build] Generating code...")
    generated = code.write(
        "a function that checks if a number is prime",
        language="python"
    )
    print(f"Generated Code:\n{generated}")
    
    # Review code
    print("\n[Search] Reviewing code...")
    sample_code = """
def calc(x, y):
    result = x + y
    result = result * 2
    return result
    """
    review_result = code.review(sample_code)
    print(f"Review Score: {review_result.score}/10")
    print(f"Suggestions: {review_result.suggestions}")


# =============================================================================
# FEATURE 5: Interactive Chat Session
# Conversational AI with persistent memory
# =============================================================================

def demo_chat():
    """
    Demonstrate pyai's chat() for interactive sessions.
    
    Key feature: The chat session REMEMBERS context!
    This is what makes it powerful for multi-turn conversations.
    """
    print("\n" + "=" * 60)
    print("[Chat] FEATURE 5: Interactive Chat with Memory")
    print("=" * 60)
    
    # Create a chat session - ONE LINE!
    session = chat("You are a helpful Python tutor. Be concise.")
    
    # Have a multi-turn conversation
    conversations = [
        "What's the difference between a list and tuple?",
        "Show me an example of each",  # It remembers we're talking about lists/tuples!
        "Which one is faster?"         # Still in context!
    ]
    
    for user_message in conversations:
        print(f"\n[User] User: {user_message}")
        response = session.say(user_message)
        print(f"[Bot] Tutor: {response}")


# =============================================================================
# FEATURE 6: Sentiment Analysis
# Analyze text sentiment instantly
# =============================================================================

def demo_sentiment():
    """
    Demonstrate pyai's analyze module for text analysis.
    """
    print("\n" + "=" * 60)
    print("[Chart] FEATURE 6: Sentiment Analysis")
    print("=" * 60)
    
    texts = [
        "pyai is absolutely amazing! Best library I've ever used!",
        "The documentation could use some improvements.",
        "I love how simple it makes AI development."
    ]
    
    for text in texts:
        print(f"\n[Note] Text: {text[:50]}...")
        result = analyze.sentiment(text)
        
        # Handle both dict and object return types
        if isinstance(result, dict):
            sentiment = result.get("sentiment", "unknown")
            score = result.get("score", 0.0)
        else:
            sentiment = result.sentiment
            score = result.score
            
        emoji = "" if sentiment == "positive" else "" if sentiment == "neutral" else ""
        print(f"   Sentiment: {emoji} {sentiment} (score: {score:.2f})")


# =============================================================================
# MAIN: Run All Demos
# =============================================================================

def main():
    """Run the complete Smart Research Assistant demo."""
    print("=" * 60)
    print("   pyai SMART RESEARCH ASSISTANT")
    print("=" * 60)
    
    # Run all feature demos
    demo_research()
    demo_rag()
    demo_agents()
    demo_code_review()
    demo_chat()
    demo_sentiment()
    
    # Summary
    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
    print("""
What you've seen:
  - research()  - Deep research in ONE line
  - rag         - Document Q&A in TWO lines
  - agent()     - Expert personas in ONE line
  - code.*      - Code operations instantly
  - chat()      - Conversations with memory
  - analyze.*   - Text analysis made simple

This is the pyai difference:
  Traditional frameworks: 100+ lines of boilerplate
  pyai: 1-3 lines per feature

pyai - The Pandas of AI Development
    """)


if __name__ == "__main__":
    main()
