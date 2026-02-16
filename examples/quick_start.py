"""
PyAgent Quick Start Examples - Revolutionary AI in 3 Lines or Less

This file demonstrates the power of PyAgent's pandas-like simplicity.
Each example shows a complete, working solution to common AI tasks.

NOTE: This is a REFERENCE file showing API examples.
      For runnable examples, see: comprehensive_examples.py

Authentication Options:
    # Option 1: OpenAI API Key
    export OPENAI_API_KEY=sk-your-key
    
    # Option 2: Azure OpenAI with Azure AD (recommended - no key needed)
    export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
    export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
    # Uses your Azure login automatically (az login / VS Code)
"""

import os
import sys

# Add paths for local development (works from any directory, including PyCharm)
_examples_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_examples_dir)
sys.path.insert(0, _project_dir)  # For pyagent imports
sys.path.insert(0, _examples_dir)  # For config_helper import

# Configure PyAgent with available credentials (supports OpenAI, Azure API Key, or Azure AD)
from config_helper import setup_pyagent
if not setup_pyagent():
    print("Please configure credentials - see instructions above")
    sys.exit(1)

# =============================================================================
# 1. ASK - The Simplest AI Function
# =============================================================================

from pyagent import ask

# Basic question
answer = ask("What is the capital of France?")
print(answer)  # "Paris"

# Detailed answer
explanation = ask("How does photosynthesis work?", detailed=True)

# Concise answer
brief = ask("Explain blockchain", concise=True)

# Formatted output
tips = ask("Give me 5 Python tips", format="bullet")

# Creative output
story = ask("Write a haiku about coding", creative=True)

# JSON output
data = ask("Generate a sample user profile", as_json=True)


# =============================================================================
# 2. RAG - Document Q&A in 2 Lines
# =============================================================================

from pyagent import rag

# One-shot RAG
answer = rag.ask("./docs/research_paper.pdf", "What is the main finding?")

# Or create an index for multiple queries
docs = rag.index("./documents")
answer1 = docs.ask("What is the conclusion?")
answer2 = docs.ask("What methodology was used?")
answer3 = docs.ask("What are the limitations?")

# RAG from URL
answer = rag.from_url("https://example.com/article", "Summarize the key points")

# RAG from text
long_text = "..." 
answer = rag.from_text(long_text, "What is the main argument?")


# =============================================================================
# 3. RESEARCH - Deep Research in 1 Line
# =============================================================================

from pyagent import research

# Full research
result = research("quantum computing applications")
print(result.summary)
print(result.key_points)
print(result.insights)

# Quick research (just summary)
summary = research("benefits of meditation", quick=True)

# Get insights only
insights = research("future of remote work", as_insights=True)

# Focused research
result = research("climate change", focus="economic impact")


# =============================================================================
# 4. FETCH - Real-time Data in 1 Line
# =============================================================================

from pyagent import fetch

# Weather
weather = fetch.weather("Tokyo")
print(f"Tokyo: {weather.temperature}C, {weather.conditions}")
print(f"Humidity: {weather.humidity}%")

# News
news = fetch.news("artificial intelligence")
for article in news[:3]:
    print(f"- {article.title}")

# Stocks
apple = fetch.stock("AAPL")
print(f"Apple: ${apple.price} ({apple.change_percent:+.2f}%)")

# Crypto
btc = fetch.crypto("BTC")
print(f"Bitcoin: ${btc.price:,.2f}")

# Facts
facts = fetch.facts("space exploration", count=3)
for fact in facts:
    print(f" {fact}")


# =============================================================================
# 5. SUMMARIZE - Summarize Anything
# =============================================================================

from pyagent import summarize

# Text
summary = summarize("Your long text here...")

# File
summary = summarize("./report.pdf")
summary = summarize("./document.docx")

# URL
summary = summarize("https://news.site/long-article")

# Options
short = summarize(text, length="short")
bullets = summarize(text, bullet_points=True)
executive = summarize(text, style="executive")


# =============================================================================
# 6. EXTRACT - Get Structured Data from Text
# =============================================================================

from pyagent import extract

text = "John Smith is 35 years old, works as a software engineer at Google, and lives in San Francisco."

# Extract specific fields
data = extract(text, ["name", "age", "company", "location"])
# {"name": "John Smith", "age": 35, "company": "Google", "location": "San Francisco"}

# Natural language extraction
emails = extract(email_text, "all email addresses mentioned")

# Entity extraction
entities = extract_entities(article_text)
# {"people": [...], "organizations": [...], "locations": [...]}


# =============================================================================
# 7. GENERATE - Create Any Content
# =============================================================================

from pyagent import generate

# Code
code = generate("function to calculate fibonacci", type="code")

# Email
email = generate("welcome email for new SaaS users", type="email", tone="friendly")

# Article
article = generate("blog post about AI trends", type="article", length="long")

# Social media
tweet = generate("announcement for new product launch", type="social")


# =============================================================================
# 8. TRANSLATE - Instant Translation
# =============================================================================

from pyagent import translate

# Basic
spanish = translate("Hello, how are you?", to="spanish")
# "Hola, cmo ests?"

# With formality
formal_japanese = translate("Thank you for your help", to="japanese", formal=True)

# Language codes work too
french = translate("Good morning", to="fr")


# =============================================================================
# 9. ANALYZE - Data & Text Analysis
# =============================================================================

from pyagent import analyze

# Analyze data
sales_data = [100, 150, 120, 200, 180, 500, 190]
result = analyze.data(sales_data)
print(result.summary)
print(result.insights)
print(result.recommendations)

# Sentiment analysis
sentiment = analyze.sentiment("I absolutely love this product!")
# {"sentiment": "positive", "confidence": 0.95, "emotions": ["joy", "satisfaction"]}

# Text analysis
text_insights = analyze.text(customer_review, analyze_for="sentiment")


# =============================================================================
# 10. CODE - AI-Powered Coding
# =============================================================================

from pyagent import code

# Write code
python_function = code.write("REST API for user management")

# Review code
review = code.review(my_code)
print(f"Score: {review.score}/10")
print(f"Issues: {review.issues}")
print(f"Suggestions: {review.suggestions}")

# Debug errors
fix = code.debug("TypeError: cannot unpack non-iterable int object")

# Explain code
explanation = code.explain(complex_algorithm, for_beginner=True)

# Refactor
improved = code.refactor(legacy_code, goal="readability")

# Convert between languages
javascript = code.convert(python_code, from_lang="python", to_lang="javascript")


# =============================================================================
# 11. AGENT - Create Custom Agents
# =============================================================================

from pyagent import agent

# Simple agent
helper = agent("You are a helpful assistant")
response = helper("What's a good way to learn Python?")

# Prebuilt personas
coder = agent(persona="coder")
researcher = agent(persona="researcher")
writer = agent(persona="writer")
teacher = agent(persona="teacher")

# Agent with memory (multi-turn conversations)
analyst = agent("You are a data analyst", memory=True)
analyst("I have sales data for Q1")
analyst("What trends do you see?")  # Remembers previous message
analyst("How does it compare to Q2?")  # Still has context

# Named agent with specific model
expert = agent(
    "You are an expert in quantum physics",
    name="QuantumExpert",
    model="gpt-4o"
)


# =============================================================================
# 12. CHAT - Interactive Sessions
# =============================================================================

from pyagent import chat

# Create session
session = chat(persona="teacher")

# Have a conversation
session("What is machine learning?")
session("Can you give me an example?")  # Remembers context
session("What about deep learning?")    # Continues conversation

# Check history
print(session.history)

# Reset when needed
session.reset()


# =============================================================================
# BONUS: One-liner Solutions to Common Tasks
# =============================================================================

# Summarize a PDF
summary = summarize("./quarterly_report.pdf")

# Answer questions about documents  
answer = rag.ask("./manual.pdf", "How do I reset the settings?")

# Get weather
weather = fetch.weather("London")

# Research a topic
insights = research("electric vehicles market", quick=True)

# Create a coding assistant
coder = agent(persona="coder")

# Generate code
api_code = code.write("REST API for todo app with Flask")

# Debug an error
solution = code.debug("IndexError: list index out of range")

# Translate text
german = translate("Hello world", to="german")

# Analyze sentiment
mood = analyze.sentiment("This is the best day ever!")

# Extract data
info = extract("Jane Doe, CEO of TechCorp, announced...", ["name", "title", "company"])

# Ask anything
answer = ask("What's the meaning of life?", creative=True)

print("\n[OK] All examples completed! PyAgent makes AI simple.")
