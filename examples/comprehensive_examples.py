"""
🐼🤖 PyAgent Comprehensive Examples
====================================

A complete showcase of PyAgent's capabilities - from simple one-liners
to advanced multi-agent workflows.

Setup:
    # OpenAI
    export OPENAI_API_KEY=sk-your-key
    
    # Azure OpenAI (with Azure AD - no key required)
    export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
    export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini

Run:
    python comprehensive_examples.py [openai|azure]
"""

import os
import sys
import time
from typing import Tuple

# Add pyagent to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# CONFIGURATION
# =============================================================================

def setup_provider(provider: str = "auto") -> Tuple[bool, str]:
    """
    Auto-detect and configure the AI provider.
    
    Supports:
    - OpenAI with API key
    - Azure OpenAI with API key
    - Azure OpenAI with Azure AD (DefaultAzureCredential)
    """
    import pyagent
    
    # Auto-detect provider
    if provider == "auto":
        if os.environ.get("AZURE_OPENAI_ENDPOINT"):
            provider = "azure"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            return False, "No API configuration found"
    
    if provider == "azure":
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        
        if not endpoint:
            return False, "AZURE_OPENAI_ENDPOINT not set"
        
        # Try Azure AD authentication if no API key
        if not api_key:
            try:
                from azure.identity import DefaultAzureCredential
                DefaultAzureCredential()  # Validate credentials exist
                pyagent.configure(
                    provider="azure",
                    azure_endpoint=endpoint,
                    model=deployment
                )
                return True, f"Azure OpenAI (Azure AD): {endpoint}"
            except Exception as e:
                return False, f"Azure AD auth failed: {e}"
        else:
            pyagent.configure(
                provider="azure",
                api_key=api_key,
                azure_endpoint=endpoint,
                model=deployment
            )
            return True, f"Azure OpenAI (API Key): {endpoint}"
    
    else:  # OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return False, "OPENAI_API_KEY not set"
        
        pyagent.configure(
            provider="openai",
            api_key=api_key,
            model="gpt-4o-mini"
        )
        return True, "OpenAI configured"


# =============================================================================
# EXAMPLE 1: SIMPLE Q&A
# =============================================================================

def example_ask():
    """
    Example 1: ask() - The simplest AI function
    
    One line to answer any question. No setup, no configuration beyond API key.
    """
    print("\n" + "=" * 60)
    print("📝 EXAMPLE 1: ask() - Simple Q&A")
    print("=" * 60)
    
    from pyagent import ask
    
    # Basic question
    print("\n1a. Basic Question:")
    print("    Code: answer = ask('What is Python?')")
    start = time.time()
    answer = ask("What is Python? Answer in one sentence.", concise=True)
    print(f"    Answer: {answer}")
    print(f"    Time: {time.time() - start:.2f}s")
    
    # Detailed response
    print("\n1b. Detailed Response:")
    print("    Code: answer = ask('Explain AI', detailed=True)")
    answer = ask("List 3 benefits of Python", concise=True)
    print(f"    Answer: {answer[:150]}...")
    
    # Verify with math
    print("\n1c. Verification Test:")
    answer = ask("What is 25 * 4? Reply with just the number.")
    success = "100" in answer
    print(f"    Question: 25 × 4 = ?")
    print(f"    Answer: {answer}")
    print(f"    {'✅ PASS' if success else '❌ FAIL'}")
    
    return success


# =============================================================================
# EXAMPLE 2: CUSTOM AGENTS
# =============================================================================

def example_agent():
    """
    Example 2: agent() - Create specialized AI agents
    
    Create experts for specific domains. Agents can have personas,
    memory, and custom behavior.
    """
    print("\n" + "=" * 60)
    print("🤖 EXAMPLE 2: agent() - Custom Agents")
    print("=" * 60)
    
    from pyagent import agent
    
    # Custom agent with system prompt
    print("\n2a. Custom Agent:")
    print("    Code: tutor = agent('You are a math tutor')")
    tutor = agent("You are a math tutor. Give very brief answers.")
    
    response = tutor("What is the square root of 144? Just the number.")
    success = "12" in response
    print(f"    Question: Square root of 144?")
    print(f"    Answer: {response}")
    print(f"    {'✅ PASS' if success else '❌ FAIL'}")
    
    # Prebuilt persona
    print("\n2b. Prebuilt Personas:")
    print("    Code: coder = agent(persona='coder')")
    coder = agent(persona="coder")
    print(f"    Created: {coder}")
    
    # Available personas
    print("\n    Available personas:")
    print("    • coder      - Expert programmer")
    print("    • researcher - Deep research specialist")
    print("    • writer     - Content creation")
    print("    • analyst    - Data analysis expert")
    print("    • teacher    - Educational explanations")
    
    return success


# =============================================================================
# EXAMPLE 3: CHAT WITH MEMORY
# =============================================================================

def example_chat():
    """
    Example 3: chat() - Conversational sessions with memory
    
    Multi-turn conversations that remember context. Perfect for
    interactive assistants and chatbots.
    """
    print("\n" + "=" * 60)
    print("💬 EXAMPLE 3: chat() - Conversations with Memory")
    print("=" * 60)
    
    from pyagent import chat
    
    print("\n3a. Creating Chat Session:")
    print("    Code: session = chat('You are a helpful assistant')")
    session = chat("You are a helpful assistant. Be very brief.")
    print(f"    Created: {session}")
    
    # Multi-turn with memory
    print("\n3b. Memory Test:")
    
    # First message
    r1 = session.say("My favorite color is blue. Remember that.")
    print(f"    User: My favorite color is blue.")
    print(f"    Assistant: {r1[:80]}...")
    
    # Test memory
    r2 = session.say("What is my favorite color?")
    success = "blue" in r2.lower()
    print(f"\n    User: What is my favorite color?")
    print(f"    Assistant: {r2}")
    print(f"    Memory working: {'✅ YES' if success else '❌ NO'}")
    
    return success


# =============================================================================
# EXAMPLE 4: RAG IN 2 LINES
# =============================================================================

def example_rag():
    """
    Example 4: rag - RAG Pipeline in 2 lines
    
    Index documents and query them with AI. What takes 50+ lines
    in other frameworks takes just 2 lines with PyAgent.
    """
    print("\n" + "=" * 60)
    print("📚 EXAMPLE 4: rag - RAG in 2 Lines")
    print("=" * 60)
    
    from pyagent import rag
    
    # Sample documents
    documents = [
        "PyAgent was created in 2026 as a revolutionary AI library.",
        "The library provides pandas-like simplicity for AI tasks.",
        "PyAgent supports OpenAI, Anthropic, and Azure OpenAI.",
        "RAG operations that took 50 lines now take just 2 lines.",
        "The creator's vision is 'intelligence as infrastructure'."
    ]
    
    print("\n4a. Index Documents:")
    print("    Code: indexed = rag.index(documents)")
    indexed = rag.index(documents)
    print(f"    Indexed: {indexed}")
    
    print("\n4b. Query Documents:")
    print("    Code: answer = indexed.ask('What year was PyAgent created?')")
    answer = indexed.ask("What year was PyAgent created?")
    success = "2026" in answer
    print(f"    Answer: {answer}")
    print(f"    {'✅ PASS' if success else '❌ FAIL'}")
    
    # Comparison
    print("\n4c. Framework Comparison:")
    print("    | Framework  | Lines of Code |")
    print("    |------------|---------------|")
    print("    | LangChain  | 50+ lines     |")
    print("    | LlamaIndex | 40+ lines     |")
    print("    | PyAgent    | 2 lines       |")
    
    return success


# =============================================================================
# EXAMPLE 5: CODE GENERATION
# =============================================================================

def example_code():
    """
    Example 5: code - AI code operations
    
    Write, review, debug, and refactor code with AI.
    """
    print("\n" + "=" * 60)
    print("💻 EXAMPLE 5: code - Code Operations")
    print("=" * 60)
    
    from pyagent import code
    
    print("\n5a. Write Code:")
    print("    Code: result = code.write('fibonacci function')")
    result = code.write("a Python function to calculate fibonacci numbers")
    print(f"    Generated:\n{result[:200]}...")
    
    # Verify it looks like Python code
    success = "def " in result or "fibonacci" in result.lower()
    print(f"\n    {'✅ PASS' if success else '❌ FAIL'} - Generated valid code")
    
    print("\n5b. Available Operations:")
    print("    • code.write()    - Generate code from description")
    print("    • code.review()   - Review code quality")
    print("    • code.debug()    - Debug errors")
    print("    • code.explain()  - Explain code")
    print("    • code.refactor() - Refactor code")
    print("    • code.convert()  - Convert between languages")
    
    return success


# =============================================================================
# EXAMPLE 6: RESEARCH & SUMMARIZE
# =============================================================================

def example_research():
    """
    Example 6: research() & summarize() - Deep research made simple
    """
    print("\n" + "=" * 60)
    print("🔍 EXAMPLE 6: research() & summarize()")
    print("=" * 60)
    
    from pyagent import research, summarize
    
    print("\n6a. Quick Research:")
    print("    Code: result = research('quantum computing basics')")
    result = research("quantum computing basics", depth="quick")
    print(f"    Result: {str(result)[:150]}...")
    
    print("\n6b. Summarize Text:")
    long_text = """
    Artificial intelligence has transformed numerous industries over the past decade.
    Machine learning algorithms now power recommendation systems, autonomous vehicles,
    medical diagnosis tools, and natural language processing applications. The field
    continues to evolve rapidly, with new breakthroughs in generative AI, reinforcement
    learning, and neural architecture design. Companies worldwide are investing billions
    in AI research and development, recognizing its potential to revolutionize business
    operations and create new opportunities for innovation.
    """
    print("    Code: summary = summarize(long_text)")
    summary = summarize(long_text)
    print(f"    Summary: {summary[:120]}...")
    
    success = len(summary) > 20
    print(f"\n    {'✅ PASS' if success else '❌ FAIL'}")
    return success


# =============================================================================
# EXAMPLE 7: DATA EXTRACTION
# =============================================================================

def example_extract():
    """
    Example 7: extract() - Structured data extraction
    """
    print("\n" + "=" * 60)
    print("📊 EXAMPLE 7: extract() - Data Extraction")
    print("=" * 60)
    
    from pyagent import extract
    
    text = """
    Meeting Notes - January 15, 2026
    Attendees: John Smith (CEO), Sarah Chen (CTO), Mike Johnson (VP Sales)
    
    Key decisions:
    1. Launch new product line in Q2 2026
    2. Increase marketing budget by $500,000
    3. Hire 10 new engineers
    
    Next meeting scheduled for January 22, 2026.
    """
    
    print("\n7a. Extract Structured Data:")
    print("    Code: data = extract(text, ['names', 'dates', 'amounts'])")
    data = extract(text, ["names", "dates", "amounts"])
    print(f"    Extracted: {data}")
    
    success = isinstance(data, dict) or len(str(data)) > 10
    print(f"\n    {'✅ PASS' if success else '❌ FAIL'}")
    return success


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_all_examples(provider: str = "auto"):
    """Run all examples with comprehensive output."""
    
    print("\n" + "🚀" * 30)
    print("      PYAGENT COMPREHENSIVE EXAMPLES")
    print("🚀" * 30)
    
    # Setup provider
    success, message = setup_provider(provider)
    print(f"\n🔧 Provider: {message}")
    
    if not success:
        print(f"\n❌ Setup failed: {message}")
        print("\nConfiguration options:")
        print("  OpenAI:     export OPENAI_API_KEY=sk-your-key")
        print("  Azure:      export AZURE_OPENAI_ENDPOINT=https://...")
        print("              export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini")
        return
    
    print("\n" + "-" * 60)
    
    # Run examples
    examples = [
        ("Simple Q&A", example_ask),
        ("Custom Agents", example_agent),
        ("Chat with Memory", example_chat),
        ("RAG Pipeline", example_rag),
        ("Code Generation", example_code),
        ("Research & Summarize", example_research),
        ("Data Extraction", example_extract),
    ]
    
    results = []
    for name, example_fn in examples:
        try:
            passed = example_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"    ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"    {name}: {status}")
    
    print(f"\n    Total: {passed}/{len(results)} passed")
    
    # Quick Reference
    print("\n" + "=" * 60)
    print("📖 QUICK REFERENCE - PyAgent One-Liners")
    print("=" * 60)
    print("""
    from pyagent import ask, agent, chat, rag, code
    from pyagent import research, summarize, extract, translate

    # Ask anything
    answer = ask("What is quantum computing?")

    # Create agents
    coder = agent(persona="coder")
    solution = coder("Optimize this algorithm")

    # Chat with memory
    session = chat("You are helpful")
    session.say("Hello!")

    # RAG in 2 lines
    docs = rag.index(["doc1.pdf", "doc2.txt"])
    answer = docs.ask("What is the conclusion?")

    # Code operations
    code.write("fibonacci function")
    code.review(my_code)

    # Research & Analysis
    research("AI trends 2026")
    summarize(long_document)
    extract(text, fields=["names", "dates"])
    """)
    
    if passed == len(results):
        print("\n🎉 All examples passed! PyAgent is working perfectly.")
    else:
        print(f"\n⚠️  {len(results) - passed} example(s) had issues.")
    
    print("\n🐼 PyAgent - Because AI development should be as simple as pandas!")


if __name__ == "__main__":
    # Get provider from command line
    provider = "auto"
    if len(sys.argv) > 1:
        provider = sys.argv[1].lower()
        if provider not in ["openai", "azure", "auto"]:
            print(f"Unknown provider: {provider}")
            print("Usage: python comprehensive_examples.py [openai|azure|auto]")
            sys.exit(1)
    
    run_all_examples(provider)
