# pyright: reportMissingImports=false, reportUnusedVariable=false, reportRedeclaration=false, reportUnusedImport=false, reportGeneralTypeIssues=false
"""
pyai Azure Integration Tests
================================

Integration tests that require Azure OpenAI connection.
These tests are skipped if Azure is not configured.

Environment Variables Required:
- AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL
- AZURE_OPENAI_DEPLOYMENT: Deployment name (e.g., gpt-4o-mini)

Run: pytest tests/test_azure_integration.py -v
"""

import pytest
import os
import sys

# Check if Azure is configured
AZURE_CONFIGURED = bool(
    os.environ.get("AZURE_OPENAI_ENDPOINT") and 
    os.environ.get("AZURE_OPENAI_DEPLOYMENT")
)

# Skip all tests if Azure not configured
pytestmark = pytest.mark.skipif(
    not AZURE_CONFIGURED,
    reason="Azure OpenAI not configured. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT."
)


@pytest.fixture(scope="module")
def azure_config():
    """Configure Azure OpenAI for tests."""
    import pyai
    
    pyai.configure(
        provider="azure",
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    )
    
    return {
        "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
        "deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    }


class TestAzureConnection:
    """Test Azure OpenAI connection."""
    
    def test_azure_ask(self, azure_config):
        """Test basic ask function with Azure."""
        from pyai import ask
        
        response = ask("What is 2 + 2?")
        assert response is not None
        assert len(response) > 0
        assert "4" in response
    
    def test_azure_agent(self, azure_config):
        """Test agent with Azure."""
        from pyai import agent
        
        math_agent = agent("You are a math tutor. Answer concisely.")
        response = math_agent("What is the square root of 16?")
        
        assert response is not None
        assert "4" in response


class TestAzureEasyFunctions:
    """Test easy module functions with Azure."""
    
    def test_summarize(self, azure_config):
        """Test summarize function."""
        from pyai import summarize
        
        text = """
        Artificial intelligence (AI) is intelligence demonstrated by machines, 
        as opposed to intelligence of humans and other animals. AI research 
        has been defined as the field of study of intelligent agents, which 
        refers to any system that perceives its environment and takes actions 
        that maximize its chance of achieving its goals.
        """
        
        result = summarize(text)
        assert result is not None
        assert len(result) > 10  # Should produce meaningful summary
        # Note: LLMs may sometimes expand rather than strictly shorten
    
    def test_extract(self, azure_config):
        """Test extract function."""
        from pyai import extract
        
        text = "John Smith works at Microsoft in Seattle. His email is john@microsoft.com."
        
        result = extract(text, schema={
            "name": str,
            "company": str,
            "location": str,
            "email": str
        })
        
        assert result is not None
        assert isinstance(result, dict)
    
    def test_translate(self, azure_config):
        """Test translate function."""
        from pyai import translate
        
        result = translate("Hello, world!", to="Spanish")
        
        assert result is not None
        assert "hola" in result.lower() or "mundo" in result.lower()
    
    def test_generate(self, azure_config):
        """Test generate function."""
        from pyai import generate
        
        result = generate("haiku about Python programming")
        
        assert result is not None
        assert len(result) > 10
    
    def test_chat(self, azure_config):
        """Test chat function."""
        from pyai import chat
        
        # Create a chat session
        session = chat()
        response = session("My name is TestUser")
        
        assert response is not None
        
        # Follow-up
        response2 = session("What is my name?")
        assert "TestUser" in response2 or "test" in response2.lower()


class TestAzureAgentFeatures:
    """Test agent features with Azure."""
    
    def test_agent_with_persona(self, azure_config):
        """Test agent with persona."""
        from pyai import agent
        
        coder = agent(persona="coder")
        response = coder("Write a one-line Python hello world")
        
        assert response is not None
        assert "print" in response.lower()
    
    def test_agent_with_memory(self, azure_config):
        """Test agent with memory."""
        from pyai import agent
        
        bot = agent("You remember everything", memory=True, name="MemBot")
        
        bot("Remember the code word is 'ALPHA'")
        response = bot("What is the code word?")
        
        assert "ALPHA" in response.upper()
    
    def test_multiple_agents(self, azure_config):
        """Test multiple agents working."""
        from pyai import agent
        
        researcher = agent(persona="researcher", name="Researcher")
        writer = agent(persona="writer", name="Writer")
        
        research = researcher("Briefly describe what Python is")
        article = writer(f"Write a short paragraph about: {research}")
        
        assert research is not None
        assert article is not None
        assert len(article) > 50  # Both should produce meaningful output
        assert len(research) > 50


class TestAzureHandoff:
    """Test multi-agent handoff with Azure."""
    
    def test_simple_handoff(self, azure_config):
        """Test simple agent handoff."""
        from pyai import handoff, agent
        
        a1 = agent("You are agent 1. Say 'ALPHA' and pass to agent 2.", name="Agent1")
        a2 = agent("You are agent 2. Say 'BETA' when you receive handoff.", name="Agent2")
        
        result = handoff(a1, a2, "Begin", reason="Testing handoff")
        
        assert result is not None
        assert len(result.agents_used) == 2
    
    def test_team_routing(self, azure_config):
        """Test team routing."""
        from pyai import handoff, agent
        
        coder = agent(persona="coder", name="Coder")
        writer = agent(persona="writer", name="Writer")
        
        team = handoff.team([coder, writer])
        result = team.route("Write a Python function to greet someone")
        
        assert result is not None
        assert "Coder" in result.agents_used or "def" in str(result)


class TestAzureCode:
    """Test code generation with Azure."""
    
    def test_code_generate(self, azure_config):
        """Test code generation."""
        from pyai import code
        
        result = code.write("function to calculate factorial in Python")
        
        assert result is not None
        assert "def" in result or "factorial" in result.lower()
    
    def test_code_explain(self, azure_config):
        """Test code explanation."""
        from pyai import code
        
        sample_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        
        explanation = code.explain(sample_code)
        
        assert explanation is not None
        assert "fibonacci" in explanation.lower() or "recursive" in explanation.lower()


class TestAzureRAG:
    """Test RAG functionality with Azure."""
    
    def test_rag_simple(self, azure_config):
        """Test simple RAG."""
        from pyai import rag
        
        documents = [
            "pyai is a Python library for building AI agents.",
            "pyai supports Azure OpenAI, OpenAI, and Anthropic.",
            "pyai provides simple one-liner functions for AI tasks."
        ]
        
        result = rag.ask("What providers does pyai support?", documents=documents)
        
        assert result is not None
        assert "azure" in result.lower() or "openai" in result.lower()


class TestAzureGuardrails:
    """Test guardrails with Azure."""
    
    def test_guardrails_wrap_function(self, azure_config):
        """Test wrapping function with guardrails."""
        from pyai import guardrails, ask
        
        safe_ask = guardrails.wrap(ask, block_pii=True)
        
        # Clean input should work
        response = safe_ask("What is the capital of France?")
        assert response is not None
        assert "Paris" in response
    
    def test_guardrails_blocks_pii(self, azure_config):
        """Test that guardrails blocks PII."""
        from pyai import guardrails, ask
        
        safe_ask = guardrails.wrap(ask, block_pii=True)
        
        # PII input should be blocked
        try:
            response = safe_ask("My SSN is 123-45-6789")
            # If it doesn't raise, check the result
            assert response is None or "blocked" in str(response).lower()
        except Exception as e:
            # Expected - guardrails blocked the input
            assert "blocked" in str(e).lower() or "pii" in str(e).lower()


class TestAzureTracing:
    """Test tracing with Azure."""
    
    def test_traced_call(self, azure_config):
        """Test traced LLM call."""
        from pyai import trace, ask
        
        trace.enable()
        trace.clear()
        
        response = ask("Say 'hello'")
        
        summary = trace.summary()
        assert summary["llm_calls"] >= 1
        
        trace.disable()
        trace.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
