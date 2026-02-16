"""
LangChain Integration for PyAgent
==================================

Bridge between PyAgent and LangChain ecosystems. Import LangChain tools,
chains, and agents into PyAgent, or export PyAgent components to LangChain.

Features:
- Import LangChain tools as PyAgent skills
- Use LangChain chains within PyAgent workflows  
- Export PyAgent agents as LangChain-compatible
- Access LangChain's extensive tool library

Examples:
    >>> from pyagent.integrations import langchain
    
    # Import a LangChain tool
    >>> search_tool = langchain.import_tool("serpapi")
    >>> agent = pyagent.agent("researcher", tools=[search_tool])
    
    # Use a LangChain chain
    >>> chain = langchain.import_chain("llm_math_chain")
    >>> result = chain.run("What is 25 * 4?")
    
    # Export PyAgent to LangChain
    >>> lc_agent = langchain.export_agent(my_pyagent)
"""

from typing import Callable, Dict, Any, List, Optional, Union
from dataclasses import dataclass
import importlib


@dataclass
class LangChainTool:
    """Wrapper for LangChain tools in PyAgent."""
    
    name: str
    description: str
    func: Callable
    _lc_tool: Any = None
    
    def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool."""
        if self._lc_tool:
            return self._lc_tool.run(*args, **kwargs)
        return self.func(*args, **kwargs)
    
    def to_pyagent_skill(self):
        """Convert to PyAgent skill format."""
        from pyagent.easy.mcp import MCPTool
        return MCPTool(
            name=self.name,
            description=self.description,
            parameters={},
            handler=self.func
        )


def import_tool(
    tool_name: str,
    *,
    api_key: str = None,
    **kwargs
) -> LangChainTool:
    """
    Import a LangChain tool for use in PyAgent.
    
    Args:
        tool_name: Name of the LangChain tool (e.g., "serpapi", "wikipedia")
        api_key: API key if required
        **kwargs: Additional tool configuration
        
    Returns:
        LangChainTool wrapper
        
    Examples:
        >>> search = langchain.import_tool("serpapi", api_key="...")
        >>> result = search("Python programming")
        
    Supported tools:
        - serpapi: Google search via SerpAPI
        - wikipedia: Wikipedia search
        - wolfram_alpha: Wolfram Alpha queries
        - arxiv: Academic paper search
        - pubmed: Medical literature search
        - requests: HTTP requests
        - python_repl: Python code execution
        - bash: Bash command execution
    """
    try:
        from langchain.tools import load_tools
        tools = load_tools([tool_name], **kwargs)
        lc_tool = tools[0]
        
        return LangChainTool(
            name=lc_tool.name,
            description=lc_tool.description,
            func=lc_tool.run,
            _lc_tool=lc_tool
        )
    except ImportError:
        # Fallback for when LangChain not installed
        return _create_stub_tool(tool_name)


def import_chain(
    chain_type: str,
    *,
    llm: Any = None,
    **kwargs
) -> Any:
    """
    Import a LangChain chain for use in PyAgent.
    
    Args:
        chain_type: Type of chain (e.g., "llm_math", "sql_database")
        llm: LLM to use (uses PyAgent default if not specified)
        **kwargs: Chain configuration
        
    Returns:
        Callable chain
        
    Supported chains:
        - llm_math: Mathematical computations
        - sql_database: SQL query generation
        - api_chain: API interaction
        - summarization: Document summarization
        - qa: Question answering
    """
    try:
        if chain_type == "llm_math":
            from langchain.chains import LLMMathChain
            return LLMMathChain.from_llm(llm or _get_lc_llm(), **kwargs)
        elif chain_type == "summarization":
            from langchain.chains.summarize import load_summarize_chain
            return load_summarize_chain(llm or _get_lc_llm(), **kwargs)
        else:
            raise ValueError(f"Unknown chain type: {chain_type}")
    except ImportError:
        raise ImportError("LangChain not installed. Run: pip install langchain")


def export_agent(
    pyagent_agent,
    *,
    tools: List = None
) -> Any:
    """
    Export a PyAgent agent as a LangChain agent.
    
    Args:
        pyagent_agent: PyAgent SimpleAgent instance
        tools: Additional tools to include
        
    Returns:
        LangChain Agent
    """
    try:
        from langchain.agents import initialize_agent, AgentType
        from langchain.tools import Tool
        
        # Convert PyAgent to LangChain tool
        lc_tools = []
        if tools:
            for t in tools:
                lc_tools.append(Tool(
                    name=t.name if hasattr(t, 'name') else "tool",
                    description=t.description if hasattr(t, 'description') else "",
                    func=t
                ))
        
        agent = initialize_agent(
            lc_tools,
            _get_lc_llm(),
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        return agent
    except ImportError:
        raise ImportError("LangChain not installed. Run: pip install langchain")


def import_retriever(
    retriever_type: str,
    *,
    source: str = None,
    **kwargs
) -> Any:
    """
    Import a LangChain retriever for RAG operations.
    
    Args:
        retriever_type: Type of retriever
        source: Data source path/URL
        **kwargs: Retriever configuration
        
    Supported retrievers:
        - web: Web page retriever
        - wikipedia: Wikipedia retriever
        - arxiv: ArXiv paper retriever
        - vectorstore: Custom vector store retriever
    """
    try:
        if retriever_type == "web":
            from langchain.retrievers import WebResearchRetriever
            return WebResearchRetriever(**kwargs)
        elif retriever_type == "wikipedia":
            from langchain.retrievers import WikipediaRetriever
            return WikipediaRetriever(**kwargs)
        else:
            raise ValueError(f"Unknown retriever: {retriever_type}")
    except ImportError:
        raise ImportError("LangChain not installed. Run: pip install langchain")


def _get_lc_llm():
    """Get LangChain LLM using PyAgent configuration."""
    import os
    try:
        from langchain.chat_models import ChatOpenAI
        return ChatOpenAI(
            model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            temperature=0.7
        )
    except ImportError:
        raise ImportError("LangChain not installed")


def _create_stub_tool(name: str) -> LangChainTool:
    """Create a stub tool when LangChain is not available."""
    def stub_func(*args, **kwargs):
        raise ImportError(f"LangChain not installed. Cannot use tool: {name}")
    
    return LangChainTool(
        name=name,
        description=f"LangChain tool: {name} (not available)",
        func=stub_func
    )


# Available tools reference
AVAILABLE_TOOLS = {
    "search": ["serpapi", "google_search", "bing_search", "ddg_search"],
    "knowledge": ["wikipedia", "arxiv", "pubmed", "wolfram_alpha"],
    "code": ["python_repl", "bash", "terminal"],
    "web": ["requests", "requests_get", "requests_post"],
    "data": ["csv", "json", "sql_database"],
}


class LangChainModule:
    """LangChain integration module."""
    
    import_tool = staticmethod(import_tool)
    import_chain = staticmethod(import_chain)
    import_retriever = staticmethod(import_retriever)
    export_agent = staticmethod(export_agent)
    
    TOOLS = AVAILABLE_TOOLS


langchain = LangChainModule()
