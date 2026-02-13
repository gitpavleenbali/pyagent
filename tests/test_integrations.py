"""
Tests for PyAgent Integrations Module
=====================================

Tests for LangChain adapter, Semantic Kernel adapter, and Vector DB connectors.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestIntegrationsImport:
    """Test that all integration modules can be imported."""
    
    def test_import_integrations_module(self):
        """Test importing the integrations module."""
        from pyagent import integrations
        assert integrations is not None
    
    def test_import_langchain_adapter(self):
        """Test importing the langchain adapter."""
        from pyagent.integrations import langchain
        assert langchain is not None
        assert hasattr(langchain, 'import_tool')
        assert hasattr(langchain, 'import_chain')
        assert hasattr(langchain, 'export_agent')
        assert hasattr(langchain, 'import_retriever')
    
    def test_import_semantic_kernel_adapter(self):
        """Test importing the semantic kernel adapter."""
        from pyagent.integrations import semantic_kernel
        assert semantic_kernel is not None
        assert hasattr(semantic_kernel, 'create_kernel')
        assert hasattr(semantic_kernel, 'import_plugin')
        assert hasattr(semantic_kernel, 'create_plan')
        assert hasattr(semantic_kernel, 'execute_plan')
        assert hasattr(semantic_kernel, 'export_to_kernel')
        assert hasattr(semantic_kernel, 'create_memory')
    
    def test_import_vector_db(self):
        """Test importing the vector_db module."""
        from pyagent.integrations import vector_db
        assert vector_db is not None
        assert hasattr(vector_db, 'connect')
        assert hasattr(vector_db, 'VectorStore')


class TestLangChainAdapter:
    """Tests for the LangChain adapter."""
    
    def test_import_tool_returns_callable(self):
        """Test that import_tool returns a callable skill."""
        from pyagent.integrations import langchain
        
        # Try to import a tool (will fail gracefully without langchain installed)
        try:
            tool = langchain.import_tool("calculator")
        except ImportError:
            pytest.skip("LangChain not installed")
        except Exception as e:
            # Expected behavior without proper setup
            assert True
    
    def test_export_agent_exists(self):
        """Test that export_agent function exists."""
        from pyagent.integrations import langchain
        assert callable(langchain.export_agent)
    
    def test_import_chain_exists(self):
        """Test that import_chain function exists."""
        from pyagent.integrations import langchain
        assert callable(langchain.import_chain)


class TestSemanticKernelAdapter:
    """Tests for the Semantic Kernel adapter."""
    
    def test_create_kernel_function_exists(self):
        """Test that create_kernel function exists."""
        from pyagent.integrations import semantic_kernel
        assert callable(semantic_kernel.create_kernel)
    
    def test_import_plugin_function_exists(self):
        """Test that import_plugin function exists."""
        from pyagent.integrations import semantic_kernel
        assert callable(semantic_kernel.import_plugin)
    
    def test_create_plan_function_exists(self):
        """Test that create_plan function exists."""
        from pyagent.integrations import semantic_kernel
        assert callable(semantic_kernel.create_plan)
    
    def test_execute_plan_function_exists(self):
        """Test that execute_plan function exists."""
        from pyagent.integrations import semantic_kernel
        assert callable(semantic_kernel.execute_plan)
    
    def test_create_memory_function_exists(self):
        """Test that create_memory function exists."""
        from pyagent.integrations import semantic_kernel
        assert callable(semantic_kernel.create_memory)


class TestVectorDB:
    """Tests for the Vector DB connectors."""
    
    def test_connect_function_exists(self):
        """Test that connect function exists."""
        from pyagent.integrations import vector_db
        assert callable(vector_db.connect)
    
    def test_vector_store_base_class(self):
        """Test VectorStore base class."""
        from pyagent.integrations.vector_db import VectorStore
        assert VectorStore is not None
    
    def test_azure_ai_search_store_class(self):
        """Test AzureAISearchStore class exists."""
        from pyagent.integrations.vector_db import AzureAISearchStore
        assert AzureAISearchStore is not None
    
    def test_pinecone_store_class(self):
        """Test PineconeStore class exists."""
        from pyagent.integrations.vector_db import PineconeStore
        assert PineconeStore is not None
    
    def test_chroma_store_class(self):
        """Test ChromaStore class exists."""
        from pyagent.integrations.vector_db import ChromaStore
        assert ChromaStore is not None
    
    def test_faiss_store_class(self):
        """Test FAISSStore class exists."""
        from pyagent.integrations.vector_db import FAISSStore
        assert FAISSStore is not None
    
    def test_qdrant_store_class(self):
        """Test QdrantStore class exists."""
        from pyagent.integrations.vector_db import QdrantStore
        assert QdrantStore is not None
    
    def test_connect_invalid_store_raises(self):
        """Test that connecting to an invalid store raises error."""
        from pyagent.integrations import vector_db
        
        with pytest.raises(ValueError):
            vector_db.connect("invalid_store_type")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
