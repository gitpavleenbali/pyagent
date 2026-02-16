"""
RAG Module - Retrieval Augmented Generation in 2 Lines

The simplest RAG implementation possible. Index documents and ask questions.

Examples:
    >>> from pyagent import rag
    >>> 
    >>> # Index and query in 2 lines
    >>> docs = rag.index("./documents")
    >>> answer = docs.ask("What is the main conclusion?")
    >>> 
    >>> # Or even simpler - one line!
    >>> answer = rag.ask("./documents", "What is the main topic?")
    >>>
    >>> # Quick RAG from text
    >>> answer = rag.from_text(long_text, "Summarize the key points")
"""

import os
from typing import Union, List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

from pyagent.easy.llm_interface import get_llm
from pyagent.easy.summarize import _extract_content, _read_file


@dataclass 
class IndexedDocuments:
    """
    A simple indexed document collection for RAG.
    
    Provides immediate access to document-based QA without complex setup.
    """
    
    content: str
    source: str
    chunks: List[str] = field(default_factory=list)
    chunk_size: int = 1000
    overlap: int = 200
    
    def __post_init__(self):
        """Chunk the content on initialization."""
        if not self.chunks:
            self.chunks = self._chunk_text(self.content)
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > self.chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - self.overlap
        
        return [c for c in chunks if c]  # Remove empty chunks
    
    def ask(
        self,
        question: str,
        *,
        top_k: int = 3,
        model: str = None,
        include_sources: bool = False,
        **kwargs
    ) -> str:
        """
        Ask a question about the indexed documents.
        
        Args:
            question: Your question
            top_k: Number of relevant chunks to use
            model: Override default model
            include_sources: Include source references
            **kwargs: Additional parameters
            
        Returns:
            str: The answer based on the documents
        """
        # Simple keyword-based retrieval (works without embeddings!)
        relevant_chunks = self._retrieve(question, top_k=top_k)
        
        # Build context
        context = "\n\n---\n\n".join(relevant_chunks)
        
        # Get LLM
        llm_kwargs = {"model": model} if model else {}
        llm = get_llm(**llm_kwargs)
        
        system = """You are a helpful assistant that answers questions based on the provided context.
Only answer based on the information in the context. If the answer cannot be found in the context, say so clearly.
Be concise but comprehensive."""
        
        prompt = f"""Context from documents:
{context}

Question: {question}

Answer based only on the context above:"""
        
        response = llm.complete(prompt, system=system, temperature=0.3, **kwargs)
        
        answer = response.content
        
        if include_sources:
            answer += f"\n\n[Source: {self.source}]"
        
        return answer
    
    def _retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Simple keyword-based retrieval.
        
        For production, upgrade to embeddings-based retrieval.
        """
        # Tokenize query
        query_terms = set(query.lower().split())
        
        # Score each chunk by keyword overlap
        scored_chunks = []
        for chunk in self.chunks:
            chunk_terms = set(chunk.lower().split())
            score = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            # Boost if exact phrase appears
            if query.lower() in chunk.lower():
                score += 1.0
            scored_chunks.append((score, chunk))
        
        # Sort by score and return top_k
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]
    
    def search(self, query: str, top_k: int = 5) -> List[str]:
        """Search for relevant passages."""
        return self._retrieve(query, top_k=top_k)
    
    def __repr__(self) -> str:
        return f"IndexedDocuments(source='{self.source}', chunks={len(self.chunks)})"


def index(
    source: Union[str, Path, List[str]],
    *,
    chunk_size: int = 1000,
    overlap: int = 200,
    **kwargs
) -> IndexedDocuments:
    """
    Index documents for RAG.
    
    Args:
        source: File path, directory, URL, or list of texts
        chunk_size: Size of text chunks
        overlap: Overlap between chunks
        **kwargs: Additional parameters
        
    Returns:
        IndexedDocuments: An indexed collection you can query
    
    Examples:
        >>> docs = rag.index("./research_papers")
        >>> docs.ask("What methodology was used?")
        
        >>> docs = rag.index("https://website.com/article")
        >>> docs.ask("What is the main argument?")
    """
    # Handle list of texts
    if isinstance(source, list):
        content = "\n\n---\n\n".join(source)
        return IndexedDocuments(
            content=content,
            source="multiple_sources",
            chunk_size=chunk_size,
            overlap=overlap
        )
    
    source_path = str(source)
    
    # Check if it's a directory
    if os.path.isdir(source_path):
        content = _read_directory(source_path)
        source_name = os.path.basename(source_path)
    else:
        content = _extract_content(source_path)
        source_name = source_path
    
    return IndexedDocuments(
        content=content,
        source=source_name,
        chunk_size=chunk_size,
        overlap=overlap
    )


def _read_directory(directory: str) -> str:
    """Read all readable files from a directory."""
    texts = []
    supported_extensions = {'.txt', '.md', '.pdf', '.docx', '.html', '.py', '.js', '.json'}
    
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_extensions:
                filepath = os.path.join(root, file)
                try:
                    text = _read_file(filepath)
                    texts.append(f"=== {file} ===\n{text}")
                except Exception:
                    continue  # Skip unreadable files
    
    return "\n\n".join(texts)


def ask(
    source: Union[str, Path, IndexedDocuments, List[str]] = None,
    question: str = None,
    *,
    documents: List[str] = None,
    **kwargs
) -> str:
    """
    One-line RAG - index and query in a single call.
    
    Args:
        source: File, directory, URL, or already indexed documents
        question: Your question
        documents: Alternative - list of document strings
        **kwargs: Additional parameters
        
    Returns:
        str: The answer
    
    Examples:
        >>> rag.ask("./docs", "What is the conclusion?")
        'The main conclusion is...'
        
        >>> rag.ask("https://arxiv.org/paper", "What method is proposed?")
        'The paper proposes...'
        
        >>> rag.ask("What topic?", documents=["Doc 1", "Doc 2"])
        'The topic is...'
    """
    # Handle documents= kwarg (question as first arg)
    if documents is not None:
        actual_question = source  # First arg is actually the question
        docs = index(documents)
        return docs.ask(actual_question, **kwargs)
    
    if isinstance(source, IndexedDocuments):
        docs = source
    elif isinstance(source, list):
        docs = index(source)
    else:
        docs = index(source)
    
    return docs.ask(question, **kwargs)


def from_text(
    text: str,
    question: str,
    **kwargs
) -> str:
    """
    RAG from raw text.
    
    Args:
        text: The text to query
        question: Your question
        **kwargs: Additional parameters
        
    Returns:
        str: The answer
    
    Examples:
        >>> long_article = "..."
        >>> rag.from_text(long_article, "What is the key takeaway?")
    """
    docs = IndexedDocuments(content=text, source="text_input")
    return docs.ask(question, **kwargs)


def from_url(url: str, question: str, **kwargs) -> str:
    """
    Fetch URL and answer questions.
    
    Args:
        url: The URL to fetch
        question: Your question
        
    Returns:
        str: The answer
    """
    content = _extract_content(url)
    return from_text(content, question, **kwargs)


# Aliases for convenience
query = ask
search = lambda source, query, **kw: index(source, **kw).search(query)
