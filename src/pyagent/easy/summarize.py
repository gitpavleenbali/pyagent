"""
summarize() - Summarize anything in one line

Works with text, files, URLs, and more.

Examples:
    >>> from pyagent import summarize
    >>> summarize("Long article text here...")
    'This article discusses...'
    
    >>> summarize("./document.pdf")
    'The document covers...'
    
    >>> summarize("https://example.com/article")
    'The webpage explains...'
"""

from typing import Optional, Union, List
from pathlib import Path
import os

from pyagent.easy.llm_interface import get_llm


def summarize(
    content: Union[str, Path, List[str]],
    *,
    length: str = "medium",  # "short", "medium", "long", "bullet"
    focus: str = None,
    style: str = None,  # "academic", "casual", "executive", "technical"
    max_words: int = None,
    bullet_points: bool = False,
    model: str = None,
    **kwargs
) -> str:
    """
    Summarize any content - text, files, or URLs.
    
    Args:
        content: Text to summarize, file path, URL, or list of texts
        length: Summary length ("short", "medium", "long", "bullet")
        focus: Specific aspect to focus on
        style: Writing style ("academic", "casual", "executive", "technical")
        max_words: Maximum words in summary
        bullet_points: Return as bullet points
        model: Override default model
        **kwargs: Additional parameters
        
    Returns:
        str: The summary
    
    Examples:
        >>> summarize("Long text here...", length="short")
        'Brief summary...'
        
        >>> summarize("./report.pdf", bullet_points=True)
        '• Point 1\\n• Point 2...'
        
        >>> summarize("https://news.site/article")
        'The article discusses...'
    """
    # Determine content type and extract text
    text = _extract_content(content)
    
    if isinstance(content, list):
        text = "\n\n---\n\n".join(content)
    
    # Build system prompt
    style_prompts = {
        "academic": "Write in an academic, formal style with precise language.",
        "casual": "Write in a friendly, conversational style.",
        "executive": "Write in a professional, business-oriented style suitable for executives.",
        "technical": "Write in a technical style with appropriate terminology.",
    }
    
    length_prompts = {
        "short": "Keep the summary very brief (1-2 sentences).",
        "medium": "Provide a moderate-length summary (1-2 paragraphs).",
        "long": "Provide a comprehensive summary covering all main points.",
        "bullet": "Format as bullet points.",
    }
    
    system_parts = ["You are an expert summarizer. Create clear, accurate summaries."]
    
    if style and style in style_prompts:
        system_parts.append(style_prompts[style])
    
    if bullet_points or length == "bullet":
        system_parts.append("Format your summary as bullet points using •")
    else:
        system_parts.append(length_prompts.get(length, length_prompts["medium"]))
    
    if max_words:
        system_parts.append(f"Keep the summary under {max_words} words.")
    
    system = " ".join(system_parts)
    
    # Build prompt
    prompt = f"Summarize the following content:\n\n{text}"
    
    if focus:
        prompt += f"\n\nFocus particularly on: {focus}"
    
    # Get LLM and generate
    llm_kwargs = {"model": model} if model else {}
    llm = get_llm(**llm_kwargs)
    
    response = llm.complete(prompt, system=system, temperature=0.3, **kwargs)
    
    return response.content


def _extract_content(content: Union[str, Path]) -> str:
    """Extract text content from various sources."""
    
    # Handle Path objects
    if isinstance(content, Path):
        content = str(content)
    
    # Check if it's a file path
    if isinstance(content, str) and len(content) < 500:
        # Check for file extensions
        if any(content.endswith(ext) for ext in ['.txt', '.md', '.pdf', '.docx', '.html']):
            if os.path.exists(content):
                return _read_file(content)
        
        # Check for URL
        if content.startswith(('http://', 'https://', 'www.')):
            return _fetch_url(content)
        
        # Check if it's a valid path
        if os.path.exists(content):
            return _read_file(content)
    
    # Assume it's raw text
    return content


def _read_file(filepath: str) -> str:
    """Read content from a file."""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.txt' or ext == '.md':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    elif ext == '.pdf':
        try:
            import pypdf
            reader = pypdf.PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except ImportError:
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(filepath)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except ImportError:
                raise ImportError(
                    "PDF reading requires pypdf or PyPDF2. Install with: pip install pypdf"
                )
    
    elif ext == '.docx':
        try:
            import docx
            doc = docx.Document(filepath)
            return "\n".join(para.text for para in doc.paragraphs)
        except ImportError:
            raise ImportError(
                "DOCX reading requires python-docx. Install with: pip install python-docx"
            )
    
    elif ext == '.html':
        try:
            from bs4 import BeautifulSoup
            with open(filepath, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            return soup.get_text(separator='\n', strip=True)
        except ImportError:
            # Fallback to basic reading
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
    
    else:
        # Generic text file
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()


def _fetch_url(url: str) -> str:
    """Fetch content from a URL."""
    try:
        import requests
        from bs4 import BeautifulSoup
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        return soup.get_text(separator='\n', strip=True)
    
    except ImportError:
        raise ImportError(
            "URL fetching requires requests and beautifulsoup4. "
            "Install with: pip install requests beautifulsoup4"
        )


# Convenient aliases
def tldr(content: str, **kwargs) -> str:
    """Get a very short summary (TL;DR)."""
    return summarize(content, length="short", **kwargs)


def bullet_summary(content: str, **kwargs) -> str:
    """Get summary as bullet points."""
    return summarize(content, bullet_points=True, **kwargs)
