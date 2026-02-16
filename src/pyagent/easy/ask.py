"""
ask() - The simplest way to get answers from AI

One function to rule them all. Ask anything, get intelligent answers.

Examples:
    >>> from pyagent import ask
    >>> ask("What is the capital of France?")
    'Paris'
    
    >>> ask("Explain quantum computing", detailed=True)
    'Quantum computing is a type of computation...'
    
    >>> ask("What's 15% of 847?")
    '127.05'
    
    >>> ask("Write a haiku about Python")
    'Code flows like water...'
"""

from typing import Optional, Union, List, Any
from pyagent.easy.llm_interface import get_llm, LLMResponse


def ask(
    question: str,
    *,
    context: str = None,
    detailed: bool = False,
    creative: bool = False,
    concise: bool = False,
    format: str = None,
    model: str = None,
    temperature: float = None,
    as_json: bool = False,
    schema: dict = None,
    **kwargs
) -> Union[str, dict]:
    """
    Ask anything, get intelligent answers. The simplest AI function.
    
    Args:
        question: Your question or prompt
        context: Optional context to include (text, data, etc.)
        detailed: Get a detailed, comprehensive answer
        creative: Enable more creative responses
        concise: Get brief, to-the-point answers
        format: Output format ("bullet", "numbered", "markdown", "code")
        model: Override default model
        temperature: Override default temperature
        as_json: Return response as parsed JSON
        schema: JSON schema for structured output
        **kwargs: Additional LLM parameters
        
    Returns:
        str: The answer (or dict if as_json=True)
    
    Examples:
        >>> ask("What is 2+2?")
        '4'
        
        >>> ask("List 3 benefits of Python", format="bullet")
        '• Easy to learn\\n• Large ecosystem\\n• Versatile'
        
        >>> ask("Generate user data", as_json=True, schema={"name": "str", "age": "int"})
        {"name": "John", "age": 30}
    """
    # Build system message based on modifiers
    system_parts = ["You are a helpful AI assistant."]
    
    if concise:
        system_parts.append("Be concise and direct. Give brief answers.")
        if temperature is None:
            temperature = 0.3
    elif detailed:
        system_parts.append("Provide detailed, comprehensive answers with explanations.")
        
    if creative:
        system_parts.append("Be creative and think outside the box.")
        if temperature is None:
            temperature = 1.0
            
    if format:
        format_instructions = {
            "bullet": "Format your response as bullet points using •",
            "numbered": "Format your response as a numbered list",
            "markdown": "Format your response using markdown",
            "code": "Respond with code only, no explanations",
            "json": "Respond with valid JSON only",
        }
        if format.lower() in format_instructions:
            system_parts.append(format_instructions[format.lower()])
            
    system = " ".join(system_parts)
    
    # Build the prompt
    prompt = question
    if context:
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
    
    # Get LLM instance
    llm_kwargs = {}
    if model:
        llm_kwargs["model"] = model
        
    llm = get_llm(**llm_kwargs)
    
    # Handle JSON output
    if as_json or schema:
        return llm.json(prompt, schema=schema, system=system, **kwargs)
    
    # Get response
    response = llm.complete(
        prompt,
        system=system,
        temperature=temperature,
        **kwargs
    )
    
    return response.content


# Convenient aliases
def question(q: str, **kwargs) -> str:
    """Alias for ask()."""
    return ask(q, **kwargs)


def prompt(p: str, **kwargs) -> str:
    """Alias for ask()."""
    return ask(p, **kwargs)
