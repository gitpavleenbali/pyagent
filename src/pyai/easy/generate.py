"""
generate() - Generate any content with AI

Create text, code, documents, and more.

Examples:
    >>> from pyai import generate
    >>> generate("A professional bio for a software engineer")
    'John is a passionate software engineer...'

    >>> generate("Python function to calculate factorial", type="code")
    'def factorial(n):\\n    ...'
"""

from pyai.easy.llm_interface import get_llm


def generate(
    what: str,
    *,
    type: str = "text",  # "text", "code", "email", "article", "docs", etc.
    style: str = None,
    tone: str = None,  # "formal", "casual", "professional", "friendly"
    length: str = "medium",  # "short", "medium", "long"
    language: str = "python",  # For code generation
    context: str = None,
    model: str = None,
    creative: bool = False,
    **kwargs,
) -> str:
    """
    Generate any type of content.

    Args:
        what: Description of what to generate
        type: Content type - "text", "code", "email", "article", "docs", "story"
        style: Writing style
        tone: Tone of voice
        length: Output length ("short", "medium", "long")
        language: Programming language (for code)
        context: Additional context
        model: Override default model
        creative: Enable more creative output
        **kwargs: Additional parameters

    Returns:
        str: The generated content

    Examples:
        >>> generate("Welcome email for new users", type="email")
        'Subject: Welcome to Our Platform!...'

        >>> generate("REST API for todo app", type="code", language="python")
        'from flask import Flask...'

        >>> generate("Blog post about AI trends", type="article", length="long")
        '# The Future of AI...'
    """
    llm_kwargs = {"model": model} if model else {}
    llm = get_llm(**llm_kwargs)

    # Build system prompt based on type
    type_systems = {
        "text": "You are a skilled writer. Generate clear, engaging text.",
        "code": f"You are an expert {language} programmer. Generate clean, well-documented code. Return code only.",
        "email": "You are a professional email writer. Create clear, effective emails.",
        "article": "You are a professional content writer. Write engaging, well-structured articles.",
        "docs": "You are a technical documentation writer. Create clear, comprehensive documentation.",
        "story": "You are a creative storyteller. Write engaging, imaginative stories.",
        "marketing": "You are a marketing copywriter. Create compelling, persuasive content.",
        "social": "You are a social media content creator. Create engaging, shareable content.",
    }

    system_parts = [type_systems.get(type, type_systems["text"])]

    # Add tone
    if tone:
        tone_prompts = {
            "formal": "Use formal, professional language.",
            "casual": "Use casual, conversational language.",
            "professional": "Maintain a professional yet approachable tone.",
            "friendly": "Be warm and friendly.",
            "technical": "Use precise technical language.",
            "persuasive": "Be compelling and persuasive.",
        }
        if tone in tone_prompts:
            system_parts.append(tone_prompts[tone])

    # Add length guidance
    length_prompts = {
        "short": "Keep it brief and concise.",
        "medium": "Provide a moderate amount of content.",
        "long": "Be comprehensive and detailed.",
    }
    system_parts.append(length_prompts.get(length, length_prompts["medium"]))

    if style:
        system_parts.append(f"Write in {style} style.")

    system = " ".join(system_parts)

    # Build prompt
    prompt = f"Generate: {what}"

    if context:
        prompt = f"Context: {context}\n\n{prompt}"

    # Temperature based on type
    temperature = kwargs.pop("temperature", None)
    if temperature is None:
        if creative or type == "story":
            temperature = 1.0
        elif type == "code":
            temperature = 0.2
        else:
            temperature = 0.7

    response = llm.complete(prompt, system=system, temperature=temperature, **kwargs)

    # Clean up code output
    if type == "code":
        content = response.content
        # Remove markdown code blocks if present
        if "```" in content:
            lines = content.split("\n")
            code_lines = []
            in_code = False
            for line in lines:
                if line.startswith("```"):
                    in_code = not in_code
                elif in_code:
                    code_lines.append(line)
            if code_lines:
                return "\n".join(code_lines)
        return content

    return response.content


# Convenience functions for specific types
def write(what: str, **kwargs) -> str:
    """Generate text content."""
    return generate(what, type="text", **kwargs)


def write_code(what: str, language: str = "python", **kwargs) -> str:
    """Generate code."""
    return generate(what, type="code", language=language, **kwargs)


def write_email(what: str, **kwargs) -> str:
    """Generate an email."""
    return generate(what, type="email", **kwargs)


def write_article(what: str, **kwargs) -> str:
    """Generate an article."""
    return generate(what, type="article", **kwargs)
