"""
code Module - AI-Powered Code Operations

Write, review, debug, and explain code.

Examples:
    >>> from pyai import code
    >>>
    >>> # Generate code
    >>> code.write("function to calculate fibonacci")
    'def fibonacci(n):\\n    ...'
    >>>
    >>> # Review code
    >>> code.review(my_function)
    'This code looks good, but consider...'
    >>>
    >>> # Debug errors
    >>> code.debug(error_traceback)
    'The error is caused by...'
    >>>
    >>> # Explain code
    >>> code.explain(complex_function)
    'This function does...'
"""

from dataclasses import dataclass, field
from typing import List

from pyai.easy.llm_interface import get_llm


@dataclass
class CodeResult:
    """Result of a code operation."""

    code: str
    language: str
    explanation: str = ""

    def __str__(self) -> str:
        return self.code

    def __repr__(self) -> str:
        lines = self.code.count("\n") + 1
        return f"CodeResult({self.language}, {lines} lines)"


@dataclass
class CodeReview:
    """Result of a code review."""

    summary: str
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    score: int = 0  # 1-10

    def __str__(self) -> str:
        return self.summary


def write(
    description: str,
    *,
    language: str = "python",
    style: str = None,  # "clean", "documented", "minimal"
    include_tests: bool = False,
    model: str = None,
    **kwargs,
) -> str:
    """
    Generate code from a description.

    Args:
        description: What the code should do
        language: Programming language
        style: Code style preference
        include_tests: Include unit tests
        model: Override default model
        **kwargs: Additional parameters

    Returns:
        str: The generated code

    Examples:
        >>> code.write("function to calculate factorial")
        'def factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n-1)'

        >>> code.write("REST API endpoint", language="javascript")
    """
    llm_kwargs = {"model": model} if model else {}
    llm = get_llm(**llm_kwargs)

    style_prompts = {
        "clean": "Write clean, readable code following best practices.",
        "documented": "Include comprehensive docstrings and comments.",
        "minimal": "Write minimal, concise code without extra comments.",
        "production": "Write production-ready code with error handling and logging.",
    }

    system = f"You are an expert {language} programmer. {style_prompts.get(style, style_prompts['clean'])} Return only code, no explanations."

    prompt = f"Write {language} code for: {description}"

    if include_tests:
        prompt += "\n\nAlso include unit tests."

    response = llm.complete(prompt, system=system, temperature=0.2, **kwargs)

    # Clean up response
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


def review(
    code_str: str,
    *,
    focus: str = None,  # "security", "performance", "readability"
    model: str = None,
    **kwargs,
) -> CodeReview:
    """
    Review code and get feedback.

    Args:
        code_str: The code to review
        focus: Specific area to focus on
        model: Override default model
        **kwargs: Additional parameters

    Returns:
        CodeReview: Review with issues and suggestions

    Examples:
        >>> code.review(my_function)
        CodeReview(score=8, issues=['Variable naming could be clearer'])
    """
    llm_kwargs = {"model": model} if model else {}
    llm = get_llm(**llm_kwargs)

    focus_prompt = ""
    if focus:
        focus_prompt = f"\n\nFocus particularly on: {focus}"

    prompt = f"""Review this code and provide feedback:

```
{code_str}
```
{focus_prompt}

Provide:
1. A brief summary
2. List of issues (if any)
3. Suggestions for improvement
4. Overall score (1-10)"""

    result = llm.json(
        prompt,
        system="You are a senior code reviewer. Be constructive and thorough.",
        schema={
            "summary": "string",
            "issues": ["list of issues"],
            "suggestions": ["list of suggestions"],
            "score": "integer 1-10",
        },
        temperature=0.3,
    )

    return CodeReview(
        summary=result.get("summary", ""),
        issues=result.get("issues", []),
        suggestions=result.get("suggestions", []),
        score=result.get("score", 5),
    )


def debug(error: str, *, code_context: str = None, model: str = None, **kwargs) -> str:
    """
    Debug an error and get a solution.

    Args:
        error: The error message or traceback
        code_context: Optional code that caused the error
        model: Override default model
        **kwargs: Additional parameters

    Returns:
        str: Explanation and solution

    Examples:
        >>> code.debug("TypeError: cannot unpack non-iterable NoneType object")
        'This error occurs when...'
    """
    llm_kwargs = {"model": model} if model else {}
    llm = get_llm(**llm_kwargs)

    prompt = f"Debug this error:\n\n{error}"

    if code_context:
        prompt += f"\n\nCode context:\n```\n{code_context}\n```"

    prompt += "\n\nExplain the cause and provide a solution."

    response = llm.complete(
        prompt,
        system="You are a debugging expert. Explain errors clearly and provide working solutions.",
        temperature=0.3,
        **kwargs,
    )

    return response.content


def explain(
    code_str: str,
    *,
    detail_level: str = "medium",  # "brief", "medium", "detailed"
    for_beginner: bool = False,
    model: str = None,
    **kwargs,
) -> str:
    """
    Explain what code does.

    Args:
        code_str: The code to explain
        detail_level: How detailed to be
        for_beginner: Explain for beginners
        model: Override default model
        **kwargs: Additional parameters

    Returns:
        str: Explanation of the code

    Examples:
        >>> code.explain(complex_function)
        'This function takes an input and...'
    """
    llm_kwargs = {"model": model} if model else {}
    llm = get_llm(**llm_kwargs)

    detail_prompts = {
        "brief": "Give a brief, one-paragraph explanation.",
        "medium": "Explain the code in moderate detail.",
        "detailed": "Provide a thorough, line-by-line explanation.",
    }

    audience = "Explain as if to a beginner with simple terms." if for_beginner else ""

    prompt = f"""Explain this code:

```
{code_str}
```

{detail_prompts.get(detail_level, detail_prompts["medium"])}
{audience}"""

    response = llm.complete(
        prompt,
        system="You are a patient teacher explaining code. Be clear and helpful.",
        temperature=0.4,
        **kwargs,
    )

    return response.content


def refactor(
    code_str: str,
    *,
    goal: str = None,  # "readability", "performance", "modern"
    model: str = None,
    **kwargs,
) -> str:
    """
    Refactor code to improve it.

    Args:
        code_str: The code to refactor
        goal: Refactoring goal
        model: Override default model
        **kwargs: Additional parameters

    Returns:
        str: Refactored code
    """
    llm_kwargs = {"model": model} if model else {}
    llm = get_llm(**llm_kwargs)

    goal_prompts = {
        "readability": "Improve readability and clarity.",
        "performance": "Optimize for better performance.",
        "modern": "Update to modern patterns and syntax.",
        "clean": "Apply clean code principles.",
    }

    goal_text = goal_prompts.get(goal, "Improve the code quality.")

    response = llm.complete(
        f"Refactor this code. {goal_text} Return only the improved code:\n\n{code_str}",
        system="You are a refactoring expert. Return clean, improved code only.",
        temperature=0.2,
        **kwargs,
    )

    # Clean up response
    content = response.content
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


def convert(code_str: str, *, from_lang: str, to_lang: str, model: str = None, **kwargs) -> str:
    """
    Convert code from one language to another.

    Args:
        code_str: Source code
        from_lang: Source language
        to_lang: Target language
        model: Override default model
        **kwargs: Additional parameters

    Returns:
        str: Converted code
    """
    llm_kwargs = {"model": model} if model else {}
    llm = get_llm(**llm_kwargs)

    response = llm.complete(
        f"Convert this {from_lang} code to {to_lang}:\n\n{code_str}",
        system=f"You are an expert in both {from_lang} and {to_lang}. Convert code idiomatically. Return only code.",
        temperature=0.2,
        **kwargs,
    )

    # Clean up response
    content = response.content
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


# Aliases
generate = write
fix = debug
