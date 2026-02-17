"""
research() - Deep research on any topic in one line

Performs comprehensive research with source gathering, synthesis, and summarization.

Examples:
    >>> from pyai import research
    >>> result = research("quantum computing breakthroughs 2024")
    >>> print(result.summary)
    >>> print(result.sources)

    >>> # Quick research
    >>> summary = research("benefits of meditation", quick=True)

    >>> # Get structured insights
    >>> insights = research("AI trends", as_insights=True)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from pyai.easy.llm_interface import get_llm


@dataclass
class ResearchResult:
    """Result of a research operation."""

    query: str
    summary: str
    key_points: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    sources: List[Dict[str, str]] = field(default_factory=list)
    confidence: float = 0.8

    def __str__(self) -> str:
        return self.summary

    def __repr__(self) -> str:
        return f"ResearchResult(query='{self.query[:30]}...', points={len(self.key_points)})"

    @property
    def bullets(self) -> str:
        """Get key points as bullet list."""
        return "\n".join(f"• {point}" for point in self.key_points)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "summary": self.summary,
            "key_points": self.key_points,
            "insights": self.insights,
            "sources": self.sources,
            "confidence": self.confidence,
        }


def research(
    topic: str,
    *,
    depth: str = "standard",  # "quick", "standard", "deep"
    focus: str = None,
    num_points: int = 5,
    include_sources: bool = True,
    quick: bool = False,
    as_insights: bool = False,
    model: str = None,
    **kwargs,
) -> Union[ResearchResult, str, List[str]]:
    """
    Research any topic comprehensively.

    Args:
        topic: The topic to research
        depth: Research depth ("quick", "standard", "deep")
        focus: Specific aspect to focus on
        num_points: Number of key points to extract
        include_sources: Whether to include source references
        quick: Return just a summary string (shortcut for depth="quick")
        as_insights: Return list of insights only
        model: Override default model
        **kwargs: Additional parameters

    Returns:
        ResearchResult: Comprehensive research result (default)
        str: Just the summary (if quick=True)
        List[str]: List of insights (if as_insights=True)

    Examples:
        >>> result = research("machine learning in healthcare")
        >>> print(result.summary)
        >>> print(result.key_points)

        >>> quick_answer = research("what causes aurora borealis", quick=True)

        >>> insights = research("future of remote work", as_insights=True)
    """
    if quick:
        depth = "quick"

    # Get LLM
    llm_kwargs = {"model": model} if model else {}
    llm = get_llm(**llm_kwargs)

    # Build research prompt based on depth
    if depth == "quick":
        system = "You are a research assistant. Provide concise, accurate summaries."
        prompt = f"Briefly summarize the key facts about: {topic}"

        if focus:
            prompt += f"\nFocus on: {focus}"

        response = llm.complete(prompt, system=system, temperature=0.3)

        if quick:
            return response.content
    else:
        # Standard or deep research
        depth_instructions = {
            "standard": "Provide comprehensive research with key points and insights.",
            "deep": "Provide exhaustive research with detailed analysis, multiple perspectives, and critical evaluation.",
        }

        system = f"""You are an expert research analyst. {depth_instructions.get(depth, depth_instructions["standard"])}

Your research should be:
- Accurate and factual
- Well-organized
- Balanced across different perspectives
- Include actionable insights"""

        prompt = f"""Research the following topic comprehensively: {topic}

Provide:
1. A clear summary (2-3 paragraphs)
2. {num_points} key points
3. Key insights or implications
"""
        if focus:
            prompt += f"\nPay special attention to: {focus}"

        if include_sources:
            prompt += "\n4. Note that these are synthesized insights from general knowledge."

        # Use JSON for structured output
        response = llm.json(
            prompt,
            system=system,
            schema={"summary": "string", "key_points": ["string"], "insights": ["string"]},
            temperature=0.4,
        )

    # Build result
    if isinstance(response, dict):
        result = ResearchResult(
            query=topic,
            summary=response.get("summary", ""),
            key_points=response.get("key_points", [])[:num_points],
            insights=response.get("insights", []),
            sources=[{"type": "synthesized", "note": "Based on training data knowledge"}]
            if include_sources
            else [],
        )
    else:
        result = ResearchResult(
            query=topic,
            summary=response.content if hasattr(response, "content") else str(response),
            key_points=[],
            insights=[],
        )

    if as_insights:
        return result.insights

    return result


def investigate(topic: str, **kwargs) -> ResearchResult:
    """Alias for research() with deep analysis."""
    kwargs.setdefault("depth", "deep")
    return research(topic, **kwargs)


def quick_research(topic: str, **kwargs) -> str:
    """Quick research returning just a summary."""
    return research(topic, quick=True, **kwargs)
