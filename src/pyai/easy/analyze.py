"""
analyze Module - Data Analysis in One Line

Analyze data, get insights, visualize patterns.

Examples:
    >>> from pyai import analyze
    >>>
    >>> # Analyze any data
    >>> insights = analyze.data(sales_data)
    >>> print(insights.summary)
    >>>
    >>> # Get insights from text
    >>> insights = analyze.text(customer_reviews)
    >>>
    >>> # Quick stats
    >>> stats = analyze.describe(numbers)
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from pyai.easy.llm_interface import get_llm


@dataclass
class AnalysisResult:
    """Result of a data analysis."""

    data_type: str
    summary: str
    insights: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.summary

    @property
    def bullets(self) -> str:
        return "\n".join(f"• {i}" for i in self.insights)


def data(
    data: Union[str, List, Dict, Any],
    *,
    focus: str = None,
    detailed: bool = False,
    model: str = None,
    **kwargs,
) -> AnalysisResult:
    """
    Analyze any data and get insights.

    Args:
        data: Data to analyze (JSON, list, dict, CSV string, etc.)
        focus: Specific aspect to focus on
        detailed: Get detailed analysis
        model: Override default model
        **kwargs: Additional parameters

    Returns:
        AnalysisResult: Analysis with insights and recommendations

    Examples:
        >>> analyze.data([1, 2, 3, 4, 5, 100])
        AnalysisResult with outlier detection

        >>> analyze.data(sales_dict, focus="trends")
    """
    llm_kwargs = {"model": model} if model else {}
    llm = get_llm(**llm_kwargs)

    # Convert data to string representation
    if isinstance(data, (list, dict)):
        data_str = json.dumps(data, indent=2, default=str)
    else:
        data_str = str(data)

    # Determine data type
    if isinstance(data, list):
        data_type = "list/array"
    elif isinstance(data, dict):
        data_type = "dictionary/object"
    else:
        data_type = "text/other"

    depth = "comprehensive and detailed" if detailed else "concise but insightful"

    prompt = f"""Analyze the following {data_type} data and provide {depth} analysis:

{data_str}

Provide:
1. A clear summary of what this data represents
2. Key insights and patterns
3. Statistical observations if applicable
4. Actionable recommendations
{"5. Focus particularly on: " + focus if focus else ""}
"""

    result = llm.json(
        prompt,
        system="You are a data analyst. Provide clear, actionable analysis.",
        schema={
            "summary": "string",
            "insights": ["list of insight strings"],
            "statistics": {"key": "value pairs"},
            "recommendations": ["list of recommendation strings"],
        },
        temperature=0.4,
    )

    return AnalysisResult(
        data_type=data_type,
        summary=result.get("summary", ""),
        insights=result.get("insights", []),
        statistics=result.get("statistics", {}),
        recommendations=result.get("recommendations", []),
    )


def text(
    text: str,
    *,
    analyze_for: str = None,  # "sentiment", "topics", "entities", etc.
    model: str = None,
    **kwargs,
) -> AnalysisResult:
    """
    Analyze text content.

    Args:
        text: Text to analyze
        analyze_for: Specific analysis type
        model: Override default model
        **kwargs: Additional parameters

    Returns:
        AnalysisResult: Text analysis results

    Examples:
        >>> analyze.text(customer_reviews, analyze_for="sentiment")
        >>> analyze.text(article, analyze_for="topics")
    """
    llm_kwargs = {"model": model} if model else {}
    llm = get_llm(**llm_kwargs)

    analysis_type = analyze_for or "general"

    prompts = {
        "sentiment": "Analyze the sentiment (positive/negative/neutral) and emotional tone.",
        "topics": "Identify the main topics, themes, and subject matter.",
        "entities": "Extract and categorize named entities (people, places, organizations).",
        "readability": "Assess readability, complexity, and writing quality.",
        "general": "Provide a comprehensive text analysis including tone, topics, and key takeaways.",
    }

    prompt = f"""{prompts.get(analysis_type, prompts["general"])}

Text:
{text}

Provide structured analysis with summary, insights, and any relevant statistics."""

    result = llm.json(
        prompt,
        system="You are a text analyst. Provide thorough, accurate analysis.",
        temperature=0.3,
    )

    return AnalysisResult(
        data_type=f"text ({analysis_type})",
        summary=result.get("summary", ""),
        insights=result.get("insights", []),
        statistics=result.get("statistics", {}),
        recommendations=result.get("recommendations", []),
    )


def sentiment(text: str, **kwargs) -> Dict[str, Any]:
    """
    Quick sentiment analysis.

    Args:
        text: Text to analyze

    Returns:
        Dict with sentiment and confidence

    Examples:
        >>> analyze.sentiment("I love this product!")
        {"sentiment": "positive", "confidence": 0.95, "emotions": ["joy"]}
    """
    llm = get_llm(**kwargs)

    return llm.json(
        f"Analyze the sentiment of this text:\n\n{text}",
        system="You are a sentiment analyzer. Return sentiment, confidence (0-1), and detected emotions.",
        schema={
            "sentiment": "positive/negative/neutral",
            "confidence": "float 0-1",
            "emotions": ["list of emotions"],
        },
        temperature=0.2,
    )


def describe(data: Union[List, Dict], **kwargs) -> str:
    """
    Get a quick description of data.

    Args:
        data: Data to describe

    Returns:
        str: Human-readable description
    """
    llm = get_llm(**kwargs)

    response = llm.complete(
        f"Briefly describe this data in 2-3 sentences:\n{json.dumps(data, default=str)}",
        system="You are a data describer. Be concise and informative.",
        temperature=0.3,
    )

    return response.content


def compare(data1: Any, data2: Any, **kwargs) -> Dict[str, Any]:
    """
    Compare two pieces of data.

    Args:
        data1: First data
        data2: Second data

    Returns:
        Dict with similarities, differences, and conclusion
    """
    llm = get_llm(**kwargs)

    return llm.json(
        f"""Compare these two items:

Item 1: {json.dumps(data1, default=str)}

Item 2: {json.dumps(data2, default=str)}

Identify similarities, differences, and provide a conclusion.""",
        system="You are a comparison analyst. Be thorough and objective.",
        schema={"similarities": ["list"], "differences": ["list"], "conclusion": "string"},
    )
