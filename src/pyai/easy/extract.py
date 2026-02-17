"""
extract() - Extract structured data from text

Turn unstructured text into structured data.

Examples:
    >>> from pyai import extract
    >>> extract("John is 30 years old and lives in NYC", ["name", "age", "city"])
    {"name": "John", "age": 30, "city": "NYC"}

    >>> extract(email_text, "Extract all action items")
    ["Schedule meeting", "Send report", "Review document"]
"""

import json
from typing import Any, Dict, List, Union

from pydantic import BaseModel

from pyai.easy.llm_interface import get_llm


def extract(
    text: str,
    schema: Union[List[str], Dict[str, Any], type, str],
    *,
    examples: List[Dict] = None,
    model: str = None,
    **kwargs,
) -> Union[Dict[str, Any], List[Any]]:
    """
    Extract structured data from unstructured text.

    Args:
        text: The text to extract from
        schema: What to extract - can be:
            - List of field names: ["name", "age", "city"]
            - Dict schema: {"name": "string", "age": "integer"}
            - Pydantic model class
            - Natural language: "Extract all email addresses"
        examples: Optional few-shot examples
        model: Override default model
        **kwargs: Additional parameters

    Returns:
        Extracted data matching the schema

    Examples:
        >>> text = "Contact John at john@email.com, he's 30"
        >>> extract(text, ["name", "email", "age"])
        {"name": "John", "email": "john@email.com", "age": 30}

        >>> extract(text, "Find all email addresses")
        ["john@email.com"]

        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>> extract(text, Person)
        {"name": "John", "age": 30}
    """
    llm_kwargs = {"model": model} if model else {}
    llm = get_llm(**llm_kwargs)

    # Handle different schema types
    if isinstance(schema, str):
        # Natural language extraction
        prompt = f"""From the following text, {schema}

Text:
{text}

Return the extracted information as JSON."""

        return llm.json(
            prompt,
            system="You are a data extraction expert. Extract the requested information accurately.",
            temperature=0.1,
        )

    elif isinstance(schema, list):
        # List of field names
        fields = ", ".join(schema)
        json_schema = {field: "extracted value" for field in schema}

        prompt = f"""Extract the following fields from the text: {fields}

Text:
{text}

Return as JSON with these exact field names."""

        if examples:
            examples_str = "\n".join(f"Example: {json.dumps(ex)}" for ex in examples)
            prompt += f"\n\nExamples:\n{examples_str}"

        return llm.json(
            prompt,
            system="You are a data extraction expert. Extract values accurately. Return null for missing fields.",
            schema=json_schema,
            temperature=0.1,
        )

    elif isinstance(schema, dict):
        # Dict schema - convert Python types to string names for JSON serialization
        def type_to_string(v):
            if isinstance(v, type):
                type_map = {
                    str: "string",
                    int: "integer",
                    float: "number",
                    bool: "boolean",
                    list: "array",
                    dict: "object",
                }
                return type_map.get(v, v.__name__)
            return v

        json_friendly_schema = {k: type_to_string(v) for k, v in schema.items()}

        prompt = f"""Extract information according to this schema:
{json.dumps(json_friendly_schema, indent=2)}

Text:
{text}

Return as JSON matching the schema exactly."""

        return llm.json(
            prompt,
            system="You are a data extraction expert. Match the schema types precisely.",
            schema=json_friendly_schema,
            temperature=0.1,
        )

    elif isinstance(schema, type) and issubclass(schema, BaseModel):
        # Pydantic model
        pydantic_schema = schema.model_json_schema()

        prompt = f"""Extract information according to this schema:
{json.dumps(pydantic_schema, indent=2)}

Text:
{text}

Return as JSON matching the schema fields."""

        result = llm.json(
            prompt,
            system="You are a data extraction expert. Match the schema types precisely.",
            temperature=0.1,
        )

        # Optionally validate with Pydantic
        try:
            validated = schema.model_validate(result)
            return validated.model_dump()
        except Exception:
            return result

    else:
        raise ValueError(f"Unsupported schema type: {type(schema)}")


def extract_list(text: str, what: str, **kwargs) -> List[str]:
    """
    Extract a list of items from text.

    Args:
        text: The text to extract from
        what: What to extract (e.g., "email addresses", "names", "dates")

    Returns:
        List of extracted items
    """
    result = extract(text, f"Extract all {what}", **kwargs)

    if isinstance(result, list):
        return result
    elif isinstance(result, dict):
        # Try to find the list in the dict
        for key, value in result.items():
            if isinstance(value, list):
                return value

    return [result] if result else []


def extract_entities(text: str, **kwargs) -> Dict[str, List[str]]:
    """
    Extract named entities from text.

    Returns:
        Dict with entity types as keys and lists of entities as values
    """
    return extract(
        text,
        {
            "people": ["list of person names"],
            "organizations": ["list of company/org names"],
            "locations": ["list of places"],
            "dates": ["list of dates"],
            "amounts": ["list of monetary amounts or quantities"],
        },
        **kwargs,
    )
