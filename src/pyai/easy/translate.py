"""
translate() - Translate text between languages

Examples:
    >>> from pyai import translate
    >>> translate("Hello, how are you?", to="spanish")
    'Hola, ¿cómo estás?'

    >>> translate("Bonjour", to="english")
    'Hello'
"""

from pyai.easy.llm_interface import get_llm

# Language aliases for convenience
LANGUAGE_ALIASES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "nl": "Dutch",
    "sv": "Swedish",
    "pl": "Polish",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
}


def translate(
    text: str,
    *,
    to: str,
    from_lang: str = None,
    formal: bool = None,
    preserve_formatting: bool = True,
    model: str = None,
    **kwargs,
) -> str:
    """
    Translate text to another language.

    Args:
        text: Text to translate
        to: Target language (name or code like "es", "french")
        from_lang: Source language (auto-detected if not specified)
        formal: Use formal (True) or informal (False) tone
        preserve_formatting: Keep original formatting
        model: Override default model
        **kwargs: Additional parameters

    Returns:
        str: Translated text

    Examples:
        >>> translate("Good morning!", to="spanish")
        '¡Buenos días!'

        >>> translate("How are you?", to="japanese", formal=True)
        'お元気ですか？'

        >>> translate("Ciao", to="en")
        'Hello'
    """
    llm_kwargs = {"model": model} if model else {}
    llm = get_llm(**llm_kwargs)

    # Resolve language aliases
    target_lang = LANGUAGE_ALIASES.get(to.lower(), to.title())
    source_lang = (
        LANGUAGE_ALIASES.get(from_lang.lower(), from_lang.title()) if from_lang else "auto-detect"
    )

    # Build system prompt
    system_parts = [
        f"You are a professional translator specializing in {target_lang}.",
        "Translate accurately while preserving the original meaning and tone.",
        "Return only the translation, no explanations.",
    ]

    if formal is True:
        system_parts.append("Use formal language and polite forms.")
    elif formal is False:
        system_parts.append("Use casual, informal language.")

    if preserve_formatting:
        system_parts.append("Preserve the original formatting, line breaks, and structure.")

    system = " ".join(system_parts)

    # Build prompt
    if source_lang == "auto-detect":
        prompt = f"Translate the following to {target_lang}:\n\n{text}"
    else:
        prompt = f"Translate the following from {source_lang} to {target_lang}:\n\n{text}"

    response = llm.complete(prompt, system=system, temperature=0.3, **kwargs)

    return response.content.strip()


# Convenient shortcuts for common translations
def to_english(text: str, **kwargs) -> str:
    """Translate to English."""
    return translate(text, to="english", **kwargs)


def to_spanish(text: str, **kwargs) -> str:
    """Translate to Spanish."""
    return translate(text, to="spanish", **kwargs)


def to_french(text: str, **kwargs) -> str:
    """Translate to French."""
    return translate(text, to="french", **kwargs)


def to_german(text: str, **kwargs) -> str:
    """Translate to German."""
    return translate(text, to="german", **kwargs)


def to_chinese(text: str, **kwargs) -> str:
    """Translate to Chinese."""
    return translate(text, to="chinese", **kwargs)


def to_japanese(text: str, **kwargs) -> str:
    """Translate to Japanese."""
    return translate(text, to="japanese", **kwargs)
