# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Token Counter

Count tokens for various model providers.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# Try to import tiktoken for OpenAI models
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


@dataclass
class TokenCount:
    """Result of token counting.
    
    Attributes:
        input_tokens: Number of tokens in input/prompt
        output_tokens: Number of tokens in output/completion
        total_tokens: Total tokens (input + output)
        model: Model used for encoding
        method: Method used for counting
    """
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str = "unknown"
    method: str = "estimate"
    
    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


class TokenCounter:
    """Count tokens for various models.
    
    Supports:
    - OpenAI models (via tiktoken)
    - Anthropic models (via character estimation)
    - Azure OpenAI models
    
    Example:
        counter = TokenCounter("gpt-4")
        count = counter.count("Hello, how are you?")
        print(f"Tokens: {count.input_tokens}")
    """
    
    # Model to encoding mapping
    MODEL_ENCODINGS = {
        # GPT-4 models
        "gpt-4": "cl100k_base",
        "gpt-4-0314": "cl100k_base",
        "gpt-4-0613": "cl100k_base",
        "gpt-4-32k": "cl100k_base",
        "gpt-4-32k-0314": "cl100k_base",
        "gpt-4-32k-0613": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4-turbo-preview": "cl100k_base",
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        # GPT-3.5 models
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-3.5-turbo-0301": "cl100k_base",
        "gpt-3.5-turbo-0613": "cl100k_base",
        "gpt-3.5-turbo-16k": "cl100k_base",
        "gpt-3.5-turbo-instruct": "cl100k_base",
        # Embedding models
        "text-embedding-ada-002": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
        # Claude models (character-based estimation)
        "claude-3-opus": "char_estimate",
        "claude-3-sonnet": "char_estimate",
        "claude-3-haiku": "char_estimate",
        "claude-3.5-sonnet": "char_estimate",
        "claude-2": "char_estimate",
        "claude-2.1": "char_estimate",
        # Gemini models
        "gemini-pro": "char_estimate",
        "gemini-ultra": "char_estimate",
        "gemini-1.5-pro": "char_estimate",
        "gemini-1.5-flash": "char_estimate",
    }
    
    # Average characters per token for estimation
    CHARS_PER_TOKEN = 4
    
    def __init__(
        self,
        model: str = "gpt-4",
        encoding: Optional[str] = None
    ):
        """Initialize token counter.
        
        Args:
            model: Model name for encoding selection
            encoding: Override encoding name
        """
        self.model = model
        self._encoding_name = encoding or self._get_encoding_name(model)
        self._encoder = None
        
        # Initialize tiktoken encoder if available
        if TIKTOKEN_AVAILABLE and self._encoding_name not in ["char_estimate"]:
            try:
                self._encoder = tiktoken.get_encoding(self._encoding_name)
            except Exception:
                # Fall back to estimation
                self._encoding_name = "char_estimate"
    
    def _get_encoding_name(self, model: str) -> str:
        """Get encoding name for a model."""
        # Check exact match
        if model in self.MODEL_ENCODINGS:
            return self.MODEL_ENCODINGS[model]
        
        # Check prefix match
        for model_prefix, encoding in self.MODEL_ENCODINGS.items():
            if model.startswith(model_prefix):
                return encoding
        
        # Default to character estimation
        return "char_estimate"
    
    def count(
        self,
        text: Union[str, List[Dict[str, Any]]],
        is_output: bool = False
    ) -> TokenCount:
        """Count tokens in text or messages.
        
        Args:
            text: Text string or list of message dicts
            is_output: Whether this is output/completion text
            
        Returns:
            TokenCount with token counts
        """
        # Convert messages to string if needed
        if isinstance(text, list):
            text = self._messages_to_text(text)
        
        # Count tokens
        if self._encoder:
            tokens = len(self._encoder.encode(text))
            method = "tiktoken"
        else:
            tokens = self._estimate_tokens(text)
            method = "char_estimate"
        
        # Create result
        if is_output:
            return TokenCount(
                output_tokens=tokens,
                total_tokens=tokens,
                model=self.model,
                method=method
            )
        else:
            return TokenCount(
                input_tokens=tokens,
                total_tokens=tokens,
                model=self.model,
                method=method
            )
    
    def count_messages(
        self,
        messages: List[Dict[str, Any]],
        completion: Optional[str] = None
    ) -> TokenCount:
        """Count tokens in a conversation.
        
        Args:
            messages: List of message dicts with role/content
            completion: Optional assistant completion
            
        Returns:
            TokenCount with input and output tokens
        """
        # Count input tokens (messages)
        input_text = self._messages_to_text(messages)
        
        if self._encoder:
            input_tokens = len(self._encoder.encode(input_text))
            # Add per-message overhead (4 tokens per message for GPT models)
            input_tokens += len(messages) * 4
            method = "tiktoken"
        else:
            input_tokens = self._estimate_tokens(input_text)
            method = "char_estimate"
        
        # Count output tokens
        output_tokens = 0
        if completion:
            if self._encoder:
                output_tokens = len(self._encoder.encode(completion))
            else:
                output_tokens = self._estimate_tokens(completion)
        
        return TokenCount(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            model=self.model,
            method=method
        )
    
    def _messages_to_text(self, messages: List[Dict[str, Any]]) -> str:
        """Convert message list to text."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle content that's a list (e.g., vision messages)
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                content = " ".join(text_parts)
            
            parts.append(f"{role}: {content}")
        
        return "\n".join(parts)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate tokens using character count."""
        # Use 4 characters per token as a rough estimate
        # This is a reasonable approximation for most languages
        return max(1, len(text) // self.CHARS_PER_TOKEN)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        if self._encoder:
            return self._encoder.encode(text)
        else:
            # Return pseudo-tokens based on character estimation
            return list(range(self._estimate_tokens(text)))
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text
        """
        if self._encoder:
            return self._encoder.decode(tokens)
        else:
            # Cannot decode without actual encoder
            return f"[{len(tokens)} tokens]"


def count_tokens(
    text: Union[str, List[Dict[str, Any]]],
    model: str = "gpt-4"
) -> int:
    """Count tokens in text.
    
    Simple utility function for quick token counting.
    
    Args:
        text: Text string or messages list
        model: Model for encoding
        
    Returns:
        Number of tokens
        
    Example:
        tokens = count_tokens("Hello, world!", model="gpt-4")
    """
    counter = TokenCounter(model)
    result = counter.count(text)
    return result.input_tokens


def estimate_tokens(
    text: str,
    chars_per_token: float = 4.0
) -> int:
    """Estimate tokens using character count.
    
    Fast estimation without requiring tiktoken.
    
    Args:
        text: Text to estimate
        chars_per_token: Characters per token ratio
        
    Returns:
        Estimated token count
    """
    return max(1, int(len(text) / chars_per_token))
