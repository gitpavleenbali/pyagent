# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Filter Registry

Middleware/filter system for the Kernel.
Filters can intercept and modify prompts, function calls, and results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar


class FilterType(Enum):
    """Types of filters."""
    PROMPT = "prompt"           # Before/after prompt rendering
    FUNCTION = "function"       # Before/after function invocation
    AUTO_FUNCTION = "auto"      # For auto function calling
    RESULT = "result"           # Post-processing results


@dataclass
class FilterContext:
    """Context passed to filters.
    
    Attributes:
        kernel: Reference to the kernel
        function_name: Name of function being invoked (if applicable)
        plugin_name: Name of plugin (if applicable)
        arguments: Function arguments
        metadata: Additional context metadata
    """
    kernel: Any  # Avoid circular import
    function_name: Optional[str] = None
    plugin_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Filter(ABC):
    """Base filter class.
    
    Filters can intercept processing at various points:
    - Before/after prompt rendering
    - Before/after function invocation
    - Post-processing results
    
    Example:
        class LoggingFilter(Filter):
            def on_function_invoking(self, ctx, args):
                print(f"Calling {ctx.function_name} with {args}")
                return args  # Can modify args
                
            def on_function_invoked(self, ctx, result):
                print(f"Result: {result}")
                return result
    """
    
    @property
    def filter_type(self) -> FilterType:
        """Type of filter."""
        return FilterType.FUNCTION
    
    def on_function_invoking(
        self,
        context: FilterContext,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Called before function invocation.
        
        Args:
            context: Filter context
            arguments: Function arguments
            
        Returns:
            Modified arguments (or original)
        """
        return arguments
    
    def on_function_invoked(
        self,
        context: FilterContext,
        result: Any
    ) -> Any:
        """Called after function invocation.
        
        Args:
            context: Filter context
            result: Function result
            
        Returns:
            Modified result (or original)
        """
        return result


class PromptFilter(Filter):
    """Filter for prompt rendering."""
    
    @property
    def filter_type(self) -> FilterType:
        return FilterType.PROMPT
    
    def on_prompt_rendering(
        self,
        context: FilterContext,
        prompt: str
    ) -> str:
        """Called before prompt is sent to LLM.
        
        Args:
            context: Filter context
            prompt: The prompt text
            
        Returns:
            Modified prompt (or original)
        """
        return prompt
    
    def on_prompt_rendered(
        self,
        context: FilterContext,
        prompt: str,
        result: str
    ) -> str:
        """Called after LLM response.
        
        Args:
            context: Filter context
            prompt: The original prompt
            result: The LLM response
            
        Returns:
            Modified result (or original)
        """
        return result


class FunctionFilter(Filter):
    """Filter for function invocation."""
    
    @property
    def filter_type(self) -> FilterType:
        return FilterType.FUNCTION


class FilterRegistry:
    """Registry for managing filters.
    
    Provides centralized filter management with:
    - Ordered filter chains
    - Type-based filtering
    - Filter lifecycle management
    
    Example:
        registry = FilterRegistry()
        
        # Add filters
        registry.add(LoggingFilter())
        registry.add(ValidationFilter())
        
        # Apply filters to function call
        args = registry.apply_function_invoking(ctx, args)
        result = func(**args)
        result = registry.apply_function_invoked(ctx, result)
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._filters: List[Filter] = []
    
    def add(self, filter: Filter, priority: int = 100) -> "FilterRegistry":
        """Add a filter.
        
        Args:
            filter: Filter to add
            priority: Execution priority (lower = earlier)
            
        Returns:
            Self for chaining
        """
        # Store with priority for sorting
        self._filters.append((priority, filter))
        self._filters.sort(key=lambda x: x[0])
        return self
    
    def remove(self, filter: Filter) -> bool:
        """Remove a filter.
        
        Args:
            filter: Filter to remove
            
        Returns:
            True if filter was removed
        """
        for i, (_, f) in enumerate(self._filters):
            if f is filter:
                self._filters.pop(i)
                return True
        return False
    
    def get_filters(
        self,
        filter_type: Optional[FilterType] = None
    ) -> List[Filter]:
        """Get filters, optionally filtered by type.
        
        Args:
            filter_type: Filter by type (optional)
            
        Returns:
            List of filters
        """
        filters = [f for _, f in self._filters]
        if filter_type is not None:
            filters = [f for f in filters if f.filter_type == filter_type]
        return filters
    
    def apply_function_invoking(
        self,
        context: FilterContext,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply all function filters before invocation.
        
        Args:
            context: Filter context
            arguments: Function arguments
            
        Returns:
            Modified arguments
        """
        result = arguments
        for filter in self.get_filters(FilterType.FUNCTION):
            result = filter.on_function_invoking(context, result)
        return result
    
    def apply_function_invoked(
        self,
        context: FilterContext,
        result: Any
    ) -> Any:
        """Apply all function filters after invocation.
        
        Args:
            context: Filter context
            result: Function result
            
        Returns:
            Modified result
        """
        for filter in self.get_filters(FilterType.FUNCTION):
            result = filter.on_function_invoked(context, result)
        return result
    
    def apply_prompt_rendering(
        self,
        context: FilterContext,
        prompt: str
    ) -> str:
        """Apply all prompt filters before rendering.
        
        Args:
            context: Filter context
            prompt: The prompt text
            
        Returns:
            Modified prompt
        """
        result = prompt
        for filter in self.get_filters(FilterType.PROMPT):
            if isinstance(filter, PromptFilter):
                result = filter.on_prompt_rendering(context, result)
        return result
    
    def apply_prompt_rendered(
        self,
        context: FilterContext,
        prompt: str,
        result: str
    ) -> str:
        """Apply all prompt filters after rendering.
        
        Args:
            context: Filter context
            prompt: Original prompt
            result: LLM response
            
        Returns:
            Modified result
        """
        for filter in self.get_filters(FilterType.PROMPT):
            if isinstance(filter, PromptFilter):
                result = filter.on_prompt_rendered(context, prompt, result)
        return result
    
    def clear(self) -> None:
        """Remove all filters."""
        self._filters.clear()
