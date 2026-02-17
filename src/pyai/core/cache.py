# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Context Caching

Cache contexts and prompts to reduce token costs.
Like Google ADK's context caching.
"""

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, Optional


@dataclass
class CacheEntry:
    """A cached context entry.

    Attributes:
        key: Cache key
        value: Cached value
        created_at: Creation timestamp
        expires_at: Expiration timestamp
        hits: Number of cache hits
        tokens: Estimated token count
    """

    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    hits: int = 0
    tokens: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class ContextCache:
    """Cache for context data to reduce token costs.

    Useful for caching:
    - Large system prompts
    - Repeated context data
    - RAG results

    Example:
        cache = ContextCache(ttl=3600)  # 1 hour TTL

        # Cache a value
        cache.set("system_prompt", long_prompt)

        # Get cached value
        prompt = cache.get("system_prompt")

        # Or use decorator
        @cache.cached
        def get_knowledge_base():
            return expensive_load()
    """

    def __init__(
        self, ttl: Optional[float] = None, max_entries: int = 1000, max_tokens: Optional[int] = None
    ):
        """Initialize cache.

        Args:
            ttl: Default time-to-live in seconds
            max_entries: Maximum cache entries
            max_tokens: Maximum total tokens to cache
        """
        self.ttl = ttl
        self.max_entries = max_entries
        self.max_tokens = max_tokens

        self._entries: Dict[str, CacheEntry] = {}
        self._total_tokens = 0
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get a cached value.

        Args:
            key: Cache key
            default: Default if not found

        Returns:
            Cached value or default
        """
        with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return default

            if entry.is_expired:
                self._remove_entry(key)
                self._stats["misses"] += 1
                return default

            entry.hits += 1
            self._stats["hits"] += 1
            return entry.value

    def set(
        self, key: str, value: Any, ttl: Optional[float] = None, tokens: Optional[int] = None
    ) -> CacheEntry:
        """Set a cached value.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Override TTL for this entry
            tokens: Estimated token count

        Returns:
            The cache entry
        """
        with self._lock:
            # Calculate expiration
            entry_ttl = ttl if ttl is not None else self.ttl
            expires_at = None
            if entry_ttl is not None:
                expires_at = time.time() + entry_ttl

            # Estimate tokens if not provided
            if tokens is None:
                tokens = self._estimate_tokens(value)

            # Evict if necessary
            self._evict_if_needed(tokens)

            # Remove existing entry
            if key in self._entries:
                self._remove_entry(key)

            # Create entry
            entry = CacheEntry(key=key, value=value, expires_at=expires_at, tokens=tokens)

            self._entries[key] = entry
            self._total_tokens += tokens

            return entry

    def delete(self, key: str) -> bool:
        """Delete a cached entry.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if key in self._entries:
                self._remove_entry(key)
                return True
            return False

    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._entries.clear()
            self._total_tokens = 0

    def has(self, key: str) -> bool:
        """Check if key is cached and not expired."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False
            if entry.is_expired:
                self._remove_entry(key)
                return False
            return True

    def _remove_entry(self, key: str):
        """Remove an entry and update token count."""
        entry = self._entries.pop(key, None)
        if entry:
            self._total_tokens -= entry.tokens

    def _evict_if_needed(self, incoming_tokens: int):
        """Evict entries if limits exceeded."""
        # Check entry limit
        while len(self._entries) >= self.max_entries:
            self._evict_oldest()

        # Check token limit
        if self.max_tokens:
            while self._total_tokens + incoming_tokens > self.max_tokens:
                if not self._evict_oldest():
                    break

    def _evict_oldest(self) -> bool:
        """Evict the oldest entry."""
        if not self._entries:
            return False

        # Find oldest entry
        oldest_key = None
        oldest_time = float("inf")

        for key, entry in self._entries.items():
            if entry.created_at < oldest_time:
                oldest_time = entry.created_at
                oldest_key = key

        if oldest_key:
            self._remove_entry(oldest_key)
            self._stats["evictions"] += 1
            return True

        return False

    def _estimate_tokens(self, value: Any) -> int:
        """Estimate token count for a value."""
        if isinstance(value, str):
            # Rough estimate: 4 chars per token
            return max(1, len(value) // 4)
        elif isinstance(value, (list, dict)):
            # Serialize and estimate
            try:
                serialized = json.dumps(value)
                return max(1, len(serialized) // 4)
            except:
                return 100  # Default estimate
        else:
            return 10  # Default for unknown types

    def cached(self, key: Optional[str] = None, ttl: Optional[float] = None) -> Callable:
        """Decorator to cache function results.

        Args:
            key: Cache key (defaults to function name)
            ttl: Override TTL

        Returns:
            Decorator function

        Example:
            cache = ContextCache(ttl=3600)

            @cache.cached
            def load_knowledge():
                return expensive_operation()
        """

        def decorator(func: Callable) -> Callable:
            cache_key = key or func.__name__

            @wraps(func)
            def wrapper(*args, **kwargs):
                # Build key with args
                if args or kwargs:
                    arg_key = hashlib.md5(
                        json.dumps((args, kwargs), sort_keys=True, default=str).encode()
                    ).hexdigest()[:8]
                    full_key = f"{cache_key}:{arg_key}"
                else:
                    full_key = cache_key

                # Check cache
                cached_value = self.get(full_key)
                if cached_value is not None:
                    return cached_value

                # Call function
                result = func(*args, **kwargs)

                # Cache result
                self.set(full_key, result, ttl=ttl)

                return result

            wrapper._cache_key = cache_key
            return wrapper

        # Handle @cache.cached without parentheses
        if callable(key):
            func = key
            key = None
            return decorator(func)

        return decorator

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = 0.0
            total = self._stats["hits"] + self._stats["misses"]
            if total > 0:
                hit_rate = self._stats["hits"] / total

            return {
                **self._stats,
                "entries": len(self._entries),
                "total_tokens": self._total_tokens,
                "hit_rate": hit_rate,
            }


def cache_context(ttl: Optional[float] = None, key: Optional[str] = None) -> Callable:
    """Decorator to cache context-returning functions.

    Creates a shared cache for context caching.

    Args:
        ttl: Time-to-live in seconds
        key: Cache key (defaults to function name)

    Returns:
        Decorator function

    Example:
        @cache_context(ttl=3600)
        def get_system_prompt():
            return "Very long system prompt..."
    """
    _shared_cache = ContextCache(ttl=ttl)

    def decorator(func: Callable) -> Callable:
        cache_key = key or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check cache
            cached = _shared_cache.get(cache_key)
            if cached is not None:
                return cached

            # Execute and cache
            result = func(*args, **kwargs)
            _shared_cache.set(cache_key, result)

            return result

        wrapper._cache = _shared_cache
        return wrapper

    return decorator
