"""
Context - Dynamic context injection for instructions
"""

from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime


@dataclass
class Context:
    """
    Context - Static context data to inject into instructions.
    
    Context provides additional information that shapes agent behavior
    at runtime, such as user preferences, session data, or domain knowledge.
    
    Example:
        >>> context = Context(
        ...     user_name="Alice",
        ...     user_role="developer",
        ...     project="e-commerce app",
        ... )
        >>> instruction = Instruction("Help {user_name} with their {project}")
        >>> rendered = instruction.render(context.to_dict())
    """
    
    name: str = "default"
    data: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, name: str = "default", **kwargs):
        self.name = name
        self.data = kwargs
    
    def set(self, key: str, value: Any) -> "Context":
        """Set a context value"""
        self.data[key] = value
        return self
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a context value"""
        return self.data.get(key, default)
    
    def update(self, **kwargs) -> "Context":
        """Update multiple context values"""
        self.data.update(kwargs)
        return self
    
    def merge(self, other: "Context") -> "Context":
        """Merge another context into this one"""
        merged_data = {**self.data, **other.data}
        return Context(name=self.name, **merged_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.data.copy()
    
    def to_prompt_section(self) -> str:
        """Render context as a prompt section"""
        if not self.data:
            return ""
        
        lines = ["## Current Context"]
        for key, value in self.data.items():
            # Format key nicely
            formatted_key = key.replace("_", " ").title()
            lines.append(f"- {formatted_key}: {value}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"Context({self.name}, keys={list(self.data.keys())})"


class DynamicContext(ABC):
    """
    DynamicContext - Context that is computed at runtime.
    
    Use this for context that needs to be fetched or computed
    when the agent runs, such as current time, API data, etc.
    
    Example:
        >>> class TimeContext(DynamicContext):
        ...     def resolve(self) -> Dict[str, Any]:
        ...         return {"current_time": datetime.now().isoformat()}
    """
    
    @abstractmethod
    def resolve(self) -> Dict[str, Any]:
        """Resolve the dynamic context to a dictionary"""
        pass
    
    def to_context(self) -> Context:
        """Convert to a static Context"""
        return Context(**self.resolve())


class FunctionContext(DynamicContext):
    """
    Context that calls a function to get its values.
    
    Example:
        >>> def get_user_info():
        ...     return {"user": "Alice", "role": "admin"}
        >>> context = FunctionContext(get_user_info)
    """
    
    def __init__(self, provider: Callable[[], Dict[str, Any]]):
        self.provider = provider
    
    def resolve(self) -> Dict[str, Any]:
        return self.provider()


class TimeContext(DynamicContext):
    """Context that provides current time information"""
    
    def __init__(self, include_timezone: bool = True):
        self.include_timezone = include_timezone
    
    def resolve(self) -> Dict[str, Any]:
        now = datetime.now()
        result = {
            "current_date": now.strftime("%Y-%m-%d"),
            "current_time": now.strftime("%H:%M:%S"),
            "current_datetime": now.isoformat(),
            "day_of_week": now.strftime("%A"),
        }
        if self.include_timezone:
            result["timezone"] = datetime.now().astimezone().tzname()
        return result


class EnvironmentContext(DynamicContext):
    """Context from environment variables"""
    
    def __init__(self, keys: List[str], prefix: str = ""):
        self.keys = keys
        self.prefix = prefix
    
    def resolve(self) -> Dict[str, Any]:
        import os
        result = {}
        for key in self.keys:
            env_key = f"{self.prefix}{key}" if self.prefix else key
            if env_key in os.environ:
                result[key.lower()] = os.environ[env_key]
        return result


class CompositeContext(DynamicContext):
    """Combines multiple context sources"""
    
    def __init__(self, *sources: Union[Context, DynamicContext]):
        self.sources = sources
    
    def resolve(self) -> Dict[str, Any]:
        result = {}
        for source in self.sources:
            if isinstance(source, Context):
                result.update(source.to_dict())
            elif isinstance(source, DynamicContext):
                result.update(source.resolve())
        return result


class SessionContext(Context):
    """
    Context that tracks session state across interactions.
    
    Useful for maintaining state like conversation turn count,
    user preferences discovered during conversation, etc.
    """
    
    def __init__(self, session_id: Optional[str] = None, **kwargs):
        import uuid
        super().__init__(name="session", **kwargs)
        self.session_id = session_id or str(uuid.uuid4())
        self.turn_count = 0
        self.created_at = datetime.now()
    
    def increment_turn(self) -> None:
        """Increment the conversation turn count"""
        self.turn_count += 1
        self.data["turn_count"] = self.turn_count
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.data,
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "session_duration_seconds": (datetime.now() - self.created_at).total_seconds(),
        }
