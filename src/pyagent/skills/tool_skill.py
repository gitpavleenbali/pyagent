"""
ToolSkill - Function-based tools with decorator support
"""

from typing import Any, Callable, Dict, List, Optional, get_type_hints
from dataclasses import dataclass, field
from functools import wraps
import inspect
import asyncio

from pyagent.skills.skill import Skill, SkillResult, SkillParameter


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs
) -> Callable:
    """
    Decorator to convert a function into a ToolSkill.
    
    Automatically extracts parameter information from type hints
    and docstrings.
    
    Example:
        >>> @tool(description="Search the web")
        ... async def search(query: str, limit: int = 10) -> list:
        ...     '''
        ...     Search for information online.
        ...     
        ...     Args:
        ...         query: The search query
        ...         limit: Maximum results to return
        ...     '''
        ...     return await web_search(query, limit)
        ...
        >>> # search is now a ToolSkill instance
        >>> result = await search(query="python tutorials")
    """
    def decorator(func: Callable) -> "ToolSkill":
        skill = ToolSkill.from_function(
            func,
            name=name,
            description=description,
            **kwargs
        )
        
        # Preserve function metadata
        @wraps(func)
        async def wrapper(**kw):
            return await skill.execute(**kw)
        
        # Attach skill to wrapper for introspection
        wrapper._skill = skill
        wrapper.to_tool_definition = skill.to_tool_definition
        
        return skill
    
    return decorator


class ToolSkill(Skill):
    """
    ToolSkill - A skill created from a callable function.
    
    ToolSkill wraps existing functions, automatically extracting:
    - Parameter names and types from signatures
    - Descriptions from docstrings
    - Default values
    
    Example:
        >>> def calculate(expression: str) -> float:
        ...     '''Evaluate a math expression'''
        ...     return eval(expression)
        ...
        >>> skill = ToolSkill.from_function(calculate)
        >>> result = await skill.execute(expression="2 + 2")
    """
    
    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs
    ):
        self._func = func
        self._is_async = asyncio.iscoroutinefunction(func)
        
        # Extract name and description
        func_name = name or func.__name__
        func_desc = description or self._extract_description(func)
        
        super().__init__(name=func_name, description=func_desc, **kwargs)
        
        # Extract parameters from function signature
        self.parameters = self._extract_parameters(func)
    
    @staticmethod
    def _extract_description(func: Callable) -> str:
        """Extract description from function docstring"""
        doc = func.__doc__
        if not doc:
            return f"Execute {func.__name__}"
        
        # Get first paragraph of docstring
        lines = doc.strip().split('\n\n')
        return lines[0].strip()
    
    @staticmethod
    def _extract_parameters(func: Callable) -> List[SkillParameter]:
        """Extract parameters from function signature and type hints"""
        params = []
        sig = inspect.signature(func)
        hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}
        
        # Parse docstring for parameter descriptions
        param_docs = ToolSkill._parse_param_docs(func.__doc__ or "")
        
        for pname, param in sig.parameters.items():
            if pname in ('self', 'cls'):
                continue
            
            # Determine type
            ptype = "string"
            if pname in hints:
                hint = hints[pname]
                ptype = ToolSkill._type_to_json_type(hint)
            
            # Check if required (has no default)
            required = param.default == inspect.Parameter.empty
            default = None if required else param.default
            
            # Get description from docstring
            pdesc = param_docs.get(pname, f"The {pname} parameter")
            
            params.append(SkillParameter(
                name=pname,
                description=pdesc,
                type=ptype,
                required=required,
                default=default,
            ))
        
        return params
    
    @staticmethod
    def _parse_param_docs(docstring: str) -> Dict[str, str]:
        """Parse parameter descriptions from docstring"""
        param_docs = {}
        
        if not docstring:
            return param_docs
        
        # Look for Args: or Parameters: section
        lines = docstring.split('\n')
        in_args = False
        current_param = None
        
        for line in lines:
            stripped = line.strip()
            
            if stripped.lower().startswith(('args:', 'parameters:')):
                in_args = True
                continue
            elif stripped.lower().startswith(('returns:', 'raises:', 'example:')):
                in_args = False
                continue
            
            if in_args and stripped:
                # Check for parameter definition (name: description)
                if ':' in stripped and not stripped.startswith(' '):
                    parts = stripped.split(':', 1)
                    param_name = parts[0].strip()
                    param_desc = parts[1].strip() if len(parts) > 1 else ""
                    param_docs[param_name] = param_desc
                    current_param = param_name
                elif current_param and stripped:
                    # Continuation of previous param description
                    param_docs[current_param] += " " + stripped
        
        return param_docs
    
    @staticmethod
    def _type_to_json_type(hint: type) -> str:
        """Convert Python type hint to JSON Schema type"""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        
        # Handle generic types
        origin = getattr(hint, '__origin__', None)
        if origin:
            return type_map.get(origin, "string")
        
        return type_map.get(hint, "string")
    
    @classmethod
    def from_function(
        cls,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs
    ) -> "ToolSkill":
        """Create a ToolSkill from a function"""
        return cls(func, name=name, description=description, **kwargs)
    
    async def execute(self, **kwargs) -> SkillResult:
        """Execute the wrapped function"""
        try:
            # Validate required parameters
            if not await self.validate_input(**kwargs):
                missing = [
                    p.name for p in self.parameters
                    if p.required and p.name not in kwargs
                ]
                return SkillResult.fail(f"Missing required parameters: {missing}")
            
            # Apply defaults
            for param in self.parameters:
                if param.name not in kwargs and param.default is not None:
                    kwargs[param.name] = param.default
            
            # Execute function
            if self._is_async:
                result = await self._func(**kwargs)
            else:
                result = self._func(**kwargs)
            
            return SkillResult.ok(result)
            
        except Exception as e:
            return SkillResult.fail(str(e))


class ToolRegistry:
    """
    Registry for managing tool functions.
    
    Example:
        >>> registry = ToolRegistry()
        >>> 
        >>> @registry.register
        ... def search(query: str) -> list:
        ...     '''Search the web'''
        ...     return []
        >>> 
        >>> tools = registry.get_all_tools()
    """
    
    def __init__(self):
        self._tools: Dict[str, ToolSkill] = {}
    
    def register(
        self,
        func: Optional[Callable] = None,
        name: Optional[str] = None,
        **kwargs
    ) -> Callable:
        """Register a function as a tool"""
        def decorator(f: Callable) -> Callable:
            tool_skill = ToolSkill.from_function(f, name=name, **kwargs)
            self._tools[tool_skill.name] = tool_skill
            return f
        
        if func is not None:
            return decorator(func)
        return decorator
    
    def get(self, name: str) -> Optional[ToolSkill]:
        """Get a tool by name"""
        return self._tools.get(name)
    
    def get_all_tools(self) -> List[ToolSkill]:
        """Get all registered tools"""
        return list(self._tools.values())
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tool definitions"""
        return [t.to_tool_definition() for t in self._tools.values()]
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
    
    def __len__(self) -> int:
        return len(self._tools)
