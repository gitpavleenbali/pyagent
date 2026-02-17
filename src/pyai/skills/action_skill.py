"""
ActionSkill - Discrete actions with structured inputs/outputs
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pyai.skills.skill import Skill, SkillResult


class ActionType(Enum):
    """Types of actions"""

    READ = "read"  # Reading/retrieving data
    WRITE = "write"  # Writing/modifying data
    EXECUTE = "execute"  # Running/executing something
    ANALYZE = "analyze"  # Analyzing data
    TRANSFORM = "transform"  # Transforming data


@dataclass
class ActionInput:
    """Structured input for an action"""

    data: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getattr__(self, name: str) -> Any:
        if name in self.data:
            return self.data[name]
        raise AttributeError(f"No attribute '{name}'")


@dataclass
class ActionOutput:
    """Structured output from an action"""

    result: Any = None
    artifacts: List[Any] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)

    def add_artifact(self, artifact: Any) -> None:
        self.artifacts.append(artifact)

    def log(self, message: str) -> None:
        self.logs.append(message)


@dataclass
class Action:
    """
    Action - A discrete, named operation.

    Actions are smaller than skills - they represent single
    operations that can be composed into larger workflows.

    Example:
        >>> @Action.define("read_file")
        ... async def read_file(path: str) -> str:
        ...     with open(path) as f:
        ...         return f.read()
    """

    name: str
    description: str = ""
    action_type: ActionType = ActionType.EXECUTE
    handler: Optional[Callable] = None

    async def run(self, **kwargs) -> Any:
        """Execute the action"""
        if self.handler:
            result = self.handler(**kwargs)
            if hasattr(result, "__await__"):
                result = await result
            return result
        raise NotImplementedError("Action has no handler")

    @classmethod
    def define(
        cls,
        name: str,
        description: str = "",
        action_type: ActionType = ActionType.EXECUTE,
    ) -> Callable:
        """Decorator to define an action from a function"""

        def decorator(func: Callable) -> "Action":
            return cls(
                name=name,
                description=description or func.__doc__ or "",
                action_type=action_type,
                handler=func,
            )

        return decorator


class ActionSkill(Skill):
    """
    ActionSkill - A skill that executes discrete actions.

    ActionSkill manages a set of related actions and provides
    a unified interface for executing them.

    Example:
        >>> class FileSkill(ActionSkill):
        ...     @action("read")
        ...     async def read_file(self, path: str) -> str:
        ...         return open(path).read()
        ...
        ...     @action("write")
        ...     async def write_file(self, path: str, content: str):
        ...         open(path, 'w').write(content)
    """

    def __init__(self, name: Optional[str] = None, description: Optional[str] = None, **kwargs):
        super().__init__(name=name, description=description, **kwargs)
        self._actions: Dict[str, Action] = {}
        self._collect_actions()

    def _collect_actions(self) -> None:
        """Collect action methods from the class"""
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue

            attr = getattr(self, attr_name)
            if isinstance(attr, Action):
                self._actions[attr.name] = attr
            elif hasattr(attr, "_action_metadata"):
                # Method decorated with @action
                metadata = attr._action_metadata
                self._actions[metadata["name"]] = Action(
                    name=metadata["name"],
                    description=metadata.get("description", ""),
                    action_type=metadata.get("action_type", ActionType.EXECUTE),
                    handler=attr,
                )

    def register_action(self, action: Action) -> None:
        """Register an action with this skill"""
        self._actions[action.name] = action

    @property
    def actions(self) -> List[str]:
        """List of available action names"""
        return list(self._actions.keys())

    async def execute(self, action: str = None, **kwargs) -> SkillResult:
        """
        Execute an action.

        Args:
            action: The action name to execute
            **kwargs: Parameters for the action
        """
        if not action:
            return SkillResult.fail("No action specified")

        if action not in self._actions:
            return SkillResult.fail(f"Unknown action: {action}. Available: {self.actions}")

        try:
            act = self._actions[action]
            result = await act.run(**kwargs)
            return SkillResult.ok(result)
        except Exception as e:
            return SkillResult.fail(str(e))

    def to_tool_definition(self) -> Dict[str, Any]:
        """Convert to tool definition with action parameter"""
        # Include action as an enum parameter
        base_def = super().to_tool_definition()

        base_def["function"]["parameters"]["properties"]["action"] = {
            "type": "string",
            "description": "The action to execute",
            "enum": self.actions,
        }

        if "required" not in base_def["function"]["parameters"]:
            base_def["function"]["parameters"]["required"] = []
        base_def["function"]["parameters"]["required"].append("action")

        return base_def


def action(
    name: str,
    description: str = "",
    action_type: ActionType = ActionType.EXECUTE,
) -> Callable:
    """
    Decorator to mark a method as an action.

    Example:
        >>> class MySkill(ActionSkill):
        ...     @action("greet", description="Greet a user")
        ...     async def greet(self, name: str) -> str:
        ...         return f"Hello, {name}!"
    """

    def decorator(func: Callable) -> Callable:
        func._action_metadata = {
            "name": name,
            "description": description or func.__doc__ or "",
            "action_type": action_type,
        }
        return func

    return decorator


# Common action patterns
class CRUDSkill(ActionSkill):
    """
    Base class for CRUD (Create, Read, Update, Delete) skills.

    Provides a standard interface for resource management.
    """

    resource_name: str = "resource"

    @action("create", action_type=ActionType.WRITE)
    async def create(self, **kwargs) -> Any:
        """Create a new resource"""
        raise NotImplementedError

    @action("read", action_type=ActionType.READ)
    async def read(self, id: str) -> Any:
        """Read a resource by ID"""
        raise NotImplementedError

    @action("update", action_type=ActionType.WRITE)
    async def update(self, id: str, **kwargs) -> Any:
        """Update a resource"""
        raise NotImplementedError

    @action("delete", action_type=ActionType.WRITE)
    async def delete(self, id: str) -> bool:
        """Delete a resource"""
        raise NotImplementedError

    @action("list", action_type=ActionType.READ)
    async def list(self, **filters) -> List[Any]:
        """List resources with optional filters"""
        raise NotImplementedError
