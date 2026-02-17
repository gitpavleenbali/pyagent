"""
Base classes and interfaces for pyai components
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class BaseComponent(ABC):
    """
    Base class for all pyai components.

    Provides common functionality like unique identification,
    metadata tracking, and lifecycle management.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Initialize component after dataclass creation"""
        if self.name is None:
            self.name = self.__class__.__name__

    @abstractmethod
    def validate(self) -> bool:
        """Validate the component configuration"""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "type": self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseComponent":
        """Deserialize component from dictionary"""
        raise NotImplementedError("Subclasses must implement from_dict")


class Configurable(ABC):
    """Mixin for components that can be configured"""

    @abstractmethod
    def configure(self, **kwargs) -> None:
        """Configure the component with given parameters"""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        pass


class Executable(ABC):
    """Mixin for components that can be executed"""

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Execute the component's primary function"""
        pass

    @abstractmethod
    async def validate_input(self, *args, **kwargs) -> bool:
        """Validate input before execution"""
        pass
