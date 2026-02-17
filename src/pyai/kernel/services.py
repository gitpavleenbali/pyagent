# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Service Registry

Centralized service management for the Kernel.
Supports LLM providers, memory systems, vector stores, etc.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar


class ServiceType(Enum):
    """Types of services that can be registered."""

    LLM = "llm"
    MEMORY = "memory"
    VECTOR = "vector"
    EMBEDDING = "embedding"
    CACHE = "cache"
    LOGGING = "logging"
    CUSTOM = "custom"


@dataclass
class Service:
    """Base service descriptor.

    Attributes:
        name: Unique service identifier
        instance: Service instance or factory
        service_type: Type of service
        is_default: Whether this is the default for its type
        metadata: Additional service metadata
    """

    name: str
    instance: Any
    service_type: ServiceType = ServiceType.CUSTOM
    is_default: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


T = TypeVar("T")


class ServiceRegistry(Generic[T]):
    """Registry for managing services.

    Provides centralized service management with:
    - Type-safe service retrieval
    - Default service selection
    - Service lifecycle management

    Example:
        registry = ServiceRegistry()

        # Register services
        registry.add(Service(
            name="azure-gpt4",
            instance=azure_client,
            service_type=ServiceType.LLM,
            is_default=True
        ))

        # Get default LLM
        llm = registry.get_default(ServiceType.LLM)

        # Get specific service
        client = registry.get("azure-gpt4")
    """

    def __init__(self):
        """Initialize empty registry."""
        self._services: Dict[str, Service] = {}
        self._defaults: Dict[ServiceType, str] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}

    def add(
        self, service: Service, *, lazy: bool = False, factory: Optional[Callable[[], Any]] = None
    ) -> "ServiceRegistry":
        """Register a service.

        Args:
            service: Service descriptor
            lazy: If True, use factory for lazy initialization
            factory: Factory function for lazy initialization

        Returns:
            Self for chaining
        """
        self._services[service.name] = service

        if lazy and factory:
            self._factories[service.name] = factory

        if service.is_default:
            self._defaults[service.service_type] = service.name

        return self

    def add_instance(
        self,
        name: str,
        instance: Any,
        service_type: ServiceType = ServiceType.CUSTOM,
        is_default: bool = False,
    ) -> "ServiceRegistry":
        """Convenience method to register an instance directly.

        Args:
            name: Service name
            instance: Service instance
            service_type: Type of service
            is_default: Whether this is the default

        Returns:
            Self for chaining
        """
        return self.add(
            Service(name=name, instance=instance, service_type=service_type, is_default=is_default)
        )

    def get(self, name: str) -> Optional[Any]:
        """Get a service by name.

        Args:
            name: Service name

        Returns:
            Service instance or None
        """
        if name not in self._services:
            return None

        service = self._services[name]

        # Lazy initialization
        if name in self._factories and service.instance is None:
            service.instance = self._factories[name]()

        return service.instance

    def get_default(self, service_type: ServiceType) -> Optional[Any]:
        """Get the default service for a type.

        Args:
            service_type: Type of service

        Returns:
            Default service instance or None
        """
        if service_type not in self._defaults:
            # Return first service of type if no explicit default
            for service in self._services.values():
                if service.service_type == service_type:
                    return service.instance
            return None

        return self.get(self._defaults[service_type])

    def remove(self, name: str) -> bool:
        """Remove a service.

        Args:
            name: Service name

        Returns:
            True if service was removed
        """
        if name in self._services:
            service = self._services[name]

            # Remove from defaults if applicable
            if service.is_default:
                self._defaults.pop(service.service_type, None)

            del self._services[name]
            self._factories.pop(name, None)
            return True
        return False

    def list_services(self, service_type: Optional[ServiceType] = None) -> List[str]:
        """List registered services.

        Args:
            service_type: Filter by type (optional)

        Returns:
            List of service names
        """
        if service_type is None:
            return list(self._services.keys())

        return [name for name, svc in self._services.items() if svc.service_type == service_type]

    def has(self, name: str) -> bool:
        """Check if a service is registered."""
        return name in self._services

    def clear(self) -> None:
        """Remove all services."""
        self._services.clear()
        self._defaults.clear()
        self._factories.clear()


# Convenience service classes
class LLMService(Service):
    """LLM service descriptor."""

    def __init__(
        self,
        name: str,
        instance: Any,
        is_default: bool = False,
        model: Optional[str] = None,
        **metadata,
    ):
        super().__init__(
            name=name,
            instance=instance,
            service_type=ServiceType.LLM,
            is_default=is_default,
            metadata={"model": model, **metadata},
        )


class MemoryService(Service):
    """Memory service descriptor."""

    def __init__(
        self,
        name: str,
        instance: Any,
        is_default: bool = False,
        max_tokens: Optional[int] = None,
        **metadata,
    ):
        super().__init__(
            name=name,
            instance=instance,
            service_type=ServiceType.MEMORY,
            is_default=is_default,
            metadata={"max_tokens": max_tokens, **metadata},
        )


class VectorService(Service):
    """Vector store service descriptor."""

    def __init__(
        self,
        name: str,
        instance: Any,
        is_default: bool = False,
        dimensions: Optional[int] = None,
        **metadata,
    ):
        super().__init__(
            name=name,
            instance=instance,
            service_type=ServiceType.VECTOR,
            is_default=is_default,
            metadata={"dimensions": dimensions, **metadata},
        )
