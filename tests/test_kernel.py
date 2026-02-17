# pyright: reportMissingImports=false, reportUnusedVariable=false, reportUnusedImport=false, reportGeneralTypeIssues=false, reportRedeclaration=false
# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
Tests for the Kernel module.

Tests the Kernel Registry pattern - centralized service management,
plugin registration, filter middleware, and execution context.
"""

import pytest  # noqa: F401
from unittest.mock import Mock, patch, AsyncMock  # noqa: F401


class TestServiceRegistry:
    """Tests for ServiceRegistry."""
    
    def test_registry_creation(self):
        """Test creating a service registry."""
        from pyai.kernel import ServiceRegistry
        
        registry = ServiceRegistry()
        assert registry.list_services() == []
    
    def test_add_service(self):
        """Test adding a service."""
        from pyai.kernel import ServiceRegistry, Service, ServiceType
        
        registry = ServiceRegistry()
        
        mock_llm = Mock()
        service = Service(
            name="gpt4",
            instance=mock_llm,
            service_type=ServiceType.LLM,
            is_default=True
        )
        
        registry.add(service)
        
        assert "gpt4" in registry.list_services()
        assert registry.get("gpt4") is mock_llm
    
    def test_add_instance_directly(self):
        """Test adding an instance directly."""
        from pyai.kernel import ServiceRegistry, ServiceType
        
        registry = ServiceRegistry()
        mock_memory = Mock()
        
        registry.add_instance(
            name="memory",
            instance=mock_memory,
            service_type=ServiceType.MEMORY,
            is_default=True
        )
        
        assert registry.get("memory") is mock_memory
    
    def test_get_default_service(self):
        """Test getting default service by type."""
        from pyai.kernel import ServiceRegistry, ServiceType, LLMService
        
        registry = ServiceRegistry()
        
        mock_llm1 = Mock(name="llm1")
        mock_llm2 = Mock(name="llm2")
        
        registry.add(LLMService("gpt3", mock_llm1, is_default=False))
        registry.add(LLMService("gpt4", mock_llm2, is_default=True))
        
        default = registry.get_default(ServiceType.LLM)
        assert default is mock_llm2
    
    def test_remove_service(self):
        """Test removing a service."""
        from pyai.kernel import ServiceRegistry, ServiceType
        
        registry = ServiceRegistry()
        registry.add_instance("svc", Mock(), ServiceType.CUSTOM)
        
        assert registry.has("svc")
        assert registry.remove("svc")
        assert not registry.has("svc")
    
    def test_list_services_by_type(self):
        """Test listing services filtered by type."""
        from pyai.kernel import ServiceRegistry, ServiceType
        
        registry = ServiceRegistry()
        registry.add_instance("llm1", Mock(), ServiceType.LLM)
        registry.add_instance("llm2", Mock(), ServiceType.LLM)
        registry.add_instance("mem1", Mock(), ServiceType.MEMORY)
        
        llm_services = registry.list_services(ServiceType.LLM)
        assert len(llm_services) == 2
        assert "llm1" in llm_services
        assert "llm2" in llm_services


class TestFilterRegistry:
    """Tests for FilterRegistry."""
    
    def test_registry_creation(self):
        """Test creating a filter registry."""
        from pyai.kernel import FilterRegistry
        
        registry = FilterRegistry()
        assert registry.get_filters() == []
    
    def test_add_filter(self):
        """Test adding a filter."""
        from pyai.kernel import FilterRegistry, Filter
        
        registry = FilterRegistry()
        filter = Filter()
        
        registry.add(filter)
        
        assert len(registry.get_filters()) == 1
    
    def test_filter_priority(self):
        """Test filters are ordered by priority."""
        from pyai.kernel import FilterRegistry, Filter
        
        registry = FilterRegistry()
        
        filter1 = Filter()
        filter2 = Filter()
        filter3 = Filter()
        
        registry.add(filter2, priority=50)
        registry.add(filter1, priority=10)
        registry.add(filter3, priority=100)
        
        filters = registry.get_filters()
        assert filters[0] is filter1
        assert filters[1] is filter2
        assert filters[2] is filter3
    
    def test_apply_function_invoking(self):
        """Test applying function invoking filters."""
        from pyai.kernel import FilterRegistry, FunctionFilter, FilterContext
        
        class ModifyingFilter(FunctionFilter):
            def on_function_invoking(self, ctx, args):
                args["modified"] = True
                return args
        
        registry = FilterRegistry()
        registry.add(ModifyingFilter())
        
        ctx = FilterContext(kernel=None)
        args = {"value": 1}
        
        result = registry.apply_function_invoking(ctx, args)
        
        assert result["modified"] is True
        assert result["value"] == 1
    
    def test_apply_function_invoked(self):
        """Test applying function invoked filters."""
        from pyai.kernel import FilterRegistry, FunctionFilter, FilterContext
        
        class TransformingFilter(FunctionFilter):
            def on_function_invoked(self, ctx, result):
                return result.upper()
        
        registry = FilterRegistry()
        registry.add(TransformingFilter())
        
        ctx = FilterContext(kernel=None)
        result = registry.apply_function_invoked(ctx, "hello")
        
        assert result == "HELLO"


class TestKernelContext:
    """Tests for KernelContext."""
    
    def test_context_creation(self):
        """Test creating a context."""
        from pyai.kernel import KernelContext
        
        ctx = KernelContext()
        
        assert ctx.context_id is not None
        assert ctx.variables == {}
        assert ctx.invocations == []
    
    def test_variable_management(self):
        """Test setting and getting variables."""
        from pyai.kernel import KernelContext
        
        ctx = KernelContext()
        
        ctx.set_variable("name", "test")
        ctx.set_variable("count", 42)
        
        assert ctx.get_variable("name") == "test"
        assert ctx.get_variable("count") == 42
        assert ctx.get_variable("missing", "default") == "default"
    
    def test_invocation_tracking(self):
        """Test tracking invocations."""
        from pyai.kernel import KernelContext
        
        ctx = KernelContext()
        
        inv = ctx.create_invocation("weather", "get_forecast", {"city": "NYC"})
        inv.start()
        inv.complete({"temp": 72})
        
        assert ctx.total_invocations == 1
        assert inv.success
        assert inv.result == {"temp": 72}
        assert inv.duration_ms is not None
    
    def test_failed_invocation(self):
        """Test failed invocation tracking."""
        from pyai.kernel import KernelContext
        
        ctx = KernelContext()
        
        inv = ctx.create_invocation("plugin", "func")
        inv.start()
        inv.fail(ValueError("test error"))
        
        assert not inv.success
        assert len(ctx.failed_invocations) == 1
    
    def test_child_context(self):
        """Test child context inherits variables."""
        from pyai.kernel import KernelContext
        
        parent = KernelContext()
        parent.set_variable("inherited", "value")
        
        child = parent.create_child()
        
        # Child can access parent variables
        assert child.get_variable("inherited") == "value"
        
        # Child can override
        child.set_variable("inherited", "overridden")
        assert child.get_variable("inherited") == "overridden"
        assert parent.get_variable("inherited") == "value"
    
    def test_to_dict(self):
        """Test serialization to dict."""
        from pyai.kernel import KernelContext
        
        ctx = KernelContext()
        ctx.set_variable("key", "value")
        
        data = ctx.to_dict()
        
        assert "context_id" in data
        assert data["variables"]["key"] == "value"


class TestKernel:
    """Tests for the main Kernel class."""
    
    def test_kernel_creation(self):
        """Test creating a kernel."""
        from pyai.kernel import Kernel
        
        kernel = Kernel()
        
        assert kernel.services is not None
        assert kernel.plugins is not None
        assert kernel.filters is not None
        assert kernel.context is not None
    
    def test_add_service(self):
        """Test adding services to kernel."""
        from pyai.kernel import Kernel, LLMService, ServiceType
        
        kernel = Kernel()
        mock_llm = Mock()
        
        kernel.add_service(LLMService(
            name="gpt4",
            instance=mock_llm,
            is_default=True
        ))
        
        assert kernel.get_llm() is mock_llm
        assert kernel.get_service("gpt4") is mock_llm
    
    def test_add_raw_service(self):
        """Test adding raw instance as service."""
        from pyai.kernel import Kernel, ServiceType
        
        kernel = Kernel()
        mock_memory = Mock()
        
        kernel.add_service(
            mock_memory,
            name="memory",
            service_type=ServiceType.MEMORY,
            is_default=True
        )
        
        assert kernel.get_service(service_type=ServiceType.MEMORY) is mock_memory
    
    def test_add_plugin(self):
        """Test adding plugins to kernel."""
        from pyai.kernel import Kernel
        from pyai.plugins import Plugin
        
        kernel = Kernel()
        
        class TestPlugin(Plugin):
            name = "test"
        
        plugin = TestPlugin()
        kernel.add_plugin(plugin)
        
        assert kernel.get_plugin("test") is plugin
    
    def test_add_filter(self):
        """Test adding filters to kernel."""
        from pyai.kernel import Kernel, Filter
        
        kernel = Kernel()
        filter = Filter()
        
        kernel.add_filter(filter)
        
        assert len(kernel.filters.get_filters()) == 1
    
    def test_context_variables(self):
        """Test context variable management through kernel."""
        from pyai.kernel import Kernel
        
        kernel = Kernel()
        
        kernel.set_variable("input", "test")
        assert kernel.get_variable("input") == "test"
    
    def test_to_dict(self):
        """Test kernel state serialization."""
        from pyai.kernel import Kernel
        
        kernel = Kernel()
        data = kernel.to_dict()
        
        assert "services" in data
        assert "plugins" in data
        assert "agents" in data
        assert "context" in data
    
    def test_repr(self):
        """Test kernel string representation."""
        from pyai.kernel import Kernel
        
        kernel = Kernel()
        repr_str = repr(kernel)
        
        assert "Kernel" in repr_str
        assert "services=" in repr_str


class TestKernelBuilder:
    """Tests for KernelBuilder."""
    
    def test_builder_creation(self):
        """Test creating a builder."""
        from pyai.kernel import KernelBuilder
        
        builder = KernelBuilder()
        assert builder is not None
    
    def test_build_kernel(self):
        """Test building a kernel."""
        from pyai.kernel import KernelBuilder, LLMService, Filter
        
        kernel = (
            KernelBuilder()
            .add_service(LLMService("gpt4", Mock(), is_default=True))
            .add_filter(Filter())
            .build()
        )
        
        assert kernel is not None
        assert kernel.get_llm() is not None
        assert len(kernel.filters.get_filters()) == 1
    
    def test_fluent_api(self):
        """Test builder fluent API returns self."""
        from pyai.kernel import KernelBuilder, LLMService, Filter
        
        builder = KernelBuilder()
        
        result = builder.add_service(LLMService("test", Mock()))
        assert result is builder
        
        result = builder.add_filter(Filter())
        assert result is builder


class TestKernelImports:
    """Test that kernel is properly exported."""
    
    def test_import_from_pyai(self):
        """Test importing kernel from pyai."""
        from pyai import Kernel, KernelBuilder, ServiceRegistry
        
        assert Kernel is not None
        assert KernelBuilder is not None
        assert ServiceRegistry is not None
    
    def test_import_kernel_module(self):
        """Test importing kernel module."""
        from pyai import kernel
        
        assert hasattr(kernel, "Kernel")
        assert hasattr(kernel, "KernelBuilder")
        assert hasattr(kernel, "ServiceRegistry")
        assert hasattr(kernel, "FilterRegistry")
        assert hasattr(kernel, "KernelContext")


class TestServiceTypes:
    """Tests for service type classes."""
    
    def test_llm_service(self):
        """Test LLMService creation."""
        from pyai.kernel import LLMService, ServiceType
        
        service = LLMService(
            name="gpt4",
            instance=Mock(),
            is_default=True,
            model="gpt-4-turbo"
        )
        
        assert service.name == "gpt4"
        assert service.service_type == ServiceType.LLM
        assert service.is_default is True
        assert service.metadata["model"] == "gpt-4-turbo"
    
    def test_memory_service(self):
        """Test MemoryService creation."""
        from pyai.kernel import MemoryService, ServiceType
        
        service = MemoryService(
            name="memory",
            instance=Mock(),
            max_tokens=4000
        )
        
        assert service.service_type == ServiceType.MEMORY
        assert service.metadata["max_tokens"] == 4000
    
    def test_vector_service(self):
        """Test VectorService creation."""
        from pyai.kernel import VectorService, ServiceType
        
        service = VectorService(
            name="vectors",
            instance=Mock(),
            dimensions=1536
        )
        
        assert service.service_type == ServiceType.VECTOR
        assert service.metadata["dimensions"] == 1536


class TestInvocationContext:
    """Tests for InvocationContext."""
    
    def test_invocation_lifecycle(self):
        """Test invocation start/complete lifecycle."""
        from pyai.kernel import InvocationContext
        
        inv = InvocationContext(
            function_name="test",
            plugin_name="plugin"
        )
        
        inv.start()
        assert inv.start_time is not None
        
        inv.complete("result")
        assert inv.end_time is not None
        assert inv.result == "result"
        assert inv.success
        assert inv.duration_ms is not None
    
    def test_invocation_failure(self):
        """Test invocation failure."""
        from pyai.kernel import InvocationContext
        
        inv = InvocationContext()
        inv.start()
        
        error = ValueError("test")
        inv.fail(error)
        
        assert not inv.success
        assert inv.error is error
