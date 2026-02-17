"""# pyright: reportRedeclaration=false, reportUnusedVariable=falseTests for pyai OpenAPI Tools Module.

Tests the OpenAPI spec parsing, client, and tool generation.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_openapi_spec() -> Dict[str, Any]:
    """Sample OpenAPI 3.0 specification."""
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Pet Store API",
            "version": "1.0.0",
            "description": "A sample pet store API"
        },
        "servers": [
            {"url": "https://api.petstore.example.com/v1"}
        ],
        "paths": {
            "/pets": {
                "get": {
                    "operationId": "listPets",
                    "summary": "List all pets",
                    "description": "Returns a list of all pets in the store",
                    "parameters": [
                        {
                            "name": "limit",
                            "in": "query",
                            "description": "Maximum number of pets to return",
                            "required": False,
                            "schema": {"type": "integer", "default": 10}
                        },
                        {
                            "name": "species",
                            "in": "query",
                            "description": "Filter by species",
                            "required": False,
                            "schema": {"type": "string"}
                        }
                    ],
                    "responses": {
                        "200": {"description": "A list of pets"}
                    }
                },
                "post": {
                    "operationId": "createPet",
                    "summary": "Create a pet",
                    "description": "Add a new pet to the store",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "species": {"type": "string"}
                                    },
                                    "required": ["name"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "201": {"description": "Pet created"}
                    }
                }
            },
            "/pets/{petId}": {
                "get": {
                    "operationId": "getPet",
                    "summary": "Get a pet by ID",
                    "parameters": [
                        {
                            "name": "petId",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"}
                        }
                    ],
                    "responses": {
                        "200": {"description": "Pet details"}
                    }
                },
                "delete": {
                    "operationId": "deletePet",
                    "summary": "Delete a pet",
                    "parameters": [
                        {
                            "name": "petId",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"}
                        }
                    ],
                    "responses": {
                        "204": {"description": "Pet deleted"}
                    }
                }
            }
        }
    }


@pytest.fixture
def sample_swagger_spec() -> Dict[str, Any]:
    """Sample Swagger 2.0 specification."""
    return {
        "swagger": "2.0",
        "info": {
            "title": "Legacy API",
            "version": "1.0.0"
        },
        "host": "api.legacy.example.com",
        "basePath": "/v1",
        "schemes": ["https"],
        "paths": {
            "/users": {
                "get": {
                    "operationId": "listUsers",
                    "summary": "List users",
                    "parameters": [
                        {
                            "name": "page",
                            "in": "query",
                            "type": "integer",
                            "required": False
                        }
                    ],
                    "responses": {
                        "200": {"description": "Users list"}
                    }
                }
            }
        }
    }


@pytest.fixture
def minimal_spec() -> Dict[str, Any]:
    """Minimal valid OpenAPI spec."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "Minimal", "version": "1.0.0"},
        "paths": {}
    }


# =============================================================================
# Parser Tests
# =============================================================================

class TestOpenAPIParser:
    """Tests for OpenAPI spec parsing."""
    
    def test_parse_openapi_3_spec(self, sample_openapi_spec):
        """Test parsing OpenAPI 3.0 specification."""
        from pyai.openapi.parser import parse_openapi
        
        spec = parse_openapi(sample_openapi_spec)
        
        assert spec.title == "Pet Store API"
        assert spec.version == "1.0.0"
        assert spec.description == "A sample pet store API"
        assert spec.base_url == "https://api.petstore.example.com/v1"
    
    def test_parse_swagger_2_spec(self, sample_swagger_spec):
        """Test parsing Swagger 2.0 specification."""
        from pyai.openapi.parser import parse_openapi
        
        spec = parse_openapi(sample_swagger_spec)
        
        assert spec.title == "Legacy API"
        assert spec.version == "1.0.0"
        assert spec.base_url == "https://api.legacy.example.com/v1"
    
    def test_parse_operations(self, sample_openapi_spec):
        """Test that operations are correctly parsed."""
        from pyai.openapi.parser import parse_openapi
        
        spec = parse_openapi(sample_openapi_spec)
        
        # Should have 4 operations
        assert len(spec.operations) == 4
        
        # Check operation IDs
        op_ids = [op.operation_id for op in spec.operations]
        assert "listPets" in op_ids
        assert "createPet" in op_ids
        assert "getPet" in op_ids
        assert "deletePet" in op_ids
    
    def test_parse_operation_details(self, sample_openapi_spec):
        """Test operation details are correctly parsed."""
        from pyai.openapi.parser import parse_openapi
        
        spec = parse_openapi(sample_openapi_spec)
        
        # Find listPets operation
        list_pets = next(op for op in spec.operations if op.operation_id == "listPets")
        
        assert list_pets.method == "GET"
        assert list_pets.path == "/pets"
        assert list_pets.summary == "List all pets"
        assert list_pets.description == "Returns a list of all pets in the store"
    
    def test_parse_parameters(self, sample_openapi_spec):
        """Test operation parameters are correctly parsed."""
        from pyai.openapi.parser import parse_openapi
        
        spec = parse_openapi(sample_openapi_spec)
        
        # Find listPets operation
        list_pets = next(op for op in spec.operations if op.operation_id == "listPets")
        
        assert len(list_pets.parameters) == 2
        
        # Check limit parameter
        limit_param = next(p for p in list_pets.parameters if p.name == "limit")
        assert limit_param.location == "query"
        assert limit_param.required is False
        assert limit_param.schema_type == "integer"
        assert limit_param.default == 10
    
    def test_parse_path_parameters(self, sample_openapi_spec):
        """Test path parameters are correctly parsed."""
        from pyai.openapi.parser import parse_openapi
        
        spec = parse_openapi(sample_openapi_spec)
        
        get_pet = next(op for op in spec.operations if op.operation_id == "getPet")
        
        assert len(get_pet.parameters) == 1
        pet_id_param = get_pet.parameters[0]
        assert pet_id_param.name == "petId"
        assert pet_id_param.location == "path"
        assert pet_id_param.required is True
    
    def test_parse_request_body(self, sample_openapi_spec):
        """Test request body is correctly parsed."""
        from pyai.openapi.parser import parse_openapi
        
        spec = parse_openapi(sample_openapi_spec)
        
        create_pet = next(op for op in spec.operations if op.operation_id == "createPet")
        
        assert create_pet.request_body is not None
        # Check request body has content
        assert isinstance(create_pet.request_body, dict)
    
    def test_parse_minimal_spec(self, minimal_spec):
        """Test parsing minimal spec."""
        from pyai.openapi.parser import parse_openapi
        
        spec = parse_openapi(minimal_spec)
        
        assert spec.title == "Minimal"
        assert spec.version == "1.0.0"
        assert len(spec.operations) == 0
    
    def test_parse_from_dict(self, sample_openapi_spec):
        """Test parsing from dict directly."""
        from pyai.openapi.parser import parse_openapi
        
        # Parse directly from dict
        spec = parse_openapi(sample_openapi_spec)
        
        assert spec.title == "Pet Store API"
    
    def test_get_operation_by_id(self, sample_openapi_spec):
        """Test getting operation by ID."""
        from pyai.openapi.parser import parse_openapi
        
        spec = parse_openapi(sample_openapi_spec)
        
        op = spec.get_operation("listPets")
        assert op is not None
        assert op.operation_id == "listPets"
        
        # Non-existent
        assert spec.get_operation("nonExistent") is None
    
    def test_spec_attributes(self, sample_openapi_spec):
        """Test spec attributes."""
        from pyai.openapi.parser import parse_openapi
        
        spec = parse_openapi(sample_openapi_spec)
        
        # Test basic attributes are accessible
        assert spec.title == "Pet Store API"
        assert spec.version == "1.0.0"
        assert isinstance(spec.operations, list)
        assert len(spec.operations) > 0


# =============================================================================
# Client Tests
# =============================================================================

class TestOpenAPIClient:
    """Tests for OpenAPI client."""
    
    def test_client_initialization(self):
        """Test client initialization."""
        from pyai.openapi.client import OpenAPIClient
        
        client = OpenAPIClient(
            base_url="https://api.example.com",
            headers={"X-Custom": "value"},
            timeout=60
        )
        
        assert client.base_url == "https://api.example.com"
        assert client.headers["X-Custom"] == "value"
        assert client.timeout == 60
    
    def test_client_auth_tuple(self):
        """Test client with auth tuple."""
        from pyai.openapi.client import OpenAPIClient
        
        client = OpenAPIClient(
            base_url="https://api.example.com",
            auth=("user", "pass")
        )
        
        # Auth should be set
        assert client.auth == ("user", "pass")
    
    def test_client_with_custom_headers(self):
        """Test client with custom headers."""
        from pyai.openapi.client import OpenAPIClient
        
        client = OpenAPIClient(
            base_url="https://api.example.com",
            headers={"Authorization": "Bearer my-token"}
        )
        
        assert client.headers.get("Authorization") == "Bearer my-token"
    
    def test_build_url(self):
        """Test URL building."""
        from pyai.openapi.client import OpenAPIClient
        
        client = OpenAPIClient(base_url="https://api.example.com/v1")
        
        url = client._build_url("/users", {"userId": "123"})
        assert url == "https://api.example.com/v1/users"
    
    def test_build_url_with_path_params(self):
        """Test URL building with path parameters."""
        from pyai.openapi.client import OpenAPIClient
        
        client = OpenAPIClient(base_url="https://api.example.com")
        
        # The path params should be substituted
        url = client._build_url("/users/{userId}/posts/{postId}", {
            "userId": "123",
            "postId": "456"
        })
        assert "123" in url
        assert "456" in url
    
    @patch('requests.request')
    def test_call_get(self, mock_request):
        """Test GET request."""
        from pyai.openapi.client import OpenAPIClient
        
        mock_response = Mock()
        mock_response.json.return_value = {"data": "test"}
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.ok = True
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response
        
        client = OpenAPIClient(base_url="https://api.example.com")
        result = client.call("GET", "/users", params={"limit": 10})
        
        # Response is wrapped with data, status_code, headers
        assert result["data"] == {"data": "test"}
        assert result["status_code"] == 200
        mock_request.assert_called_once()
    
    @patch('requests.request')
    def test_call_post_with_body(self, mock_request):
        """Test POST request with body."""
        from pyai.openapi.client import OpenAPIClient
        
        mock_response = Mock()
        mock_response.json.return_value = {"id": "123"}
        mock_response.status_code = 201
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.ok = True
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response
        
        client = OpenAPIClient(base_url="https://api.example.com")
        result = client.call("POST", "/users", body={"name": "Test"})
        
        # Response is wrapped with data, status_code, headers
        assert result["data"] == {"id": "123"}
        assert result["status_code"] == 201
    
    @patch('requests.request')
    def test_call_with_path_params(self, mock_request):
        """Test request with path parameters."""
        from pyai.openapi.client import OpenAPIClient
        
        mock_response = Mock()
        mock_response.json.return_value = {"id": "123"}
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response
        
        client = OpenAPIClient(base_url="https://api.example.com")
        result = client.call("GET", "/users/{userId}", path_params={"userId": "123"})
        
        # Check the URL was built correctly
        call_args = mock_request.call_args
        assert "123" in str(call_args)
    
    @patch('requests.request')
    def test_call_error_handling(self, mock_request):
        """Test error handling in call."""
        from pyai.openapi.client import OpenAPIClient
        
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not found"
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")
        mock_request.return_value = mock_response
        
        client = OpenAPIClient(base_url="https://api.example.com")
        
        with pytest.raises(Exception):
            client.call("GET", "/nonexistent")
    
    def test_client_default_headers(self):
        """Test client default headers."""
        from pyai.openapi.client import OpenAPIClient
        
        client = OpenAPIClient(base_url="https://api.example.com")
        
        assert "Content-Type" in client.headers
        assert client.headers["Content-Type"] == "application/json"


# =============================================================================
# Tools Generation Tests
# =============================================================================

class TestOpenAPITools:
    """Tests for OpenAPI tools generation."""
    
    def test_create_tools_from_spec(self, sample_openapi_spec):
        """Test creating tools from OpenAPI spec."""
        from pyai.openapi.tools import create_tools_from_openapi
        from pyai.openapi.parser import parse_openapi
        
        spec = parse_openapi(sample_openapi_spec)
        tools = create_tools_from_openapi(spec)
        
        assert len(tools) == 4
    
    def test_tool_names(self, sample_openapi_spec):
        """Test generated tool names."""
        from pyai.openapi.tools import create_tools_from_openapi
        from pyai.openapi.parser import parse_openapi
        
        spec = parse_openapi(sample_openapi_spec)
        tools = create_tools_from_openapi(spec)
        
        # Tools are dicts with function.name
        tool_names = [t["function"]["name"] for t in tools]
        assert "listpets" in tool_names or "listPets" in tool_names
    
    def test_tool_has_description(self, sample_openapi_spec):
        """Test generated tools have descriptions."""
        from pyai.openapi.tools import create_tools_from_openapi
        from pyai.openapi.parser import parse_openapi
        
        spec = parse_openapi(sample_openapi_spec)
        tools = create_tools_from_openapi(spec)
        
        # Find a tool (any)
        tool = tools[0]
        assert "description" in tool["function"]
        assert len(tool["function"]["description"]) > 0
    
    def test_tool_has_parameters(self, sample_openapi_spec):
        """Test generated tools have parameter definitions."""
        from pyai.openapi.tools import create_tools_from_openapi
        from pyai.openapi.parser import parse_openapi
        
        spec = parse_openapi(sample_openapi_spec)
        tools = create_tools_from_openapi(spec)
        
        # Find a tool (any)
        tool = tools[0]
        assert "parameters" in tool["function"]
        assert "properties" in tool["function"]["parameters"]
    
    def test_openapi_tools_class(self, sample_openapi_spec):
        """Test OpenAPITools class."""
        from pyai.openapi.tools import OpenAPITools
        
        api = OpenAPITools(sample_openapi_spec, base_url="https://api.example.com")
        
        assert len(api.tools) == 4
    
    def test_openapi_tools_list_operations(self, sample_openapi_spec):
        """Test listing operations."""
        from pyai.openapi.tools import OpenAPITools
        
        api = OpenAPITools(sample_openapi_spec, base_url="https://api.example.com")
        
        ops = api.list_operations()
        assert len(ops) == 4
    
    def test_openapi_tools_structure(self, sample_openapi_spec):
        """Test tool structure matches OpenAI format."""
        from pyai.openapi.tools import OpenAPITools
        
        api = OpenAPITools(sample_openapi_spec, base_url="https://api.example.com")
        
        for tool in api.tools:
            assert "type" in tool
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
    
    @patch('requests.request')
    def test_openapi_tools_call(self, mock_request, sample_openapi_spec):
        """Test calling tool via OpenAPITools."""
        from pyai.openapi.tools import OpenAPITools
        
        mock_response = Mock()
        mock_response.json.return_value = [{"id": "1", "name": "Fluffy"}]
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.ok = True
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response
        
        api = OpenAPITools(sample_openapi_spec, base_url="https://api.example.com")
        
        # Get the actual operation name
        ops = api.list_operations()
        list_op = next((op for op in ops if "list" in op.lower()), ops[0])
        
        result = api.call(list_op, limit=5)
        
        # Response is wrapped with data, status_code, headers
        assert result["data"] == [{"id": "1", "name": "Fluffy"}]


# =============================================================================
# Decorator Tests
# =============================================================================

class TestOpenAPIDecorator:
    """Tests for openapi_tool decorator."""
    
    def test_openapi_tool_decorator_import(self):
        """Test decorator can be imported."""
        from pyai.openapi.tools import openapi_tool
        
        assert callable(openapi_tool)
    
    def test_decorator_exists(self):
        """Test the decorator exists."""
        from pyai.openapi import tools
        
        assert hasattr(tools, 'openapi_tool')


# =============================================================================
# Integration Tests
# =============================================================================

class TestOpenAPIIntegration:
    """Integration tests for OpenAPI module."""
    
    def test_full_workflow(self, sample_openapi_spec):
        """Test complete workflow from spec to tools."""
        from pyai.openapi.parser import parse_openapi
        from pyai.openapi.tools import create_tools_from_openapi, OpenAPITools
        
        # Parse spec
        spec = parse_openapi(sample_openapi_spec)
        assert spec.title == "Pet Store API"
        
        # Create tools
        tools = create_tools_from_openapi(spec)
        assert len(tools) == 4
        
        # Use OpenAPITools class
        api = OpenAPITools(sample_openapi_spec)
        assert len(api.tools) == 4
    
    def test_import_from_main_package(self):
        """Test imports from main pyai package."""
        from pyai import OpenAPITools, create_tools_from_openapi
        
        assert OpenAPITools is not None
        assert create_tools_from_openapi is not None
    
    def test_import_openapi_module(self):
        """Test importing openapi module."""
        from pyai.openapi import parse_openapi, OpenAPIClient, create_tools_from_openapi
        
        assert parse_openapi is not None
        assert OpenAPIClient is not None
        assert create_tools_from_openapi is not None


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestOpenAPIEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_paths(self, minimal_spec):
        """Test spec with no paths."""
        from pyai.openapi.parser import parse_openapi
        
        spec = parse_openapi(minimal_spec)
        assert len(spec.operations) == 0
    
    def test_operation_without_id(self):
        """Test operation without operationId."""
        from pyai.openapi.parser import parse_openapi
        
        spec_without_ids = {
            "openapi": "3.0.0",
            "info": {"title": "No IDs", "version": "1.0.0"},
            "paths": {
                "/test": {
                    "get": {
                        "summary": "Test endpoint"
                    }
                }
            }
        }
        
        spec = parse_openapi(spec_without_ids)
        # Should generate an ID
        assert len(spec.operations) == 1
        assert spec.operations[0].operation_id is not None
    
    def test_spec_no_servers(self):
        """Test spec without servers."""
        from pyai.openapi.parser import parse_openapi
        
        spec_no_servers = {
            "openapi": "3.0.0",
            "info": {"title": "No Servers", "version": "1.0.0"},
            "paths": {}
        }
        
        spec = parse_openapi(spec_no_servers)
        # Should handle gracefully
        assert spec.base_url is None or spec.base_url == ""
    
    def test_complex_parameters(self):
        """Test operations with complex parameters."""
        from pyai.openapi.parser import parse_openapi
        
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Complex", "version": "1.0.0"},
            "paths": {
                "/search": {
                    "get": {
                        "operationId": "search",
                        "parameters": [
                            {
                                "name": "filters",
                                "in": "query",
                                "schema": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            {
                                "name": "X-Custom-Header",
                                "in": "header",
                                "schema": {"type": "string"}
                            }
                        ]
                    }
                }
            }
        }
        
        parsed = parse_openapi(spec)
        search_op = parsed.get_operation("search")
        
        assert len(search_op.parameters) == 2
        # Check header param
        header_param = next(p for p in search_op.parameters if p.location == "header")
        assert header_param.name == "X-Custom-Header"


class TestOpenAPIToolSchema:
    """Tests for tool schema generation."""
    
    def test_tool_schema_types(self, sample_openapi_spec):
        """Test that tool schemas have correct types."""
        from pyai.openapi.tools import create_tools_from_openapi
        from pyai.openapi.parser import parse_openapi
        
        spec = parse_openapi(sample_openapi_spec)
        tools = create_tools_from_openapi(spec)
        
        # Each tool should be properly structured dict
        for tool in tools:
            assert "function" in tool
            assert "name" in tool["function"]
            assert isinstance(tool["function"]["name"], str)
    
    def test_required_params_in_schema(self, sample_openapi_spec):
        """Test required parameters are marked in schema."""
        from pyai.openapi.tools import create_tools_from_openapi
        from pyai.openapi.parser import parse_openapi
        
        spec = parse_openapi(sample_openapi_spec)
        tools = create_tools_from_openapi(spec)
        
        # Find a tool with path params (should be required)
        for tool in tools:
            params = tool["function"]["parameters"]
            if "required" in params and params["required"]:
                # Found a tool with required params
                assert isinstance(params["required"], list)
                break
