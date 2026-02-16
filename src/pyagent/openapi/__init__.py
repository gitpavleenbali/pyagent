# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
OpenAPI Tools Module

Auto-generate tools from OpenAPI/Swagger specifications.
Similar to Google ADK's OpenAPI tools pattern.
"""

from .client import OpenAPIClient
from .parser import OpenAPISpec, parse_openapi
from .tools import create_tools_from_openapi, openapi_tool

__all__ = [
    "OpenAPIClient",
    "OpenAPISpec",
    "parse_openapi",
    "create_tools_from_openapi",
    "openapi_tool",
]
