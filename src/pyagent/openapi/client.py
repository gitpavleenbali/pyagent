# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
OpenAPI Client

HTTP client for calling OpenAPI endpoints.
"""

import json
from typing import Any, Dict, Optional, Union
from urllib.parse import urljoin, urlencode

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


class OpenAPIClient:
    """HTTP client for OpenAPI endpoints.
    
    Handles authentication, request formatting, and response parsing.
    
    Example:
        client = OpenAPIClient(
            base_url="https://api.example.com",
            headers={"Authorization": "Bearer token123"}
        )
        
        result = client.call("GET", "/users", params={"limit": 10})
    """
    
    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        auth: Optional[tuple] = None,
        timeout: float = 30.0
    ):
        """Initialize API client.
        
        Args:
            base_url: Base URL for all requests
            headers: Default headers to include
            auth: Basic auth credentials (username, password)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.auth = auth
        self.timeout = timeout
        
        # Set default content type
        if "Content-Type" not in self.headers:
            self.headers["Content-Type"] = "application/json"
        if "Accept" not in self.headers:
            self.headers["Accept"] = "application/json"
    
    def call(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        path_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make a synchronous API call.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., "/users/{id}")
            params: Query parameters
            body: Request body (for POST, PUT, PATCH)
            headers: Additional headers
            path_params: Path parameters to substitute
            
        Returns:
            Response data as dictionary
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests library required. Install with: pip install requests"
            )
        
        # Build URL
        url = self._build_url(path, path_params)
        
        # Merge headers
        request_headers = {**self.headers, **(headers or {})}
        
        # Make request
        response = requests.request(
            method=method.upper(),
            url=url,
            params=params,
            json=body if body else None,
            headers=request_headers,
            auth=self.auth,
            timeout=self.timeout
        )
        
        # Parse response
        return self._parse_response(response)
    
    async def call_async(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        path_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make an async API call.
        
        Args:
            method: HTTP method
            path: API path
            params: Query parameters
            body: Request body
            headers: Additional headers
            path_params: Path parameters
            
        Returns:
            Response data
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp library required. Install with: pip install aiohttp"
            )
        
        url = self._build_url(path, path_params)
        request_headers = {**self.headers, **(headers or {})}
        
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=method.upper(),
                url=url,
                params=params,
                json=body,
                headers=request_headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                return await self._parse_async_response(response)
    
    def _build_url(
        self,
        path: str,
        path_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build full URL with path parameter substitution."""
        # Substitute path parameters
        if path_params:
            for key, value in path_params.items():
                path = path.replace(f"{{{key}}}", str(value))
        
        # Join with base URL
        return urljoin(self.base_url + "/", path.lstrip("/"))
    
    def _parse_response(self, response) -> Dict[str, Any]:
        """Parse sync response."""
        result = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }
        
        try:
            result["data"] = response.json()
        except json.JSONDecodeError:
            result["data"] = response.text
        
        if not response.ok:
            result["error"] = f"HTTP {response.status_code}: {response.reason}"
        
        return result
    
    async def _parse_async_response(self, response) -> Dict[str, Any]:
        """Parse async response."""
        result = {
            "status_code": response.status,
            "headers": dict(response.headers),
        }
        
        try:
            result["data"] = await response.json()
        except (json.JSONDecodeError, aiohttp.ContentTypeError):
            result["data"] = await response.text()
        
        if not response.ok:
            result["error"] = f"HTTP {response.status}"
        
        return result
    
    def set_bearer_token(self, token: str) -> None:
        """Set Bearer token authentication."""
        self.headers["Authorization"] = f"Bearer {token}"
    
    def set_api_key(self, key: str, header_name: str = "X-API-Key") -> None:
        """Set API key authentication."""
        self.headers[header_name] = key
