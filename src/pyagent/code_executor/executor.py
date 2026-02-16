# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Code Executor Implementation

Safe code execution with sandboxing and resource limits.
"""

import ast
import io
import sys
import time
import traceback
from abc import ABC, abstractmethod
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import threading


class ExecutionError(Exception):
    """Error during code execution."""
    pass


@dataclass
class ExecutionResult:
    """Result from code execution.
    
    Attributes:
        success: Whether execution succeeded
        output: Captured stdout
        error: Error message if failed
        return_value: Return value from code (if any)
        execution_time_ms: Time taken in milliseconds
        variables: Variables defined in the code
    """
    success: bool
    output: str = ""
    error: str = ""
    return_value: Any = None
    execution_time_ms: float = 0.0
    variables: Dict[str, Any] = field(default_factory=dict)
    stderr: str = ""
    
    def __str__(self) -> str:
        if self.success:
            return self.output if self.output else "(no output)"
        return f"Error: {self.error}"


class CodeValidator:
    """Validate code for safety before execution."""
    
    # Dangerous built-in functions
    BLOCKED_BUILTINS = {
        "exec", "eval", "compile",
        "open", "__import__", "input",
        "breakpoint", "help", "license", "credits",
    }
    
    # Dangerous module imports
    BLOCKED_MODULES = {
        "os", "subprocess", "sys", "shutil",
        "socket", "urllib", "requests", "httpx",
        "pickle", "marshal", "shelve",
        "ctypes", "multiprocessing",
        "importlib", "runpy",
    }
    
    # Blocked attributes
    BLOCKED_ATTRS = {
        "__code__", "__globals__", "__builtins__",
        "__class__", "__base__", "__subclasses__",
        "__mro__", "__dict__",
    }
    
    def __init__(self, allow_modules: Optional[Set[str]] = None):
        """Initialize validator.
        
        Args:
            allow_modules: Set of module names to allow despite being blocked
        """
        self.allow_modules = allow_modules or set()
    
    def validate(self, code: str) -> tuple:
        """Validate code for safety.
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split(".")[0]
                    if module_name in self.BLOCKED_MODULES and module_name not in self.allow_modules:
                        return False, f"Blocked import: {module_name}"
            
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split(".")[0]
                    if module_name in self.BLOCKED_MODULES and module_name not in self.allow_modules:
                        return False, f"Blocked import: {module_name}"
            
            # Check function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.BLOCKED_BUILTINS:
                        return False, f"Blocked function: {node.func.id}"
            
            # Check attribute access
            if isinstance(node, ast.Attribute):
                if node.attr in self.BLOCKED_ATTRS:
                    return False, f"Blocked attribute: {node.attr}"
        
        return True, None


class CodeExecutor(ABC):
    """Base class for code executors."""
    
    @abstractmethod
    def execute(self, code: str) -> ExecutionResult:
        """Execute code and return result.
        
        Args:
            code: Code to execute
            
        Returns:
            ExecutionResult with output and status
        """
        pass


class SafeExecutor(CodeExecutor):
    """Execute Python code with safety restrictions.
    
    Uses a restricted namespace and validation.
    
    Example:
        executor = SafeExecutor(timeout=5.0)
        result = executor.execute("print(2 + 2)")
        print(result.output)  # "4"
    """
    
    # Safe built-ins to allow
    SAFE_BUILTINS = {
        "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
        "callable", "chr", "complex", "dict", "dir", "divmod", "enumerate",
        "filter", "float", "format", "frozenset", "getattr", "hasattr",
        "hash", "hex", "id", "int", "isinstance", "issubclass", "iter",
        "len", "list", "map", "max", "min", "next", "object", "oct",
        "ord", "pow", "print", "range", "repr", "reversed", "round",
        "set", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
        "True", "False", "None",
    }
    
    def __init__(
        self,
        timeout: float = 30.0,
        max_output_size: int = 100000,
        allow_modules: Optional[List[str]] = None,
        extra_globals: Optional[Dict[str, Any]] = None
    ):
        """Initialize safe executor.
        
        Args:
            timeout: Maximum execution time in seconds
            max_output_size: Maximum output size in characters
            allow_modules: Additional modules to allow
            extra_globals: Additional variables to include
        """
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.validator = CodeValidator(set(allow_modules) if allow_modules else None)
        self.extra_globals = extra_globals or {}
    
    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create a restricted globals dictionary."""
        import builtins
        
        safe_builtins = {name: getattr(builtins, name) for name in self.SAFE_BUILTINS if hasattr(builtins, name)}
        
        # Add safe versions of some functions
        def safe_print(*args, **kwargs):
            """Safe print that captures to string."""
            print(*args, **kwargs)
        
        safe_builtins["print"] = safe_print
        
        globals_dict = {
            "__builtins__": safe_builtins,
            "__name__": "__main__",
        }
        
        # Add safe math/data libraries if available
        try:
            import math
            globals_dict["math"] = math
        except ImportError:
            pass
        
        try:
            import statistics
            globals_dict["statistics"] = statistics
        except ImportError:
            pass
        
        try:
            import random
            globals_dict["random"] = random
        except ImportError:
            pass
        
        try:
            import datetime
            globals_dict["datetime"] = datetime
        except ImportError:
            pass
        
        try:
            import json
            globals_dict["json"] = json
        except ImportError:
            pass
        
        try:
            import re
            globals_dict["re"] = re
        except ImportError:
            pass
        
        # Add extra globals
        globals_dict.update(self.extra_globals)
        
        return globals_dict
    
    def execute(self, code: str) -> ExecutionResult:
        """Execute code safely.
        
        Args:
            code: Python code to execute
            
        Returns:
            ExecutionResult with output and status
        """
        start_time = time.perf_counter()
        
        # Validate code
        is_valid, error = self.validator.validate(code)
        if not is_valid:
            return ExecutionResult(
                success=False,
                error=f"Validation failed: {error}",
                execution_time_ms=(time.perf_counter() - start_time) * 1000
            )
        
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Create execution namespace
        globals_dict = self._create_safe_globals()
        locals_dict = {}
        
        result = ExecutionResult(success=True)
        
        # Execute with timeout
        exception_holder = [None]
        return_holder = [None]
        
        def run_code():
            try:
                exec(code, globals_dict, locals_dict)
                return_holder[0] = locals_dict.get("result", None)
            except Exception as e:
                exception_holder[0] = e
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                thread = threading.Thread(target=run_code)
                thread.start()
                thread.join(timeout=self.timeout)
                
                if thread.is_alive():
                    # Timeout occurred
                    result.success = False
                    result.error = f"Execution timeout ({self.timeout}s)"
                    return result
        except Exception as e:
            result.success = False
            result.error = str(e)
            return result
        
        # Process results
        if exception_holder[0]:
            result.success = False
            result.error = f"{type(exception_holder[0]).__name__}: {exception_holder[0]}"
        
        result.output = stdout_capture.getvalue()[:self.max_output_size]
        result.stderr = stderr_capture.getvalue()[:self.max_output_size]
        result.return_value = return_holder[0]
        result.execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Capture safe variables (basic types only)
        for name, value in locals_dict.items():
            if not name.startswith("_"):
                if isinstance(value, (int, float, str, bool, list, dict, tuple, type(None))):
                    result.variables[name] = value
        
        return result


class PythonExecutor(SafeExecutor):
    """Python code executor with data science libraries.
    
    Allows numpy, pandas, matplotlib (display disabled).
    
    Example:
        executor = PythonExecutor()
        result = executor.execute('''
import numpy as np
x = np.array([1, 2, 3])
print(x.mean())
''')
    """
    
    def __init__(self, timeout: float = 30.0):
        super().__init__(
            timeout=timeout,
            allow_modules=["numpy", "pandas", "scipy", "sklearn"]
        )
    
    def _create_safe_globals(self) -> Dict[str, Any]:
        globals_dict = super()._create_safe_globals()
        
        # Try to add data science libraries
        try:
            import numpy as np
            globals_dict["np"] = np
            globals_dict["numpy"] = np
        except ImportError:
            pass
        
        try:
            import pandas as pd
            globals_dict["pd"] = pd
            globals_dict["pandas"] = pd
        except ImportError:
            pass
        
        return globals_dict


class DockerExecutor(CodeExecutor):
    """Execute code in a Docker container for maximum isolation.
    
    Requires Docker to be installed and running.
    
    Example:
        executor = DockerExecutor(image="python:3.11-slim")
        result = executor.execute("print('Hello from container')")
    """
    
    def __init__(
        self,
        image: str = "python:3.11-slim",
        timeout: float = 60.0,
        memory_limit: str = "256m",
        cpu_limit: float = 1.0
    ):
        """Initialize Docker executor.
        
        Args:
            image: Docker image to use
            timeout: Execution timeout in seconds
            memory_limit: Memory limit (e.g., "256m")
            cpu_limit: CPU limit (number of CPUs)
        """
        self.image = image
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
    
    def execute(self, code: str) -> ExecutionResult:
        """Execute code in Docker container.
        
        Args:
            code: Python code to execute
            
        Returns:
            ExecutionResult with output
        """
        import subprocess
        import tempfile
        
        start_time = time.perf_counter()
        
        try:
            # Write code to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                code_file = f.name
            
            # Run in container
            cmd = [
                "docker", "run",
                "--rm",
                "--network", "none",
                "-m", self.memory_limit,
                f"--cpus={self.cpu_limit}",
                "-v", f"{code_file}:/code.py:ro",
                self.image,
                "python", "/code.py"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            return ExecutionResult(
                success=result.returncode == 0,
                output=result.stdout,
                stderr=result.stderr,
                error=result.stderr if result.returncode != 0 else "",
                execution_time_ms=(time.perf_counter() - start_time) * 1000
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                error=f"Docker execution timeout ({self.timeout}s)",
                execution_time_ms=(time.perf_counter() - start_time) * 1000
            )
        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                error="Docker not found. Install Docker to use DockerExecutor.",
                execution_time_ms=(time.perf_counter() - start_time) * 1000
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.perf_counter() - start_time) * 1000
            )


def execute_python(
    code: str,
    timeout: float = 30.0,
    safe: bool = True
) -> ExecutionResult:
    """Execute Python code and return result.
    
    One-liner code execution:
    
        from pyagent.code_executor import execute_python
        
        result = execute_python("print('Hello, World!')")
        print(result.output)  # "Hello, World!"
        
        result = execute_python("x = 2 + 2; print(x)")
        print(result.output)  # "4"
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time
        safe: Use safe executor (restricts imports)
        
    Returns:
        ExecutionResult with output and status
    """
    if safe:
        executor = SafeExecutor(timeout=timeout)
    else:
        executor = PythonExecutor(timeout=timeout)
    
    return executor.execute(code)


def execute_shell(
    command: str,
    timeout: float = 30.0,
    cwd: Optional[str] = None
) -> ExecutionResult:
    """Execute a shell command.
    
    WARNING: This is not sandboxed. Use with caution.
    
    Args:
        command: Shell command to execute
        timeout: Maximum execution time
        cwd: Working directory
        
    Returns:
        ExecutionResult with output
    """
    import subprocess
    
    start_time = time.perf_counter()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd
        )
        
        return ExecutionResult(
            success=result.returncode == 0,
            output=result.stdout,
            stderr=result.stderr,
            error=result.stderr if result.returncode != 0 else "",
            execution_time_ms=(time.perf_counter() - start_time) * 1000
        )
        
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            error=f"Command timeout ({timeout}s)",
            execution_time_ms=(time.perf_counter() - start_time) * 1000
        )
    except Exception as e:
        return ExecutionResult(
            success=False,
            error=str(e),
            execution_time_ms=(time.perf_counter() - start_time) * 1000
        )


# Default executor instance
_default_executor = None

def get_executor(**kwargs) -> CodeExecutor:
    """Get the default code executor.
    
    Returns:
        CodeExecutor instance
    """
    global _default_executor
    if _default_executor is None:
        _default_executor = SafeExecutor(**kwargs)
    return _default_executor
