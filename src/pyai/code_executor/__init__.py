# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
pyai Code Executor Module

Provides safe code execution capabilities for agents.

Inspired by Google ADK's code_executors/ module.

Features:
- Safe Python code execution in sandbox
- Resource limits (time, memory)
- Capture stdout/stderr
- Support for common libraries (numpy, pandas, etc.)

Example:
    from pyai.code_executor import execute_python, CodeExecutor

    # Simple execution
    result = execute_python("print('Hello')")
    print(result.output)  # "Hello"

    # With executor
    executor = CodeExecutor(timeout=10)
    result = executor.execute("x = 2 + 2\\nprint(x)")
"""


# Lazy imports
def __getattr__(name):
    """Lazy load code executor components."""
    _exports = {
        "CodeExecutor",
        "ExecutionResult",
        "ExecutionError",
        "execute_python",
        "execute_shell",
        "PythonExecutor",
        "DockerExecutor",
        "SafeExecutor",
    }

    if name in _exports:
        from . import executor

        return getattr(executor, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CodeExecutor",
    "ExecutionResult",
    "ExecutionError",
    "execute_python",
    "execute_shell",
    "PythonExecutor",
    "DockerExecutor",
    "SafeExecutor",
]
