# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
PyAgent Development UI

Visual debugging and testing interface for agents.
Like Anthropic Workbench and OpenAI Playground.

Example:
    from pyagent import Agent
    from pyagent.devui import DevUI
    
    agent = Agent(model="gpt-4")
    
    # Launch web UI
    ui = DevUI(agent)
    ui.launch()  # Opens browser at http://localhost:7860
"""

from .ui import DevUI, launch_ui
from .dashboard import AgentDashboard
from .debugger import AgentDebugger

__all__ = [
    "DevUI",
    "launch_ui",
    "AgentDashboard",
    "AgentDebugger",
]
