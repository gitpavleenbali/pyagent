# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
pyai Development UI

Visual debugging and testing interface for agents.
Like Anthropic Workbench and OpenAI Playground.

Example:
    from pyai import Agent
    from pyai.devui import DevUI

    agent = Agent(model="gpt-4")

    # Launch web UI
    ui = DevUI(agent)
    ui.launch()  # Opens browser at http://localhost:7860
"""

from .dashboard import AgentDashboard
from .debugger import AgentDebugger
from .ui import DevUI, launch_ui

__all__ = [
    "DevUI",
    "launch_ui",
    "AgentDashboard",
    "AgentDebugger",
]
