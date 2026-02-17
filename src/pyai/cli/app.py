# Copyright (c) 2026 pyai Contributors
# Licensed under the MIT License

"""
pyai CLI Interactive App

Rich terminal interface for running agents.
"""

import sys
from typing import Any, Callable


class CLIApp:
    """Interactive CLI application for agents.

    Provides a rich terminal interface with:
    - Colorful output
    - History support
    - Command shortcuts

    Example:
        from pyai.cli import CLIApp
        from pyai import Agent

        app = CLIApp(Agent("helper"))
        app.run()
    """

    def __init__(self, agent: Any, name: str = "pyai", welcome_message: str = None):
        """Initialize CLI app.

        Args:
            agent: Agent to run
            name: App name for display
            welcome_message: Custom welcome message
        """
        self.agent = agent
        self.name = name
        self.welcome_message = (
            welcome_message or f"Welcome to {name}! Type 'help' for commands, 'exit' to quit."
        )
        self.history = []
        self.commands = {
            "help": self._cmd_help,
            "history": self._cmd_history,
            "clear": self._cmd_clear,
            "reset": self._cmd_reset,
            "exit": self._cmd_exit,
            "quit": self._cmd_exit,
            "q": self._cmd_exit,
        }

    def _print_styled(self, text: str, style: str = None):
        """Print with optional styling."""
        prefix = ""
        suffix = ""

        # Try to use rich for colored output
        try:
            from rich.console import Console

            console = Console()
            console.print(text, style=style)
            return
        except ImportError:
            pass

        # Fallback to basic ANSI colors
        colors = {
            "bold": "\033[1m",
            "green": "\033[92m",
            "blue": "\033[94m",
            "yellow": "\033[93m",
            "red": "\033[91m",
            "cyan": "\033[96m",
        }
        reset = "\033[0m"

        if style and style in colors:
            prefix = colors[style]
            suffix = reset

        print(f"{prefix}{text}{suffix}")

    def _cmd_help(self) -> str:
        """Show help message."""
        return """
Available commands:
  help     - Show this help message
  history  - Show conversation history
  clear    - Clear the screen
  reset    - Reset conversation
  exit     - Exit the app

Just type your message to chat with the agent.
"""

    def _cmd_history(self) -> str:
        """Show conversation history."""
        if not self.history:
            return "No conversation history."

        result = []
        for i, (user, agent) in enumerate(self.history, 1):
            result.append(f"[{i}] You: {user[:50]}...")
            result.append(f"    Agent: {agent[:50]}...")
        return "\n".join(result)

    def _cmd_clear(self) -> str:
        """Clear screen."""
        import os

        os.system("cls" if sys.platform == "win32" else "clear")
        return ""

    def _cmd_reset(self) -> str:
        """Reset conversation."""
        self.history = []
        if hasattr(self.agent, "reset"):
            self.agent.reset()
        return "Conversation reset."

    def _cmd_exit(self) -> None:
        """Exit the app."""
        self._print_styled("👋 Goodbye!", "cyan")
        sys.exit(0)

    def _get_input(self) -> str:
        """Get user input with prompt."""
        try:
            # Try readline for history support
            import readline
        except ImportError:
            pass

        try:
            return input("\n🧑 You: ").strip()
        except EOFError:
            return "exit"

    def _run_agent(self, message: str) -> str:
        """Run the agent and get response."""
        try:
            if hasattr(self.agent, "run"):
                result = self.agent.run(message)
                return result.output if hasattr(result, "output") else str(result)
            elif hasattr(self.agent, "invoke"):
                result = self.agent.invoke({"input": message})
                return result.get("output", str(result))
            elif callable(self.agent):
                result = self.agent(message)
                return str(result) if not isinstance(result, str) else result
            else:
                return f"Error: Don't know how to run agent of type {type(self.agent)}"
        except Exception as e:
            return f"Error: {e}"

    def run(self):
        """Start the interactive CLI."""
        self._print_styled(f"\n{'=' * 50}", "blue")
        self._print_styled(f"  {self.name}", "bold")
        self._print_styled(f"{'=' * 50}\n", "blue")
        self._print_styled(self.welcome_message, "cyan")

        while True:
            try:
                user_input = self._get_input()

                if not user_input:
                    continue

                # Check for commands
                if user_input.lower() in self.commands:
                    result = self.commands[user_input.lower()]()
                    if result:
                        print(result)
                    continue

                # Run agent
                response = self._run_agent(user_input)

                # Save to history
                self.history.append((user_input, response))

                # Display response
                self._print_styled(f"\n🤖 Agent: {response}", "green")

            except KeyboardInterrupt:
                print("\n")
                self._cmd_exit()

    def add_command(self, name: str, func: Callable[[], str]):
        """Add a custom command.

        Args:
            name: Command name (e.g., "status")
            func: Function that returns output string
        """
        self.commands[name.lower()] = func


def run_cli(agent: Any, **kwargs):
    """Convenience function to start CLI.

    Example:
        from pyai.cli import run_cli
        from pyai import Agent

        run_cli(Agent("helper"))
    """
    app = CLIApp(agent, **kwargs)
    app.run()
