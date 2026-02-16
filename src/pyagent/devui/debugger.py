# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Agent Debugger

Step-through debugging and introspection for agents.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import threading


class DebugEvent(Enum):
    """Types of debug events."""
    RUN_START = "run_start"
    RUN_END = "run_end"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    ERROR = "error"
    BREAKPOINT = "breakpoint"


@dataclass
class DebugEntry:
    """A debug log entry."""
    event: DebugEvent
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None


@dataclass
class Breakpoint:
    """A debugging breakpoint."""
    id: str
    condition: str  # e.g., "tool_call", "error", "custom"
    handler: Optional[Callable] = None
    enabled: bool = True
    hit_count: int = 0


class AgentDebugger:
    """Interactive debugger for agents.
    
    Provides step-through debugging, breakpoints, and introspection.
    
    Example:
        from pyagent.devui import AgentDebugger
        
        debugger = AgentDebugger()
        
        # Add breakpoint
        debugger.add_breakpoint("tool_call")
        
        # Wrap agent
        wrapped = debugger.wrap(agent)
        
        # Run with debugging
        debugger.start()
        result = wrapped.run("Hello")
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize debugger.
        
        Args:
            verbose: Print debug events to console
        """
        self.verbose = verbose
        
        self._entries: List[DebugEntry] = []
        self._breakpoints: Dict[str, Breakpoint] = {}
        self._paused = False
        self._step_mode = False
        self._continue_event = threading.Event()
        self._callbacks: List[Callable[[DebugEntry], None]] = []
    
    def log(
        self,
        event: DebugEvent,
        data: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> DebugEntry:
        """Log a debug event.
        
        Args:
            event: Event type
            data: Event data
            parent_id: Parent entry ID
            
        Returns:
            Debug entry
        """
        entry = DebugEntry(
            event=event,
            data=data or {},
            parent_id=parent_id,
        )
        
        self._entries.append(entry)
        
        if self.verbose:
            self._print_entry(entry)
        
        # Call registered callbacks
        for callback in self._callbacks:
            try:
                callback(entry)
            except:
                pass
        
        # Check breakpoints
        self._check_breakpoints(entry)
        
        return entry
    
    def _print_entry(self, entry: DebugEntry):
        """Print entry to console."""
        timestamp = entry.timestamp.strftime("%H:%M:%S.%f")[:-3]
        event_name = entry.event.value
        
        # Color codes
        colors = {
            DebugEvent.RUN_START: "\033[92m",  # Green
            DebugEvent.RUN_END: "\033[92m",
            DebugEvent.TOOL_CALL: "\033[94m",  # Blue
            DebugEvent.TOOL_RESULT: "\033[96m",  # Cyan
            DebugEvent.LLM_REQUEST: "\033[93m",  # Yellow
            DebugEvent.LLM_RESPONSE: "\033[93m",
            DebugEvent.ERROR: "\033[91m",  # Red
            DebugEvent.BREAKPOINT: "\033[95m",  # Magenta
        }
        reset = "\033[0m"
        color = colors.get(entry.event, "")
        
        print(f"{color}[{timestamp}] {event_name}{reset}")
        
        if entry.data:
            for key, value in entry.data.items():
                value_str = str(value)[:100]
                print(f"  {key}: {value_str}")
    
    def _check_breakpoints(self, entry: DebugEntry):
        """Check and handle breakpoints."""
        for bp in self._breakpoints.values():
            if not bp.enabled:
                continue
            
            should_break = False
            
            if bp.condition == entry.event.value:
                should_break = True
            elif bp.condition == "all":
                should_break = True
            elif bp.condition == "error" and entry.event == DebugEvent.ERROR:
                should_break = True
            
            if should_break:
                bp.hit_count += 1
                
                if bp.handler:
                    bp.handler(entry)
                
                if self._step_mode:
                    self._pause()
    
    def _pause(self):
        """Pause execution."""
        self._paused = True
        print("\n🔴 Debugger paused. Commands: continue (c), step (s), quit (q)")
        
        while self._paused:
            try:
                cmd = input("(debug) > ").strip().lower()
                
                if cmd in ("c", "continue"):
                    self._paused = False
                    self._step_mode = False
                elif cmd in ("s", "step"):
                    self._paused = False
                    self._step_mode = True
                elif cmd in ("q", "quit"):
                    self._paused = False
                    raise KeyboardInterrupt("Debug session ended")
                elif cmd in ("v", "vars"):
                    self._show_state()
                elif cmd in ("h", "help"):
                    print("  c/continue - Continue execution")
                    print("  s/step     - Step to next event")
                    print("  v/vars     - Show current state")
                    print("  q/quit     - Stop debugging")
                else:
                    print(f"Unknown command: {cmd}")
            except EOFError:
                self._paused = False
    
    def _show_state(self):
        """Show current debugger state."""
        print(f"\nEntries: {len(self._entries)}")
        print(f"Breakpoints: {len(self._breakpoints)}")
        
        if self._entries:
            last = self._entries[-1]
            print(f"\nLast event: {last.event.value}")
            for k, v in last.data.items():
                print(f"  {k}: {str(v)[:50]}")
    
    def add_breakpoint(
        self,
        condition: str,
        handler: Optional[Callable] = None,
    ) -> str:
        """Add a breakpoint.
        
        Args:
            condition: Breakpoint condition (event name or "all")
            handler: Optional handler function
            
        Returns:
            Breakpoint ID
        """
        import uuid
        
        bp_id = str(uuid.uuid4())[:8]
        self._breakpoints[bp_id] = Breakpoint(
            id=bp_id,
            condition=condition,
            handler=handler,
        )
        return bp_id
    
    def remove_breakpoint(self, bp_id: str) -> bool:
        """Remove a breakpoint."""
        if bp_id in self._breakpoints:
            del self._breakpoints[bp_id]
            return True
        return False
    
    def clear_breakpoints(self):
        """Clear all breakpoints."""
        self._breakpoints.clear()
    
    def start(self, step: bool = False):
        """Start debugging.
        
        Args:
            step: Start in step mode
        """
        self._step_mode = step
        if self.verbose:
            print("🔍 Debugger started")
    
    def stop(self):
        """Stop debugging."""
        self._step_mode = False
        self._paused = False
        if self.verbose:
            print("🔍 Debugger stopped")
    
    def on_event(self, callback: Callable[[DebugEntry], None]):
        """Register event callback.
        
        Args:
            callback: Function called for each event
        """
        self._callbacks.append(callback)
    
    def get_entries(
        self,
        event_type: Optional[DebugEvent] = None,
        limit: int = 100,
    ) -> List[DebugEntry]:
        """Get debug entries.
        
        Args:
            event_type: Filter by event type
            limit: Maximum entries to return
            
        Returns:
            List of debug entries
        """
        entries = self._entries
        if event_type:
            entries = [e for e in entries if e.event == event_type]
        return entries[-limit:]
    
    def clear(self):
        """Clear all entries."""
        self._entries.clear()
    
    def wrap(self, agent: Any) -> "DebuggedAgent":
        """Wrap an agent for debugging.
        
        Args:
            agent: Agent to wrap
            
        Returns:
            Debugged agent wrapper
        """
        return DebuggedAgent(agent, self)


class DebuggedAgent:
    """Agent wrapper with debug instrumentation."""
    
    def __init__(self, agent: Any, debugger: AgentDebugger):
        self._agent = agent
        self._debugger = debugger
    
    def run(self, input: str, **kwargs) -> Any:
        """Run agent with debugging."""
        import uuid
        
        run_id = str(uuid.uuid4())[:8]
        
        self._debugger.log(
            DebugEvent.RUN_START,
            {"run_id": run_id, "input": input[:200]},
        )
        
        try:
            result = self._agent.run(input, **kwargs)
            
            # Extract output
            if hasattr(result, "output"):
                output = result.output
            elif isinstance(result, str):
                output = result
            else:
                output = str(result)
            
            self._debugger.log(
                DebugEvent.RUN_END,
                {"run_id": run_id, "output": output[:200], "success": True},
            )
            
            return result
            
        except Exception as e:
            self._debugger.log(
                DebugEvent.ERROR,
                {"run_id": run_id, "error": str(e)},
            )
            raise
    
    def __getattr__(self, name):
        """Forward to wrapped agent."""
        return getattr(self._agent, name)
