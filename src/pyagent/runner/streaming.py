# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Streaming Runner

Support for streaming agent responses with events.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Union
from enum import Enum


class EventType(Enum):
    """Types of streaming events."""
    RUN_START = "run_start"
    TURN_START = "turn_start"
    TOKEN = "token"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TURN_END = "turn_end"
    RUN_END = "run_end"
    ERROR = "error"


@dataclass
class StreamEvent:
    """An event emitted during streaming execution.
    
    Attributes:
        type: Event type
        data: Event data
        timestamp: Event timestamp
        run_id: Associated run ID
        turn_number: Current turn number
    """
    type: EventType
    data: Any = None
    timestamp: float = field(default_factory=time.time)
    run_id: Optional[str] = None
    turn_number: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "turn_number": self.turn_number,
        }


class StreamingRunner:
    """Runner with streaming support.
    
    Yields events as the agent executes, enabling real-time updates.
    
    Example:
        # Streaming execution
        async for event in StreamingRunner.stream(agent, "Write a poem"):
            if event.type == EventType.TOKEN:
                print(event.data, end="", flush=True)
            elif event.type == EventType.RUN_END:
                print(f"\\nCompleted in {event.data['elapsed_time']:.2f}s")
    """
    
    def __init__(
        self,
        max_turns: int = 10,
        max_time: float = 300.0,
        verbose: bool = False
    ):
        """Initialize streaming runner.
        
        Args:
            max_turns: Maximum conversation turns
            max_time: Maximum execution time in seconds
            verbose: Enable verbose output
        """
        self.max_turns = max_turns
        self.max_time = max_time
        self.verbose = verbose
    
    async def execute_stream(
        self,
        agent: Any,
        input: str,
        **kwargs
    ) -> AsyncIterator[StreamEvent]:
        """Execute agent with streaming events.
        
        Args:
            agent: The agent to run
            input: User input
            **kwargs: Additional arguments
            
        Yields:
            StreamEvent objects as execution progresses
        """
        import uuid
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        start_time = time.time()
        
        # Emit run start
        yield StreamEvent(
            type=EventType.RUN_START,
            data={"input": input},
            run_id=run_id,
        )
        
        try:
            messages = [{"role": "user", "content": input}]
            final_output = None
            
            for turn in range(self.max_turns):
                turn_number = turn + 1
                
                # Check time limit
                elapsed = time.time() - start_time
                if elapsed > self.max_time:
                    yield StreamEvent(
                        type=EventType.ERROR,
                        data={"error": f"Time limit exceeded ({self.max_time}s)"},
                        run_id=run_id,
                        turn_number=turn_number,
                    )
                    break
                
                # Emit turn start
                yield StreamEvent(
                    type=EventType.TURN_START,
                    data={"turn": turn_number},
                    run_id=run_id,
                    turn_number=turn_number,
                )
                
                # Execute agent turn with streaming if supported
                response = None
                
                if hasattr(agent, "stream"):
                    # Agent supports streaming
                    async for chunk in self._stream_agent(agent, messages, **kwargs):
                        if isinstance(chunk, str):
                            yield StreamEvent(
                                type=EventType.TOKEN,
                                data=chunk,
                                run_id=run_id,
                                turn_number=turn_number,
                            )
                            if response is None:
                                response = chunk
                            else:
                                response += chunk
                        elif isinstance(chunk, dict) and "tool_calls" in chunk:
                            yield StreamEvent(
                                type=EventType.TOOL_CALL,
                                data=chunk["tool_calls"],
                                run_id=run_id,
                                turn_number=turn_number,
                            )
                else:
                    # Fall back to non-streaming
                    response = await self._run_agent_async(agent, messages, **kwargs)
                    if isinstance(response, str):
                        yield StreamEvent(
                            type=EventType.TOKEN,
                            data=response,
                            run_id=run_id,
                            turn_number=turn_number,
                        )
                
                # Emit turn end
                yield StreamEvent(
                    type=EventType.TURN_END,
                    data={"response": response[:100] if isinstance(response, str) else str(response)[:100]},
                    run_id=run_id,
                    turn_number=turn_number,
                )
                
                if isinstance(response, str):
                    final_output = response
                    break
                elif isinstance(response, dict) and "content" in response:
                    final_output = response["content"]
                    break
            
            # Emit run end
            yield StreamEvent(
                type=EventType.RUN_END,
                data={
                    "output": final_output,
                    "elapsed_time": time.time() - start_time,
                    "turn_count": turn_number,
                },
                run_id=run_id,
                turn_number=turn_number,
            )
            
        except Exception as e:
            yield StreamEvent(
                type=EventType.ERROR,
                data={"error": str(e)},
                run_id=run_id,
            )
    
    async def _stream_agent(
        self,
        agent: Any,
        messages: List[Dict],
        **kwargs
    ) -> AsyncIterator[Union[str, Dict]]:
        """Stream from agent if supported."""
        if hasattr(agent, "stream_async"):
            async for chunk in agent.stream_async(messages[-1]["content"], **kwargs):
                yield chunk
        elif hasattr(agent, "stream"):
            for chunk in agent.stream(messages[-1]["content"], **kwargs):
                yield chunk
        else:
            # Single response
            response = await self._run_agent_async(agent, messages, **kwargs)
            yield response
    
    async def _run_agent_async(
        self,
        agent: Any,
        messages: List[Dict],
        **kwargs
    ) -> Any:
        """Run agent asynchronously."""
        if hasattr(agent, "run_async"):
            return await agent.run_async(messages[-1]["content"], **kwargs)
        elif hasattr(agent, "arun"):
            return await agent.arun(messages[-1]["content"], **kwargs)
        elif hasattr(agent, "run"):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: agent.run(messages[-1]["content"], **kwargs)
            )
        elif callable(agent):
            if asyncio.iscoroutinefunction(agent):
                return await agent(messages[-1]["content"], **kwargs)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: agent(messages[-1]["content"], **kwargs)
                )
        else:
            raise TypeError(f"Agent must be callable: {type(agent)}")
    
    @classmethod
    async def stream(
        cls,
        agent: Any,
        input: str,
        max_turns: int = 10,
        max_time: float = 300.0,
        **kwargs
    ) -> AsyncIterator[StreamEvent]:
        """Convenience class method for streaming execution.
        
        Example:
            async for event in StreamingRunner.stream(agent, "Hello"):
                print(event.type, event.data)
        """
        runner = cls(max_turns=max_turns, max_time=max_time)
        async for event in runner.execute_stream(agent, input, **kwargs):
            yield event


# Sync wrapper for non-async contexts
def stream_sync(
    agent: Any,
    input: str,
    max_turns: int = 10,
    max_time: float = 300.0,
    **kwargs
) -> Iterator[StreamEvent]:
    """Synchronous wrapper for streaming execution.
    
    Example:
        for event in stream_sync(agent, "Hello"):
            print(event.type, event.data)
    """
    async def collect_events():
        events = []
        async for event in StreamingRunner.stream(agent, input, max_turns, max_time, **kwargs):
            events.append(event)
        return events
    
    loop = asyncio.new_event_loop()
    try:
        events = loop.run_until_complete(collect_events())
        for event in events:
            yield event
    finally:
        loop.close()
