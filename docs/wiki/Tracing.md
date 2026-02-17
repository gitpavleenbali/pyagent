# Tracing

The trace module provides observability and debugging for AI operations.

## Overview

Tracing enables:
- **Operation Logging**: Track all AI operations
- **Performance Monitoring**: Measure execution times
- **Debugging**: Understand what happened in complex workflows
- **Export**: Save traces for analysis

## Quick Start

```python
from pyagent import trace, ask

# Enable tracing
trace.enable()

# Perform operations
answer = ask("What is Python?")

# View trace
trace.show()
```

## Enabling Tracing

### Global Enable

```python
from pyagent import trace

# Enable globally
trace.enable()

# All operations are now traced
from pyagent import ask, research
answer = ask("What is AI?")      # Traced
result = research("ML trends")   # Traced

# Disable when done
trace.disable()
```

### Context Manager

```python
from pyagent import trace, ask

with trace.enabled():
    answer = ask("What is Python?")
    # Trace active within block

# Trace disabled after block
```

## Spans

### Creating Spans

```python
from pyagent import trace

with trace.span("my_operation") as span:
    # Log events
    span.log("Starting operation")
    
    # Perform work
    result = do_something()
    
    # Log with metadata
    span.log("Completed", result_length=len(result))
```

### Nested Spans

```python
with trace.span("parent_task") as parent:
    parent.log("Starting parent")
    
    with trace.span("child_task_1") as child1:
        child1.log("Processing step 1")
        
    with trace.span("child_task_2") as child2:
        child2.log("Processing step 2")
    
    parent.log("Parent complete")
```

## TraceEvent

Events capture individual operations:

```python
from pyagent.easy.trace import TraceEvent

event = TraceEvent(
    type="llm_call",
    message="Called GPT-4",
    metadata={
        "model": "gpt-4",
        "tokens": 150,
        "duration_ms": 1234
    }
)
```

### Event Types

| Type | Description |
|------|-------------|
| `start` | Operation started |
| `end` | Operation completed |
| `log` | Log message |
| `error` | Error occurred |
| `llm_call` | LLM API call |
| `tool_call` | Tool/skill execution |

## Span Object

```python
from pyagent.easy.trace import Span

span = Span(
    name="research_task",
    span_id="span_123",
    parent_span_id="span_parent"
)

# Log events
span.log("Found 5 sources")
span.log("Processing sources", count=5)

# Log errors
span.error("Failed to fetch", exception=e)

# End span
span.end()
```

### Span Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Span name |
| `span_id` | `str` | Unique identifier |
| `parent_span_id` | `str` | Parent span ID |
| `start_time` | `datetime` | Start timestamp |
| `end_time` | `datetime` | End timestamp |
| `events` | `list` | List of events |
| `status` | `str` | "running", "completed", "error" |

## Viewing Traces

### Console Output

```python
from pyagent import trace

trace.enable()
# ... operations ...

# Show last trace
trace.show()

# Output:
# ┌─ research_task (1234ms)
# │  ├─ log: Starting research
# │  ├─ llm_call: GPT-4 (500ms, 150 tokens)
# │  ├─ tool_call: web_search (300ms)
# │  └─ log: Completed
# └─ completed
```

### Get Trace Data

```python
# Get current trace
current = trace.current()

# Get all traces
all_traces = trace.all()

# Filter traces
llm_calls = trace.filter(type="llm_call")
```

## Exporting Traces

### JSON Export

```python
from pyagent import trace

trace.enable()
# ... operations ...

# Export to file
trace.export("traces.json")

# Export with options
trace.export(
    "traces.json",
    include_metadata=True,
    pretty=True
)
```

### Programmatic Access

```python
# Get as dictionary
data = trace.to_dict()

# Get as JSON string
json_str = trace.to_json()
```

## Custom Handlers

### Event Handler

```python
from pyagent import trace

@trace.handler
def my_handler(event):
    """Custom trace handler."""
    print(f"[{event.timestamp}] {event.type}: {event.message}")
    
    if event.type == "error":
        send_alert(event.message)

# Handler is called for all events
trace.enable()
```

### Multiple Handlers

```python
# File logger
@trace.handler
def file_logger(event):
    with open("trace.log", "a") as f:
        f.write(f"{event}\n")

# Metrics collector
@trace.handler
def metrics_collector(event):
    if event.type == "llm_call":
        metrics.record("llm_latency", event.duration_ms)
```

## Performance Monitoring

### Timing Operations

```python
from pyagent import trace

with trace.span("slow_operation") as span:
    result = slow_function()
    
# Access timing
print(f"Duration: {span.duration_ms}ms")
```

### Token Counting

```python
@trace.handler
def token_counter(event):
    if event.type == "llm_call":
        tokens = event.metadata.get("tokens", 0)
        total_tokens[0] += tokens

total_tokens = [0]
trace.enable()

# ... operations ...

print(f"Total tokens used: {total_tokens[0]}")
```

## Integration with Logging

```python
import logging
from pyagent import trace

logger = logging.getLogger("pyagent")

@trace.handler
def logging_handler(event):
    if event.type == "error":
        logger.error(event.message)
    else:
        logger.debug(f"{event.type}: {event.message}")
```

## Decorator

```python
from pyagent import trace

@trace.traced
def my_function(x, y):
    """Function is automatically traced."""
    return x + y

@trace.traced(name="custom_name")
def another_function():
    pass
```

## Configuration

```python
from pyagent import trace

trace.configure(
    enabled=True,
    handlers=[my_handler],
    max_events=10000,       # Limit stored events
    include_timestamps=True,
    include_metadata=True
)
```

## See Also

- [Guardrails](Guardrails) - Input/output protection
- [Azure-AD-Auth](Azure-AD-Auth) - Authentication
- [Evaluation-Module](Evaluation-Module) - Testing agents
