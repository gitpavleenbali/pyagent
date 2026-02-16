# PyAgent Orchestrator

**Enterprise Workflow Automation & Multi-Agent Coordination**

The `orchestrator` module provides powerful workflow management, task scheduling, and sophisticated multi-agent patterns for building complex AI systems.

## Features

- **Task Management**: Submit, schedule, and track AI tasks
- **Workflow Orchestration**: Chain tasks with dependencies
- **Agent Patterns**: Pre-built multi-agent strategies
- **Scheduling**: One-time, interval, and cron-based scheduling
- **Event System**: Pub/sub for task lifecycle events
- **State Persistence**: Save and restore workflow state

## Quick Start

```python
from pyagent.orchestrator import Orchestrator, Task, AgentPatterns
from pyagent import agent

# Create orchestrator
orch = Orchestrator()

# Submit a simple task
task = orch.submit(
    agent=agent("You are a researcher"),
    input="Research quantum computing trends",
    priority=1
)

# Check status
print(orch.status(task.id))  # COMPLETED
print(task.output)  # Research results
```

## Core Concepts

### Tasks

Tasks are units of work assigned to agents:

```python
from pyagent.orchestrator import Task, TaskStatus

task = Task(
    id="task-001",
    name="Research Task",
    agent=my_agent,
    input="Research topic",
    status=TaskStatus.PENDING,
    priority=1,
    metadata={"category": "research"}
)
```

**Task Properties:**
- `id`: Unique identifier
- `name`: Human-readable name
- `agent`: The agent to execute the task
- `input`: Task input/prompt
- `status`: Current status (PENDING, RUNNING, COMPLETED, FAILED, etc.)
- `priority`: Execution priority (1=highest)
- `output`: Task result (after completion)
- `error`: Error message (if failed)
- `created_at`, `started_at`, `completed_at`: Timestamps

### Workflows

Workflows chain multiple tasks with dependencies:

```python
from pyagent.orchestrator import Workflow

workflow = Workflow(
    id="workflow-001",
    name="Research Pipeline",
    tasks=[task1, task2, task3],
    dependencies={
        "task2": ["task1"],  # task2 waits for task1
        "task3": ["task1", "task2"]  # task3 waits for both
    }
)

# Submit workflow
orch.submit_workflow(workflow)
```

### Scheduled Jobs

Schedule tasks to run automatically:

```python
from pyagent.orchestrator import ScheduledJob

# Run once at specific time
job = ScheduledJob(
    task=daily_report_task,
    schedule_type="once",
    run_at=datetime(2024, 12, 25, 9, 0)
)

# Run every hour
job = ScheduledJob(
    task=health_check_task,
    schedule_type="interval",
    interval_seconds=3600
)

# Run on cron schedule
job = ScheduledJob(
    task=weekly_summary_task,
    schedule_type="cron",
    cron="0 9 * * MON"  # Every Monday at 9 AM
)

orch.schedule(job)
```

## Orchestrator API

### Creating an Orchestrator

```python
from pyagent.orchestrator import Orchestrator

# Basic orchestrator
orch = Orchestrator()

# With state persistence
orch = Orchestrator(
    state_file="./workflow_state.json"
)
```

### Submitting Tasks

```python
# Submit single task
task = orch.submit(
    agent=my_agent,
    input="Process this data",
    priority=1,
    metadata={"user": "alice"}
)

# Submit with callback
def on_complete(task):
    print(f"Task {task.id} completed: {task.output}")

task = orch.submit(
    agent=my_agent,
    input="...",
    callback=on_complete
)
```

### Checking Status

```python
# Get task status
status = orch.status(task.id)  # TaskStatus enum

# Get full task details
task = orch.get_task(task.id)
print(task.output)
print(task.started_at)
print(task.completed_at)
```

### Event System

```python
# Subscribe to events
@orch.on("task.completed")
def handle_completion(task):
    print(f"Task completed: {task.name}")

@orch.on("task.failed")
def handle_failure(task):
    alert_team(task.error)

@orch.on("workflow.completed")
def handle_workflow_done(workflow):
    generate_report(workflow)

# Emit custom events
orch.emit("custom.event", data={"key": "value"})
```

## Execution Patterns

The orchestrator supports various execution patterns:

```python
from pyagent.orchestrator import ExecutionPattern

# Sequential: One task at a time
orch.submit_workflow(workflow, pattern=ExecutionPattern.SEQUENTIAL)

# Parallel: All independent tasks at once
orch.submit_workflow(workflow, pattern=ExecutionPattern.PARALLEL)

# Supervisor: One agent coordinates others
orch.submit_workflow(workflow, pattern=ExecutionPattern.SUPERVISOR)

# Collaborative: Agents work together on shared output
orch.submit_workflow(workflow, pattern=ExecutionPattern.COLLABORATIVE)
```

## Agent Patterns

Pre-built multi-agent coordination strategies:

### Supervisor Pattern

One agent coordinates and delegates to specialized agents:

```python
from pyagent.orchestrator import AgentPatterns
from pyagent import agent

# Create specialized agents
researcher = agent("You are a research specialist")
writer = agent("You are a content writer")
editor = agent("You are an editor")

# Run with supervisor
result = AgentPatterns.supervisor(
    task="Write a research report on AI trends",
    agents=[researcher, writer, editor],
    supervisor_instructions="Coordinate the team to produce a report"
)
```

### Consensus Pattern

Multiple agents vote to reach agreement:

```python
# Multiple experts evaluate
experts = [
    agent("You are a security expert"),
    agent("You are a performance expert"),
    agent("You are a UX expert")
]

result = AgentPatterns.consensus(
    task="Should we approve this feature?",
    agents=experts,
    threshold=0.66  # 66% must agree
)

print(result.decision)  # "approved" or "rejected"
print(result.votes)  # Individual agent votes
print(result.reasoning)  # Combined reasoning
```

### Debate Pattern

Agents argue opposing positions:

```python
result = AgentPatterns.debate(
    topic="Should AI systems be open-sourced?",
    pro_agent=agent("Argue FOR open-source AI"),
    con_agent=agent("Argue AGAINST open-source AI"),
    judge=agent("Judge the debate fairly"),
    rounds=3
)

print(result.winner)  # "pro" or "con"
print(result.transcript)  # Full debate
print(result.final_verdict)  # Judge's reasoning
```

### Chain of Thought Pattern

Break complex problems into reasoning steps:

```python
result = AgentPatterns.chain_of_thought(
    problem="How should we scale our service to 10x users?",
    reasoner=agent("You are a systems architect"),
    steps=[
        "Identify current bottlenecks",
        "Propose scaling strategies",
        "Evaluate trade-offs",
        "Create implementation plan"
    ]
)

print(result.final_answer)
print(result.reasoning_chain)  # Step-by-step reasoning
```

## Real-World Examples

### Customer Support Workflow

```python
from pyagent.orchestrator import Orchestrator, AgentPatterns
from pyagent import agent

orch = Orchestrator()

# Agents
classifier = agent("Classify support tickets by urgency and type")
tech_support = agent("Provide technical solutions")
billing_support = agent("Handle billing questions")
escalation = agent("Handle escalations professionally")

def handle_ticket(ticket: str):
    # Classify first
    classification = classifier(ticket)
    
    # Route to appropriate agent
    if "technical" in classification.lower():
        return tech_support(ticket)
    elif "billing" in classification.lower():
        return billing_support(ticket)
    else:
        return escalation(ticket)

# Submit tickets
for ticket in incoming_tickets:
    orch.submit(
        agent=classifier,
        input=ticket,
        callback=handle_ticket
    )
```

### Research Pipeline

```python
from pyagent.orchestrator import Workflow, Task

# Create tasks
gather_task = Task(
    name="Gather Sources",
    agent=researcher,
    input="Find sources on topic X"
)

analyze_task = Task(
    name="Analyze Sources",
    agent=analyst,
    input=lambda: f"Analyze: {gather_task.output}"
)

write_task = Task(
    name="Write Report",
    agent=writer,
    input=lambda: f"Write report based on: {analyze_task.output}"
)

# Create workflow
workflow = Workflow(
    name="Research Pipeline",
    tasks=[gather_task, analyze_task, write_task],
    dependencies={
        "Analyze Sources": ["Gather Sources"],
        "Write Report": ["Analyze Sources"]
    }
)

# Execute
orch.submit_workflow(workflow)
```

### Scheduled Monitoring

```python
from pyagent.orchestrator import Orchestrator, ScheduledJob
from pyagent import agent

orch = Orchestrator()
monitor = agent("Analyze system metrics and alert on anomalies")

# Health check every 5 minutes
health_job = orch.schedule(
    agent=monitor,
    input="Check system health: {metrics}",
    interval_seconds=300
)

# Daily summary at 9 AM
summary_job = orch.schedule(
    agent=agent("Generate daily summary"),
    input="Summarize yesterday's events",
    cron="0 9 * * *"
)
```

## State Persistence

Save and restore orchestrator state:

```python
# Create with persistence
orch = Orchestrator(state_file="./state.json")

# State is automatically saved after each operation

# Later, restore state
orch = Orchestrator(state_file="./state.json")
# Pending tasks are automatically resumed
```

## Module Structure

```
orchestrator/
├── __init__.py     # Main module (Orchestrator, Task, Workflow, etc.)
└── README.md       # This documentation
```

## Best Practices

1. **Use Priorities**: Set priorities for urgent tasks
2. **Handle Failures**: Always add error handlers via events
3. **Persist State**: Enable persistence for production
4. **Monitor Progress**: Use events to track workflow progress
5. **Keep Tasks Atomic**: Each task should do one thing well
6. **Use Patterns**: Leverage built-in patterns for common scenarios

## See Also

- [Multi-Agent Workflow Example](../../examples/multi_agent_workflow.py)
- [Use Cases](../usecases/)
- [Agent Patterns Paper](https://arxiv.org/abs/...)
