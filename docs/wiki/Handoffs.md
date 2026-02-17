# Handoffs

Handoffs enable agents to transfer control to other agents during execution.

## Overview

Handoffs allow:
- Specialized agents for different tasks
- Dynamic routing based on user needs
- Collaborative multi-agent systems

## Basic Usage

```python
from pyai import Agent, handoff

# Create specialized agents
sales_agent = Agent(
    name="Sales Agent",
    instructions="Handle sales inquiries"
)

support_agent = Agent(
    name="Support Agent",
    instructions="Handle technical support"
)

# Main agent with handoffs
triage_agent = Agent(
    name="Triage Agent",
    instructions="Route users to appropriate agent",
    handoffs=[sales_agent, support_agent]
)
```

## Handoff Decorator

```python
from pyai import Agent, handoff

@handoff(target="support_agent")
def transfer_to_support(reason: str):
    """Transfer the conversation to support team"""
    return f"Transferring to support: {reason}"

agent = Agent(
    name="Router",
    handoffs=[transfer_to_support]
)
```

## Conditional Handoffs

```python
from pyai import Agent

def should_handoff(context):
    """Determine if handoff is needed"""
    if "billing" in context.message.lower():
        return "billing_agent"
    elif "technical" in context.message.lower():
        return "support_agent"
    return None

agent = Agent(
    name="Router",
    handoff_condition=should_handoff,
    handoffs={
        "billing_agent": billing_agent,
        "support_agent": support_agent
    }
)
```

## Examples

### Customer Service Bot

```python
from pyai import Agent, Runner

# Specialized agents
billing = Agent(
    name="Billing",
    instructions="Handle billing and payment issues"
)

technical = Agent(
    name="Technical",
    instructions="Handle technical support"
)

sales = Agent(
    name="Sales",
    instructions="Handle new purchases and upgrades"
)

# Router
router = Agent(
    name="Customer Service",
    instructions="""
    You are a customer service router.
    - Billing issues -> hand off to Billing
    - Technical problems -> hand off to Technical  
    - Purchase inquiries -> hand off to Sales
    """,
    handoffs=[billing, technical, sales]
)

# Run
result = Runner.run_sync(router, "I can't log into my account")
# Automatically routes to Technical agent
```

### Research Team

```python
from pyai import Agent

researcher = Agent(
    name="Researcher",
    instructions="Find and gather information"
)

analyst = Agent(
    name="Analyst",
    instructions="Analyze and synthesize findings"
)

writer = Agent(
    name="Writer",
    instructions="Write final report"
)

# Chain through handoffs
lead = Agent(
    name="Research Lead",
    instructions="""
    Coordinate research:
    1. Hand off to Researcher for gathering
    2. Hand off to Analyst for analysis
    3. Hand off to Writer for final report
    """,
    handoffs=[researcher, analyst, writer]
)
```

## Handoff Context

```python
from pyai import handoff

@handoff
def transfer_with_context(agent, context):
    """Transfer with full context"""
    return {
        "target": agent,
        "context": {
            "user_id": context.user_id,
            "history": context.messages[-5:],
            "metadata": context.metadata
        }
    }
```

## Handoff Events

```python
from pyai import Runner

async for event in Runner.run_stream(agent, message):
    if event.type == "handoff":
        print(f"Handing off to: {event.target}")
        print(f"Reason: {event.reason}")
    elif event.type == "response":
        print(event.content)
```

## See Also

- [[Workflows]] - Multi-step workflows
- [[Orchestration-Patterns]] - Orchestration patterns
- [[Agent]] - Agent class
