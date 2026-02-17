# Orchestration Patterns

PYAI provides several patterns for orchestrating multiple agents.

## Available Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Chain** | Sequential execution | Multi-step processes |
| **Router** | Dynamic routing | Task classification |
| **MapReduce** | Parallel + combine | Batch processing |
| **Supervisor** | Manager + workers | Complex coordination |

## Chain Pattern

Agents execute sequentially, passing output to the next.

```python
from pyai.blueprint import Chain

# Create agents
researcher = Agent(name="Researcher", instructions="Research topics")
analyst = Agent(name="Analyst", instructions="Analyze findings")
writer = Agent(name="Writer", instructions="Write report")

# Chain them
pipeline = Chain([researcher, analyst, writer])

# Run
result = pipeline.run("AI trends in 2024")
# researcher -> analyst -> writer
```

### Chain with Transform

```python
pipeline = Chain([
    researcher,
    lambda x: f"Analyze this: {x}",  # Transform between
    analyst,
    writer
])
```

## Router Pattern

Dynamically routes to appropriate agent based on input.

```python
from pyai.blueprint import Router

# Specialized agents
math_agent = Agent(name="Math", instructions="Solve math problems")
code_agent = Agent(name="Code", instructions="Write code")
general_agent = Agent(name="General", instructions="General questions")

# Router
router = Router(
    agents={
        "math": math_agent,
        "code": code_agent,
        "general": general_agent
    },
    classifier="auto"  # Auto-classify input
)

# Run (automatically routes)
result = router.run("What is 25 * 4?")  # -> math_agent
result = router.run("Write a Python sort")  # -> code_agent
```

### Custom Classifier

```python
def classify(message):
    if "calculate" in message.lower():
        return "math"
    elif "code" in message.lower():
        return "code"
    return "general"

router = Router(
    agents={...},
    classifier=classify
)
```

## MapReduce Pattern

Process in parallel, then combine results.

```python
from pyai.blueprint import MapReduce

# Worker agent
researcher = Agent(
    name="Researcher",
    instructions="Research the given topic thoroughly"
)

# Reducer agent
synthesizer = Agent(
    name="Synthesizer",
    instructions="Combine research into cohesive summary"
)

# MapReduce
pipeline = MapReduce(
    mapper=researcher,
    reducer=synthesizer
)

# Run with multiple inputs
topics = ["AI ethics", "quantum computing", "biotechnology"]
result = pipeline.run(topics)
# Researches all in parallel, then synthesizes
```

### Custom Map/Reduce

```python
pipeline = MapReduce(
    mapper=researcher,
    reducer=synthesizer,
    map_fn=lambda topic: f"Research: {topic}",
    reduce_fn=lambda results: "\n".join(results)
)
```

## Supervisor Pattern

Manager agent coordinates worker agents.

```python
from pyai.blueprint import Supervisor

# Worker agents
coder = Agent(name="Coder", instructions="Write code")
reviewer = Agent(name="Reviewer", instructions="Review code")
tester = Agent(name="Tester", instructions="Write tests")

# Supervisor
manager = Supervisor(
    name="Tech Lead",
    instructions="""
    Coordinate the team:
    1. Have Coder implement the feature
    2. Have Reviewer check the code
    3. Have Tester add tests
    Report final status
    """,
    workers=[coder, reviewer, tester]
)

# Run
result = manager.run("Implement user authentication")
```

### Supervisor with Roles

```python
manager = Supervisor(
    name="Project Manager",
    workers={
        "development": coder,
        "qa": reviewer,
        "testing": tester
    },
    max_iterations=5
)
```

## Custom Patterns

### Debate Pattern

```python
from pyai import Agent, Runner

advocate = Agent(name="Advocate", instructions="Argue FOR the topic")
critic = Agent(name="Critic", instructions="Argue AGAINST the topic")
judge = Agent(name="Judge", instructions="Evaluate arguments")

async def debate(topic, rounds=3):
    context = {"topic": topic, "arguments": []}
    
    for i in range(rounds):
        # Advocate
        pro = await Runner.run(advocate, f"Round {i+1}: Argue for {topic}")
        context["arguments"].append(("pro", pro.output))
        
        # Critic
        con = await Runner.run(critic, f"Round {i+1}: Argue against {topic}")
        context["arguments"].append(("con", con.output))
    
    # Judge
    verdict = await Runner.run(judge, f"Judge debate: {context['arguments']}")
    return verdict
```

### Expert Consensus

```python
experts = [
    Agent(name="Expert A", instructions="Domain A perspective"),
    Agent(name="Expert B", instructions="Domain B perspective"),
    Agent(name="Expert C", instructions="Domain C perspective"),
]

async def consensus(question):
    opinions = await asyncio.gather(*[
        Runner.run(expert, question) for expert in experts
    ])
    
    synthesizer = Agent(
        name="Synthesizer",
        instructions="Find consensus among expert opinions"
    )
    
    return await Runner.run(synthesizer, str(opinions))
```

## See Also

- [[Workflows]] - Multi-step workflows
- [[Handoffs]] - Agent handoffs
- [[Agent]] - Agent class
