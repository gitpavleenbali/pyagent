# Workflows

Multi-step agent processes in the **blueprint/** module.

---

## Basic Workflow

```python
from pyai import Agent
from pyai.blueprint import Workflow, Step

# Create specialized agents
researcher = Agent(name="Researcher", instructions="Find information on topics.")
writer = Agent(name="Writer", instructions="Write clear, engaging content.")
editor = Agent(name="Editor", instructions="Review and improve writing.")

# Build workflow
workflow = (Workflow("ContentPipeline")
    .add_step(Step("research", researcher))
    .add_step(Step("write", writer))
    .add_step(Step("edit", editor))
    .build())

# Run
result = await workflow.run("Create an article about AI trends")
print(result.final_output)
```

---

## Step Types

| Type | Description | Use Case |
|------|-------------|----------|
| `AGENT` | Execute an agent | Main processing |
| `SKILL` | Call a specific skill | Tool invocation |
| `FUNCTION` | Run Python function | Custom logic |
| `CONDITION` | Conditional branching | Decision points |
| `PARALLEL` | Concurrent execution | Speed optimization |
| `LOOP` | Repeated execution | Iteration |

---

## Conditional Steps

```python
from pyai.blueprint import Workflow, Step, Condition

def needs_review(context):
    return context.get("complexity") == "high"

workflow = (Workflow("ReviewPipeline")
    .add_step(Step("analyze", analyzer))
    .add_step(Step("review", reviewer, condition=Condition(needs_review)))
    .add_step(Step("finalize", finalizer))
    .build())
```

---

## Parallel Execution

```python
from pyai.blueprint import Workflow, ParallelStep

# Run multiple agents concurrently
workflow = (Workflow("ParallelResearch")
    .add_step(ParallelStep("research", [
        ("tech", tech_researcher),
        ("market", market_researcher),
        ("competitor", competitor_researcher),
    ]))
    .add_step(Step("synthesize", synthesizer))
    .build())
```

---

## Workflow Context

Data flows through the workflow via context:

```python
from pyai.blueprint import WorkflowContext

# Initial context
context = WorkflowContext(
    input="Research topic",
    variables={"audience": "technical"}
)

result = await workflow.run(context)

# Access step outputs
print(result.steps["research"].output)
print(result.steps["write"].output)
```

---

## Error Handling

```python
workflow = (Workflow("SafePipeline")
    .add_step(Step("process", processor, 
        on_error="skip",      # skip, retry, fail
        retry_count=3,
        retry_delay=1.0
    ))
    .build())
```

---

## Real-World Example: Content Creation

```python
from pyai import Agent
from pyai.blueprint import Workflow, Step, ParallelStep

# Specialized team
topic_researcher = Agent(
    name="TopicResearcher",
    instructions="Research topics deeply and provide key facts."
)

seo_specialist = Agent(
    name="SEOSpecialist", 
    instructions="Identify keywords and SEO opportunities."
)

content_writer = Agent(
    name="ContentWriter",
    instructions="Write engaging blog posts with SEO optimization."
)

editor = Agent(
    name="Editor",
    instructions="Polish content for grammar, clarity, and flow."
)

# Content factory workflow
content_factory = (Workflow("ContentFactory")
    # Parallel research
    .add_step(ParallelStep("research", [
        ("topic", topic_researcher),
        ("seo", seo_specialist),
    ]))
    # Sequential writing and editing
    .add_step(Step("write", content_writer))
    .add_step(Step("edit", editor))
    .build())

# Generate content
result = await content_factory.run("AI in Healthcare: 2024 Trends")
print(result.final_output)
```

---

## Workflow Persistence

Save and resume workflows:

```python
# Save checkpoint
checkpoint = workflow.save_checkpoint()

# Resume later
workflow.restore_checkpoint(checkpoint)
result = await workflow.resume()
```

---

## Combining with Patterns

Workflows can use orchestration patterns:

```python
from pyai.blueprint import ChainPattern, RouterPattern

# Chain pattern within workflow
chain = ChainPattern()
chain.add("analyze", analyzer)
chain.add("recommend", recommender)

workflow = (Workflow("Analysis")
    .add_step(Step("chain", chain))
    .build())
```

---

## Next Steps

- [[Orchestration Patterns]] - Chain, Router, MapReduce, Supervisor
- [[Handoffs]] - Agent-to-agent transfers
- [[Agent]] - Create agents
