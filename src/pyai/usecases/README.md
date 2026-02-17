# PyAgent Use Cases

**Production-Ready Agent Templates for Every Industry**

The `usecases` module provides pre-configured agent templates for common business scenarios. Each template includes specialized instructions, best practices, and industry-specific knowledge built-in.

## Why Use Templates?

- **Faster Development**: Start with a working agent, customize as needed
- **Best Practices**: Industry knowledge baked into instructions
- **Consistency**: Standardized behavior across your organization
- **Compliance**: Built-in awareness of industry regulations

## Available Categories

### General Use Cases (`__init__.py`)

| Category | Agents |
|----------|--------|
| **Customer Service** | support_agent, technical_agent, billing_agent, escalation_agent |
| **Sales & Marketing** | lead_qualifier, content_writer, sales_assistant, competitor_analyst |
| **Operations** | order_agent, inventory_agent, scheduling_agent, logistics_agent |
| **Development** | code_reviewer, debugger, documenter, architect |
| **Gaming** | npc_agent, game_master, story_writer, trivia_host |

### Industry Templates (`industry.py`)

| Industry | Agents |
|----------|--------|
| **Telecom** | plan_advisor, network_support, retention_agent, activation_agent |
| **Healthcare** | appointment_scheduler, symptom_info, insurance_helper |
| **Finance** | banking_assistant, fraud_alert, loan_advisor, investment_info |
| **E-commerce** | shopping_assistant, order_tracker, returns_agent, product_expert |
| **Education** | tutor, course_advisor, study_buddy, admissions |

## Quick Start

```python
from pyagent.usecases import customer_service, sales

# Create a support agent
support = customer_service.support_agent(
    company_name="Acme Inc",
    tone="friendly and professional"
)

# Use it
response = support("Customer can't access their account")

# Create a lead qualifier
qualifier = sales.lead_qualifier(
    qualification_criteria={
        "budget": "Has $10k+ budget",
        "timeline": "Planning to buy this quarter"
    }
)

lead_score = qualifier("Enterprise company, contacted VP of Sales...")
```

## Customer Service Agents

### Support Agent

```python
from pyagent.usecases import customer_service

support = customer_service.support_agent(
    company_name="Your Company",
    tone="professional and empathetic",
    knowledge_base=None  # Optional: path to KB
)

# Features:
# - Warm greetings
# - Issue acknowledgment
# - Step-by-step guidance
# - Knows when to escalate
```

### Technical Support

```python
tech = customer_service.technical_agent(
    products=["ProductA", "ProductB"],
    expertise_level="advanced"  # basic, intermediate, advanced
)

# Features:
# - Systematic troubleshooting
# - Diagnostic questions
# - Clear technical guidance
# - Documentation references
```

### Support Workflow

```python
# Create a complete multi-tier support system
workflow = customer_service.create_support_workflow(tiers=2)

# Process a ticket
result = workflow("Customer reports slow internet")
print(result["resolution"])
print(result["tier"])  # Which tier resolved it
```

## Sales & Marketing Agents

### Lead Qualifier

```python
from pyagent.usecases import sales

qualifier = sales.lead_qualifier(
    qualification_criteria={
        "budget": "Has allocated budget",
        "authority": "Can make purchase decision",
        "need": "Has clear use case",
        "timeline": "Buying within 90 days"
    },
    scoring_model="BANT"  # or MEDDIC, CHAMP
)

# Returns structured qualification:
# {
#   "scores": {"budget": 4, "authority": 5, ...},
#   "total_score": 16,
#   "qualification": "Qualified",
#   "next_steps": ["Schedule demo", "Send proposal"]
# }
```

### Content Writer

```python
writer = sales.content_writer(
    brand_voice="professional yet approachable",
    content_types=["blog", "social", "email", "landing"]
)

blog_post = writer("Write a blog post about AI in customer service")
```

## Operations Agents

### Order Processing

```python
from pyagent.usecases import operations

order_agent = operations.order_agent(
    order_rules={
        "max_rush_discount": 10,
        "free_shipping_threshold": 50
    }
)

order_agent("New order: 3x Widget Pro, ship to California")
```

### Scheduling

```python
scheduler = operations.scheduling_agent()

scheduler("Book a meeting with John next Tuesday at 2pm EST")
# - Confirms timezone
# - Provides meeting link
# - Sends reminders
```

## Development Agents

### Code Reviewer

```python
from pyagent.usecases import development

reviewer = development.code_reviewer(
    languages=["Python", "JavaScript"],
    standards="PEP8, ESLint recommended"
)

review = reviewer("""
def calculate(x,y):
    return x+y
""")

# Returns:
# - Line-by-line issues
# - Severity ratings
# - Suggested fixes
# - Overall score
```

### Debugger

```python
debugger = development.debugger()

fix = debugger("""
Error: IndexError: list index out of range
Code:
for i in range(len(items) + 1):
    print(items[i])
""")
```

## Industry Templates

### Telecom

```python
from pyagent.usecases.industry import telecom

# Help customers choose plans
advisor = telecom.plan_advisor(
    carrier_name="MobileNet",
    plans=[
        {"name": "Basic", "data": "5GB", "price": "$30"},
        {"name": "Unlimited", "data": "Unlimited", "price": "$60"}
    ]
)

# Network troubleshooting
tech = telecom.network_support()

# Customer retention
retention = telecom.retention_agent(
    max_discount=25,
    can_offer_upgrade=True
)
```

### Healthcare

```python
from pyagent.usecases.industry import healthcare

# Appointment scheduling
scheduler = healthcare.appointment_scheduler(
    facility_name="City Hospital",
    departments=["Primary Care", "Cardiology", "Orthopedics"]
)

# Insurance help (does NOT access real accounts)
insurance = healthcare.insurance_helper()

# General health info (NOT medical advice)
info = healthcare.symptom_info()
```

### Finance

```python
from pyagent.usecases.industry import finance

# General banking questions
banking = finance.banking_assistant(bank_name="First Bank")

# Fraud prevention
fraud = finance.fraud_alert()

# Loan information
loans = finance.loan_advisor()
```

### E-commerce

```python
from pyagent.usecases.industry import ecommerce

# Product discovery
shopping = ecommerce.shopping_assistant(
    store_name="ShopMart",
    categories=["Electronics", "Home", "Fashion"]
)

# Order tracking
tracker = ecommerce.order_tracker()

# Returns and exchanges
returns = ecommerce.returns_agent(
    return_window=30,
    free_returns=True
)
```

### Education

```python
from pyagent.usecases.industry import education

# Subject tutoring
math_tutor = education.tutor(
    subject="Mathematics",
    level="High School",
    teaching_style="patient and encouraging"
)

# Course planning
advisor = education.course_advisor(
    institution="State University",
    programs=["Computer Science", "Business"]
)
```

## Gaming Agents

### NPC Agent

```python
from pyagent.usecases import gaming

elder = gaming.npc_agent(
    character_name="Elder Thorne",
    personality="wise and mysterious",
    backstory="Guardian of ancient knowledge",
    knowledge=[
        "Knows location of the hidden temple",
        "Can teach ancient magic",
        "Warns about the dark forest"
    ]
)

response = elder("What dangers lie ahead?")
# *strokes beard thoughtfully* "The forest holds many secrets..."
```

### Game Master

```python
gm = gaming.game_master(
    setting="fantasy",
    difficulty="challenging",
    style="narrative-focused"
)

gm("The players enter the dungeon")
# Vivid description of the dungeon, atmospheric details,
# presents meaningful choices to players
```

### Interactive Story

```python
story = gaming.story_writer(
    genre="sci-fi",
    audience="adult"
)

chapter = story("Begin the adventure on a space station")
# Returns story text with 2-3 choices at the end
```

## Customizing Templates

All templates accept customization parameters:

```python
# Start with template
support = customer_service.support_agent(company_name="Acme")

# Add memory for conversation continuity
support_with_memory = customer_service.support_agent(
    company_name="Acme"
)  # memory=True is default for some agents

# Combine with custom skills
from pyagent.skills import web_search

custom_support = agent(
    customer_service.support_agent().__instructions__ + 
    "\nYou also have access to search company documentation.",
    skills=[web_search]
)
```

## Listing Available Templates

```python
from pyagent.usecases import list_usecases
from pyagent.usecases.industry import list_industries

# See all general use cases
print(list_usecases())
# {
#   "customer_service": ["support_agent", "technical_agent", ...],
#   "sales": ["lead_qualifier", "content_writer", ...],
#   ...
# }

# See all industry templates
print(list_industries())
# {
#   "telecom": ["plan_advisor", "network_support", ...],
#   "healthcare": ["appointment_scheduler", ...],
#   ...
# }
```

## Module Structure

```
usecases/
├── __init__.py      # General use cases
├── industry.py      # Industry-specific templates
└── README.md        # This documentation
```

## Best Practices

1. **Start with Templates**: Use as-is first, customize later
2. **Keep Instructions Focused**: Don't overload with too many responsibilities
3. **Test Thoroughly**: Test edge cases specific to your domain
4. **Add Memory Wisely**: Enable for conversational, disable for stateless
5. **Combine Patterns**: Use with orchestrator for complex workflows

## Contributing Templates

Have a useful agent template? Contribute it:

1. Add to appropriate file (`__init__.py` or `industry.py`)
2. Follow the existing pattern (class with static methods)
3. Document parameters and capabilities
4. Add to this README

## See Also

- [Orchestrator](../orchestrator/) - Combine templates into workflows
- [Examples](../../examples/) - See templates in action
- [API Reference](../../docs/API_REFERENCE.md)
