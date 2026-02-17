# extract

The `extract` function extracts structured data from unstructured text.

## Import

```python
from pyai import extract
```

## Basic Usage

```python
# Extract with schema
data = extract(text, schema={"name": str, "email": str})

# Extract common entities
entities = extract.entities("John works at Microsoft")
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | required | Source text |
| `schema` | dict | None | Expected data structure |
| `type` | str | None | Preset: "contact", "date", "product" |

## Examples

### Extract Contact Info

```python
from pyai import extract

text = """
Contact John Smith at john@example.com
Phone: 555-123-4567
"""

contact = extract(text, schema={
    "name": str,
    "email": str,
    "phone": str
})

print(contact)
# {"name": "John Smith", "email": "john@example.com", "phone": "555-123-4567"}
```

### Extract Entities

```python
text = "Apple announced iPhone 15 on September 12, 2023"

entities = extract.entities(text)
# {
#   "organizations": ["Apple"],
#   "products": ["iPhone 15"],
#   "dates": ["September 12, 2023"]
# }
```

### Extract from Documents

```python
# Extract invoice data
invoice_data = extract(
    invoice_text,
    schema={
        "invoice_number": str,
        "date": str,
        "total": float,
        "items": list
    }
)
```

### Custom Extraction

```python
# Extract product info
product = extract(
    description,
    schema={
        "name": str,
        "price": float,
        "features": list,
        "specifications": dict
    }
)
```

### Async Usage

```python
import asyncio
from pyai import extract

async def main():
    data = await extract.async_(
        email_text,
        schema={"sender": str, "subject": str}
    )
    print(data)

asyncio.run(main())
```

## Preset Types

| Type | Extracted Fields |
|------|-----------------|
| `contact` | name, email, phone, address |
| `date` | dates, times, durations |
| `product` | name, price, description, features |
| `receipt` | items, totals, date, vendor |
| `invoice` | number, date, items, total, tax |

## See Also

- [[ask]] - Question answering
- [[summarize]] - Summarization
- [[analyze]] - Data analysis
