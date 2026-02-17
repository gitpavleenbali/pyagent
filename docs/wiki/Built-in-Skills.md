# Built-in Skills

PYAI includes pre-built skills for common operations.

## Available Skills

| Skill | Description |
|-------|-------------|
| `SearchSkill` | Web and local search |
| `CodeSkill` | Code execution |
| `FileSkill` | File operations |
| `WebSkill` | HTTP requests |
| `MathSkill` | Mathematical operations |
| `DateTimeSkill` | Date/time utilities |

## SearchSkill

```python
from pyai.skills import SearchSkill

search = SearchSkill()

# Use with agent
agent = Agent(
    name="Researcher",
    tools=[search]
)
```

### Methods

| Method | Description |
|--------|-------------|
| `search_web(query)` | Search the web |
| `search_news(topic)` | Search news articles |
| `search_images(query)` | Search images |

### Example

```python
from pyai.skills import SearchSkill

search = SearchSkill(api_key="...")

# Direct use
results = search.search_web("Python tutorials")
for result in results:
    print(f"{result.title}: {result.url}")
```

## CodeSkill

```python
from pyai.skills import CodeSkill

code = CodeSkill()

agent = Agent(
    name="Code Assistant",
    tools=[code]
)
```

### Methods

| Method | Description |
|--------|-------------|
| `execute_python(code)` | Run Python code |
| `execute_shell(command)` | Run shell command |
| `format_code(code, language)` | Format code |

### Example

```python
from pyai.skills import CodeSkill

code = CodeSkill(sandbox=True)

# Execute code
result = code.execute_python("""
import math
print(math.sqrt(144))
""")
print(result.output)  # "12.0"
```

## FileSkill

```python
from pyai.skills import FileSkill

file = FileSkill(base_path="./workspace")

agent = Agent(
    name="File Manager",
    tools=[file]
)
```

### Methods

| Method | Description |
|--------|-------------|
| `read_file(path)` | Read file contents |
| `write_file(path, content)` | Write to file |
| `list_files(directory)` | List directory |
| `delete_file(path)` | Delete file |

### Example

```python
from pyai.skills import FileSkill

file = FileSkill()

# Read
content = file.read_file("config.json")

# Write
file.write_file("output.txt", "Hello World")

# List
files = file.list_files("./data/")
```

## WebSkill

```python
from pyai.skills import WebSkill

web = WebSkill()

agent = Agent(
    name="Web Agent",
    tools=[web]
)
```

### Methods

| Method | Description |
|--------|-------------|
| `fetch_url(url)` | GET request |
| `post_data(url, data)` | POST request |
| `download_file(url, path)` | Download file |

### Example

```python
from pyai.skills import WebSkill

web = WebSkill()

# Fetch webpage
content = web.fetch_url("https://example.com")

# POST data
response = web.post_data(
    "https://api.example.com/data",
    {"key": "value"}
)
```

## MathSkill

```python
from pyai.skills import MathSkill

math = MathSkill()

agent = Agent(
    name="Calculator",
    tools=[math]
)
```

### Methods

| Method | Description |
|--------|-------------|
| `calculate(expression)` | Evaluate expression |
| `solve_equation(equation)` | Solve algebraically |
| `statistics(data)` | Calculate stats |
| `convert_units(value, from, to)` | Unit conversion |

### Example

```python
from pyai.skills import MathSkill

math = MathSkill()

result = math.calculate("sin(45) * 2 + sqrt(16)")
stats = math.statistics([1, 2, 3, 4, 5])
converted = math.convert_units(100, "celsius", "fahrenheit")
```

## DateTimeSkill

```python
from pyai.skills import DateTimeSkill

datetime = DateTimeSkill()

agent = Agent(
    name="Scheduler",
    tools=[datetime]
)
```

### Methods

| Method | Description |
|--------|-------------|
| `get_current_time(timezone)` | Current time |
| `parse_date(text)` | Parse date string |
| `add_time(date, duration)` | Add duration |
| `diff_dates(date1, date2)` | Date difference |

## Combining Skills

```python
from pyai import Agent
from pyai.skills import SearchSkill, CodeSkill, FileSkill

agent = Agent(
    name="Power User",
    instructions="Use your skills to help users",
    tools=[
        SearchSkill(),
        CodeSkill(),
        FileSkill()
    ]
)
```

## See Also

- [[Creating-Tools]] - Custom tools
- [[OpenAPI-Tools]] - Generated tools
- [[Agent]] - Agent class
