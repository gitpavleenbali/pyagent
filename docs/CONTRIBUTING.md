# 🤝 Contributing to pyai

First off, thank you for considering contributing to pyai! We're building the **pandas of AI**, and every contribution helps make AI development simpler for everyone.

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to build something amazing together.

---

## How Can I Contribute?

### 🐛 Reporting Bugs

1. Check if the bug is already reported in [Issues](https://github.com/pyai/pyai/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS
   - pyai version (`import pyai; print(pyai.__version__)`)

### 💡 Suggesting Features

1. Check existing feature requests
2. Create an issue with the `enhancement` label
3. Describe the feature and use case
4. If possible, suggest implementation approach

### 🔧 Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Write/update tests
5. Ensure all tests pass
6. Commit (`git commit -m 'Add amazing feature'`)
7. Push (`git push origin feature/amazing-feature`)
8. Open a Pull Request

---

## Development Setup

### Prerequisites

- Python 3.9+
- pip or poetry
- OpenAI API key (for integration tests)

### Installation

```bash
# Clone the repository
git clone https://github.com/pyai/pyai.git
cd pyai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Unit tests
pytest tests/

# With coverage
pytest tests/ --cov=pyai --cov-report=html

# Integration tests (requires API key)
export OPENAI_API_KEY=sk-...
pytest tests/integration/
```

### Code Style

We use:
- **Black** for formatting
- **isort** for import sorting
- **Ruff** for linting
- **mypy** for type checking

```bash
# Format code
black pyai/
isort pyai/

# Check linting
ruff check pyai/

# Type checking
mypy pyai/
```

---

## Project Structure

```
pyai/
├── pyai/
│   ├── easy/           # Simple API (contributions welcome!)
│   ├── core/           # Core components
│   ├── instructions/   # Prompt engineering
│   ├── skills/         # Agent capabilities
│   └── blueprint/      # Complex workflows
├── tests/
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── docs/               # Documentation
└── examples/           # Usage examples
```

---

## Contribution Guidelines

### For Easy API (`pyai/easy/`)

This is where the magic happens! When adding new one-liner functions:

1. **Simplicity First**: The function should do one thing well
2. **Sensible Defaults**: Work out-of-the-box without configuration
3. **Progressive Complexity**: Basic usage is simple, advanced options available
4. **Type Hints**: Full type annotations required
5. **Documentation**: Docstrings with examples

Example:

```python
def new_function(
    input: str,
    *,
    option1: bool = False,
    option2: str = "default",
    model: Optional[str] = None,
) -> str:
    """
    One-line description.
    
    Args:
        input: What this input is
        option1: What this option does
        option2: What this option does
        model: Override default model
        
    Returns:
        What the function returns
        
    Example:
        >>> from pyai import new_function
        >>> result = new_function("example input")
        >>> print(result)
        "Expected output"
    """
    config = get_config()
    llm = LLMInterface.from_config(config, model=model)
    
    # Implementation...
    
    return result
```

### For Core Components

When modifying core components:

1. Maintain backward compatibility
2. Add tests for new functionality
3. Update type stubs (`.pyi` files)
4. Document any API changes

### For Skills

When adding new skills:

1. Extend the `Skill` base class
2. Implement `execute()` method
3. Add to `builtin.py` if it's commonly useful
4. Write comprehensive tests

---

## Adding New LLM Providers

To add support for a new LLM provider:

1. Create provider class in `pyai/core/llm.py`:

```python
class NewProvider(LLMProvider):
    def __init__(self, api_key: str, **kwargs):
        self.client = NewLLMClient(api_key)
    
    def complete(self, prompt: str, **kwargs) -> str:
        response = self.client.generate(prompt)
        return response.text
```

2. Add to provider selection in `llm_interface.py`
3. Add configuration options
4. Add integration tests
5. Update documentation

---

## Writing Documentation

- Use clear, concise language
- Include code examples
- Keep examples runnable
- Update API reference for new features
- Add to CHANGELOG.md

---

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag (`git tag v0.2.0`)
4. Push tag (`git push origin v0.2.0`)
5. CI/CD handles PyPI release

---

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Part of building the most revolutionary AI library ever!

---

## Questions?

- Open a [Discussion](https://github.com/pyai/pyai/discussions)
- Join our Discord (coming soon)
- Email: contributors@pyai.dev

---

**Thank you for being part of the pyai revolution!** 🐼🤖
