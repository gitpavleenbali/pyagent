# Contributing

Thank you for contributing to PyAgent! We're building the **pandas of AI**.

## Code of Conduct

Be respectful, inclusive, and constructive.

---

## How to Contribute

### Reporting Bugs

1. Check if already reported in [Issues](https://github.com/gitpavleenbali/PYAI/issues)
2. Create new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS
   - PyAgent version

### Suggesting Features

1. Check existing feature requests
2. Create issue with `enhancement` label
3. Describe feature and use case

### Pull Requests

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes
4. Write/update tests
5. Ensure tests pass
6. Commit: `git commit -m 'Add amazing feature'`
7. Push: `git push origin feature/amazing-feature`
8. Open Pull Request

---

## Development Setup

### Prerequisites

- Python 3.10+
- pip or poetry
- OpenAI API key (for integration tests)

### Installation

```bash
# Clone repository
git clone https://github.com/gitpavleenbali/PYAI.git
cd PYAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Unit tests
pytest tests/

# With coverage
pytest tests/ --cov=pyagent --cov-report=html

# Specific test file
pytest tests/test_core.py -v
```

### Code Style

We use:
- **Black** - Code formatting
- **Ruff** - Linting
- **mypy** - Type checking

```bash
# Format code
black src/

# Lint
ruff check src/

# Type check
mypy src/pyagent --ignore-missing-imports
```

---

## Project Structure

```
pyagent/
├── src/pyagent/      # Main package
│   ├── easy/         # One-liner functions
│   ├── core/         # Core components
│   ├── blueprint/    # Workflows
│   ├── skills/       # Skills system
│   └── ...
├── tests/            # Test suite
├── docs/             # Documentation
└── examples/         # Usage examples
```

## Adding Features

### New One-Liner Function

1. Create `src/pyagent/easy/yourfunction.py`
2. Add to `src/pyagent/easy/__init__.py`
3. Add lazy import in `src/pyagent/__init__.py`
4. Write tests in `tests/test_easy_features.py`
5. Document in wiki

### New Skill

1. Create skill in `src/pyagent/skills/`
2. Register in skill registry
3. Write tests
4. Document usage

### New Integration

1. Add to `src/pyagent/integrations/`
2. Create adapter class
3. Write tests
4. Document in wiki

---

## Guidelines

### Code Quality

- Type hints for all public functions
- Docstrings with examples
- Tests for new features
- No unused imports

### Commit Messages

Use conventional commits:
- `feat: add new feature`
- `fix: fix bug`
- `docs: update documentation`
- `test: add tests`
- `refactor: refactor code`

### Documentation

- Update wiki for new features
- Add docstrings
- Include usage examples

---

## Review Process

1. PR must pass CI checks
2. Code review by maintainer
3. Address feedback
4. Merge!

---

## Getting Help

- Open an issue
- Check existing documentation
- Review similar PRs

---

## Recognition

Contributors are recognized in:
- Release notes
- README contributors section
- GitHub insights

---

## See Also

- [Quick-Start](Quick-Start) - Getting started
- [Architecture](Architecture) - System design
- [API-Reference](API-Reference) - API docs
