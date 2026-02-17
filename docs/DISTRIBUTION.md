# pyai Distribution Guide

## Installation Methods

### 1. Install from PyPI (Coming Soon)

```bash
# Basic installation
pip install pyai

# With OpenAI support
pip install pyai[openai]

# With Azure support (recommended for enterprise)
pip install pyai[azure]

# All features
pip install pyai[all]
```

### 2. Install from Source

```bash
# Clone the repository
git clone https://github.com/gitpavleenbali/pyai.git
cd pyai

# Install in development mode
pip install -e .[all]
```

### 3. Install from ZIP Distribution

1. Download `pyai-X.X.X-release.zip`
2. Extract to your preferred location
3. Install:
   ```bash
   cd pyai-X.X.X-release
   pip install .
   ```

## Optional Dependencies

| Extra | Includes | Use Case |
|-------|----------|----------|
| `openai` | openai SDK | OpenAI API access |
| `anthropic` | anthropic SDK | Claude API access |
| `azure` | azure-identity, azure-search-documents, openai | Azure OpenAI & AI Search |
| `web` | aiohttp, requests, beautifulsoup4 | Web scraping & fetching |
| `docs` | pypdf, python-docx | Document processing |
| `langchain` | langchain, langchain-community | LangChain integration |
| `semantic-kernel` | semantic-kernel | Microsoft SK integration |
| `vector` | chromadb, faiss-cpu, pinecone, qdrant | Vector databases |
| `all` | Everything above | Full functionality |
| `dev` | pytest, black, ruff, mypy, pre-commit | Development tools |

### Example Installations

```bash
# For Azure enterprise deployment
pip install pyai[azure,web,docs]

# For RAG applications
pip install pyai[openai,vector,web]

# For LangChain users
pip install pyai[openai,langchain]

# For development
pip install -e .[all,dev]
```

## Building from Source

### Prerequisites

- Python 3.9+
- Git
- pip

### Build Commands (Windows PowerShell)

```powershell
# Build wheel and sdist
.\build.ps1 build

# Create ZIP distribution
.\build.ps1 zip

# Full build (clean + build + zip)
.\build.ps1 all

# Install locally
.\build.ps1 install

# Run tests
.\build.ps1 test
```

### Build Commands (Cross-platform)

```bash
# Install build tools
pip install build wheel

# Build distribution packages
python -m build

# Output: dist/pyai-X.X.X.tar.gz and dist/pyai-X.X.X-py3-none-any.whl
```

## Package Structure

```
pyai-X.X.X-release/
├── pyai/                 # Main package
│   ├── __init__.py          # Package root
│   ├── py.typed             # PEP 561 marker
│   ├── easy/                # One-liner functions
│   ├── core/                # Core components
│   ├── skills/              # Agent skills
│   ├── blueprint/           # Blueprints & patterns
│   ├── instructions/        # Instruction builders
│   ├── integrations/        # External integrations
│   ├── orchestrator/        # Workflow orchestration
│   └── usecases/            # Pre-built templates
├── examples/                # Example scripts
├── docs/                    # Documentation
├── README.md                # Main documentation
├── LICENSE                  # MIT License
├── pyproject.toml           # Package configuration
└── setup.py                 # Backward compatibility
```

## Verification

After installation, verify pyai is working:

```python
# Basic verification
import pyai
print(f"pyai version: {pyai.__version__}")

# Check available functions
from pyai import ask, agent, research
print("Core functions available!")

# Test with mock (no API key needed)
from pyai.easy.config import config
config.enable_mock(True)

response = ask("Hello!")
print(f"Mock response: {response}")
```

## Azure Setup

For Azure OpenAI deployment:

### 1. Create Azure Resources

```powershell
# Create resource group
az group create --name rg-pyai --location eastus2

# Create Azure OpenAI resource
az cognitiveservices account create \
  --name pyai-openai \
  --resource-group rg-pyai \
  --kind OpenAI \
  --sku S0 \
  --location eastus2

# Deploy a model
az cognitiveservices account deployment create \
  --name pyai-openai \
  --resource-group rg-pyai \
  --deployment-name gpt-4o-mini \
  --model-name gpt-4o-mini \
  --model-version "2024-07-18" \
  --model-format OpenAI \
  --sku-capacity 10 \
  --sku-name Standard
```

### 2. Configure pyai

```python
from pyai.easy.config import config

# Option 1: Using Azure AD (recommended)
config.use_azure(
    endpoint="https://pyai-openai.openai.azure.com",
    deployment="gpt-4o-mini",
    api_version="2024-02-15-preview"
)

# Option 2: Using API key
config.use_azure(
    endpoint="https://pyai-openai.openai.azure.com",
    deployment="gpt-4o-mini",
    api_key="your-api-key"
)
```

### 3. Use pyai

```python
from pyai import ask

# Now uses Azure OpenAI
response = ask("What is Azure?")
print(response)
```

## Publishing to PyPI

### 1. Prepare for Release

```bash
# Update version in pyproject.toml
# Update CHANGELOG.md
# Commit all changes
git add -A
git commit -m "Release vX.X.X"
git tag vX.X.X
git push origin main --tags
```

### 2. Build Distribution

```powershell
.\build.ps1 clean
.\build.ps1 build
```

### 3. Upload to PyPI

```bash
# Install twine
pip install twine

# Upload to Test PyPI first
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ pyai

# If successful, upload to production PyPI
twine upload dist/*
```

## Enterprise Distribution

For enterprise/internal distribution:

### Azure DevOps Artifacts

```yaml
# azure-pipelines.yml
trigger:
  - main

pool:
  vmImage: 'ubuntu-latest'

steps:
  - task: UsePythonVersion@0
    inputs:
      versionSpec: '3.11'

  - script: |
      pip install build twine
      python -m build
    displayName: 'Build Package'

  - task: TwineAuthenticate@1
    inputs:
      artifactFeed: 'MyOrg/MyFeed'

  - script: |
      twine upload -r MyFeed --config-file $(PYPIRC_PATH) dist/*
    displayName: 'Publish to Artifacts'
```

### Local Package Repository

```bash
# Create local package folder
mkdir /path/to/packages

# Copy wheel file
cp dist/pyai-*.whl /path/to/packages/

# Install from local
pip install --find-links=/path/to/packages pyai
```

## Troubleshooting

### Common Issues

**Import Error: No module named 'pyai'**
- Ensure you installed in the correct Python environment
- Check: `pip show pyai`

**Azure Authentication Error**
- Run `az login` to authenticate
- Ensure `azure-identity` is installed: `pip install azure-identity`

**Optional dependency not found**
- Install the specific extra: `pip install pyai[extra_name]`

### Getting Help

- GitHub Issues: https://github.com/gitpavleenbali/pyai/issues
- Documentation: See `/docs` folder
- Examples: See `/examples` folder
