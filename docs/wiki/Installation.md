# Installation

Multiple installation options to fit your needs.

---

## Basic Installation

```bash
pip install pyai
```

---

## With Specific Providers

### OpenAI
```bash
pip install pyai[openai]
```

### Anthropic Claude
```bash
pip install pyai[anthropic]
```

### Azure OpenAI (Recommended for Enterprise)
```bash
pip install pyai[azure]
```

Includes:
- Azure Identity (Azure AD authentication)
- Azure AI Search integration
- OpenAI SDK for Azure

---

## With Integrations

### LangChain
```bash
pip install pyai[langchain]
```

### Microsoft Semantic Kernel
```bash
pip install pyai[semantic-kernel]
```

### Vector Databases
```bash
pip install pyai[vector]
```

Includes: ChromaDB, FAISS, Pinecone, Qdrant

### Web & Documents
```bash
pip install pyai[web]      # aiohttp, requests, beautifulsoup
pip install pyai[docs]     # PDF, DOCX processing
```

---

## Full Installation

```bash
pip install pyai[all]
```

Installs everything: all providers, integrations, and utilities.

---

## Development Installation

```bash
pip install pyai[dev]
```

Includes: pytest, black, ruff, mypy, pre-commit

---

## Installation Extras Summary

| Extra | Packages Included |
|-------|-------------------|
| `openai` | OpenAI SDK |
| `anthropic` | Anthropic Claude SDK |
| `azure` | Azure Identity, AI Search, OpenAI |
| `langchain` | LangChain integration |
| `semantic-kernel` | Microsoft Semantic Kernel |
| `vector` | ChromaDB, FAISS, Pinecone, Qdrant |
| `web` | aiohttp, requests, beautifulsoup4 |
| `docs` | pypdf, python-docx |
| `all` | Everything above |
| `dev` | pytest, black, ruff, mypy |

---

## From Source

```bash
git clone https://github.com/gitpavleenbali/PYAI.git
cd PYAI
pip install -e ".[dev]"
```

---

## Verify Installation

```python
import pyai
print(pyai.__version__)  # 0.4.0
```

---

## Next Steps

- [[Configuration]] - Set up API keys
- [[Quick Start]] - Your first PYAI program
