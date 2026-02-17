# Azure AD Authentication

Enterprise-grade authentication without API keys.

---

## Overview

PYAI supports Azure AD authentication for Azure OpenAI, enabling:

- **No hardcoded API keys** in code or environment
- **Identity-based access** using your Azure credentials
- **Managed Identity support** for Azure-hosted applications
- **RBAC integration** for access control

---

## How It Works

PYAI uses Azure's `DefaultAzureCredential` which automatically tries:

1. **Environment credentials** (service principal)
2. **Managed Identity** (Azure VMs, App Service, AKS)
3. **Azure CLI** (`az login`)
4. **Visual Studio Code** (Azure extension)
5. **Azure PowerShell** (`Connect-AzAccount`)

---

## Setup

### Step 1: Configure Environment

```bash
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
# No API key needed!
```

### Step 2: Authenticate

Choose one method:

#### Option A: Azure CLI (Development)

```bash
az login
```

#### Option B: VS Code (Development)

- Install Azure Account extension
- Sign in via Command Palette: "Azure: Sign In"

#### Option C: Managed Identity (Production)

Enable on your Azure resource (VM, App Service, AKS, etc.)

#### Option D: Service Principal (CI/CD)

```bash
export AZURE_CLIENT_ID=your-client-id
export AZURE_CLIENT_SECRET=your-client-secret
export AZURE_TENANT_ID=your-tenant-id
```

### Step 3: Use PYAI

```python
from pyai import ask

# Just works! Uses your Azure credentials automatically
answer = ask("Hello, world!")
```

---

## Azure RBAC

Assign the **Cognitive Services OpenAI User** role:

```bash
az role assignment create \
  --assignee your-user@domain.com \
  --role "Cognitive Services OpenAI User" \
  --scope /subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.CognitiveServices/accounts/{account}
```

---

## Explicit Provider

```python
from pyai import Agent
from pyai.core import AzureOpenAIProvider, LLMConfig

# Azure AD authentication (no api_key)
provider = AzureOpenAIProvider(LLMConfig(
    api_base="https://your-resource.openai.azure.com/",
    model="gpt-4o-mini",
    api_version="2024-02-15-preview"
))

agent = Agent(
    name="EnterpriseBot",
    instructions="You are a helpful assistant.",
    llm=provider
)
```

---

## Managed Identity Example

For Azure-hosted applications:

```python
# app.py - No credentials in code!
import os
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://myorg.openai.azure.com/"
os.environ["AZURE_OPENAI_DEPLOYMENT"] = "gpt-4o-mini"

from pyai import ask

# Uses the app's Managed Identity
answer = ask("Process this request...")
```

---

## Troubleshooting

### "Authentication failed"

1. Verify you're logged in: `az account show`
2. Check role assignment: "Cognitive Services OpenAI User"
3. Verify endpoint format: `https://{name}.openai.azure.com/`

### "Deployment not found"

1. Check deployment name matches `AZURE_OPENAI_DEPLOYMENT`
2. Verify deployment is active in Azure Portal

### "Token expired"

1. Re-authenticate: `az login`
2. For VS Code: "Azure: Sign Out" then "Azure: Sign In"

---

## Benefits

| API Key Auth | Azure AD Auth |
|--------------|---------------|
| Key in environment | No key needed |
| Key rotation required | Automatic token refresh |
| Shared credentials | Identity-based |
| No audit trail | Full Azure audit logs |
| Manual revocation | RBAC control |

---

## Next Steps

- [[Configuration]] - Other configuration options
- [[Agent]] - Create agents with Azure OpenAI
- [[Sessions]] - Persistent conversation state
