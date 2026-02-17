# pyai Azure Setup Guide

This guide helps you configure pyai with Azure OpenAI.

## Quick Setup

### 1. Set Environment Variables

```bash
# PowerShell
$env:AZURE_OPENAI_API_KEY = "your-api-key"
$env:AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
$env:AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"  # Your deployment name

# Bash
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini"
```

### 2. Configure pyai

```python
import pyai

pyai.configure(
    provider="azure",
    api_key="your-api-key",  # Or use env var
    azure_endpoint="https://your-resource.openai.azure.com/",
    model="gpt-4o-mini"  # Your deployment name
)
```

### 3. Use pyai

```python
from pyai import ask

# Now it uses Azure OpenAI!
answer = ask("What is machine learning?")
print(answer)
```

## Available Azure Resources

Based on your subscription, you have these Azure OpenAI resources:

| Resource | Endpoint |
|----------|----------|
| openai-varcvenlme53e | https://openai-varcvenlme53e.openai.azure.com/ |
| cs-openai-varcvenlme53e | https://cs-openai-varcvenlme53e.cognitiveservices.azure.com/ |
| cost-intelligence-agent-resource | https://cost-intelligence-agent-resource.cognitiveservices.azure.com/ |

## Getting Your API Key

1. Go to Azure Portal → Your OpenAI Resource → Keys and Endpoint
2. Copy Key 1 or Key 2
3. Set as AZURE_OPENAI_API_KEY

## Getting Deployment Names

1. Go to Azure Portal → Your OpenAI Resource → Model Deployments
2. Or use Azure OpenAI Studio → Deployments
3. Common deployments: gpt-4, gpt-4o, gpt-4o-mini, gpt-35-turbo

## Full Example

```python
import pyai
from pyai import ask, agent, rag

# Configure Azure
pyai.configure(
    provider="azure",
    azure_endpoint="https://openai-varcvenlme53e.openai.azure.com/",
    model="gpt-4o-mini",  # Your deployment name
    # api_key set via AZURE_OPENAI_API_KEY env var
)

# Now use pyai normally!
answer = ask("What is Python?")
print(answer)

# Create agents
coder = agent(persona="coder")
code = coder("Write a function to sort a list")

# RAG
docs = rag.index(["Your documents here"])
answer = docs.ask("Question about documents")
```

## Troubleshooting

### Error: "AuthenticationError"
- Check your API key is correct
- Ensure the key is for the correct resource

### Error: "ResourceNotFoundError"
- Verify the endpoint URL
- Check the deployment name exists

### Error: "DeploymentNotFound"
- Go to Azure OpenAI Studio and create a deployment
- Use the exact deployment name in configuration
