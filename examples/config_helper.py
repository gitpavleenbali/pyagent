"""
PyAgent Examples Configuration Helper
======================================

Provides dual authentication support for all examples:
1. OpenAI API Key (OPENAI_API_KEY)
2. Azure OpenAI with API Key (AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY)
3. Azure OpenAI with Azure AD (AZURE_OPENAI_ENDPOINT only - uses DefaultAzureCredential)

Usage in examples:
    from config_helper import setup_pyagent
    
    if not setup_pyagent():
        print("Please configure credentials - see instructions above")
        exit(1)
    
    # Now use pyagent normally
    from pyagent import ask
    print(ask("Hello!"))

Environment Variables:
    # Option 1: OpenAI
    export OPENAI_API_KEY=sk-your-key
    
    # Option 2: Azure OpenAI with API Key
    export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
    export AZURE_OPENAI_API_KEY=your-azure-key
    export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini  # optional, defaults to gpt-4o-mini
    
    # Option 3: Azure OpenAI with Azure AD (recommended for enterprise)
    export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
    export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini  # optional
    # No API key needed - uses az login credentials automatically!
"""

import os
import sys

# Add paths for local development (works from any directory, including PyCharm)
_examples_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_examples_dir)
sys.path.insert(0, _project_dir)  # For pyagent imports


def setup_pyagent(provider: str = "auto", verbose: bool = True) -> bool:
    """
    Configure PyAgent with available credentials.
    
    Args:
        provider: "auto", "azure", or "openai"
        verbose: Print configuration status
        
    Returns:
        True if configured successfully, False otherwise
        
    Examples:
        # Auto-detect (recommended)
        setup_pyagent()
        
        # Force Azure
        setup_pyagent(provider="azure")
        
        # Force OpenAI
        setup_pyagent(provider="openai")
    """
    import pyagent
    
    # Auto-detect provider
    if provider == "auto":
        if os.environ.get("AZURE_OPENAI_ENDPOINT"):
            provider = "azure"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            if verbose:
                print_setup_instructions()
            return False
    
    if provider == "azure":
        return _setup_azure(pyagent, verbose)
    else:
        return _setup_openai(pyagent, verbose)


def _setup_azure(pyagent, verbose: bool) -> bool:
    """Configure Azure OpenAI (API key or Azure AD)."""
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    
    if not endpoint:
        if verbose:
            print("[X] AZURE_OPENAI_ENDPOINT not set")
            print_setup_instructions()
        return False
    
    # Try API key first
    if api_key:
        pyagent.configure(
            provider="azure",
            api_key=api_key,
            azure_endpoint=endpoint,
            model=deployment
        )
        if verbose:
            print("[OK] Azure OpenAI configured (API Key)")
            print(f"     Endpoint: {endpoint}")
            print(f"     Deployment: {deployment}")
        return True
    
    # Try Azure AD / DefaultAzureCredential
    try:
        from azure.identity import DefaultAzureCredential
        credential = DefaultAzureCredential()
        # Validate credentials work
        credential.get_token("https://cognitiveservices.azure.com/.default")
        
        pyagent.configure(
            provider="azure",
            azure_endpoint=endpoint,
            model=deployment
        )
        if verbose:
            print("[OK] Azure OpenAI configured (Azure AD - DefaultAzureCredential)")
            print(f"     Endpoint: {endpoint}")
            print(f"     Deployment: {deployment}")
            print("     Auth: Using your Azure login (az login / VS Code)")
        return True
        
    except ImportError:
        if verbose:
            print("[X] azure-identity package not installed")
            print("    Run: pip install azure-identity")
        return False
    except Exception as e:
        if verbose:
            print(f"[X] Azure AD authentication failed: {e}")
            print("    Try: az login")
        return False


def _setup_openai(pyagent, verbose: bool) -> bool:
    """Configure OpenAI with API key."""
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    
    if not api_key:
        if verbose:
            print("[X] OPENAI_API_KEY not set")
            print_setup_instructions()
        return False
    
    pyagent.configure(
        provider="openai",
        api_key=api_key,
        model=model
    )
    if verbose:
        print("[OK] OpenAI configured")
        print(f"     Model: {model}")
    return True


def print_setup_instructions():
    """Print setup instructions for users."""
    print("""
+======================================================================+
|                    PyAgent Configuration Required                     |
+======================================================================+
|                                                                       |
|  Option 1: OpenAI API Key                                            |
|  -------------------------                                           |
|    export OPENAI_API_KEY=sk-your-key                                 |
|                                                                       |
|  Option 2: Azure OpenAI with API Key                                 |
|  -----------------------------------                                 |
|    export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
|    export AZURE_OPENAI_API_KEY=your-azure-key                        |
|    export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini  # optional            |
|                                                                       |
|  Option 3: Azure OpenAI with Azure AD (Enterprise - Recommended)     |
|  ---------------------------------------------------------------     |
|    az login                                                          |
|    export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
|    export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini  # optional            |
|    # No API key needed! Uses your Azure login automatically          |
|                                                                       |
+======================================================================+
""")


# Run if executed directly
if __name__ == "__main__":
    print("Testing PyAgent configuration...\n")
    
    if setup_pyagent():
        print("\n[OK] Configuration successful! Running quick test...\n")
        from pyagent import ask
        response = ask("Say 'Hello from PyAgent!' in exactly those words")
        print(f"Response: {response}")
    else:
        print("\n[X] Configuration failed. Please set up credentials.")
