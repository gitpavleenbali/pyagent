# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
PyAgent CLI Main Entry Point

Provides CLI commands similar to Google ADK's `adk` CLI:
- pyagent run: Run an agent interactively
- pyagent eval: Evaluate agent on test cases  
- pyagent serve: Start API server
- pyagent init: Initialize new project
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Version info
try:
    from .. import __version__
except ImportError:
    __version__ = "0.2.0"


def load_agent_from_file(filepath: str) -> Any:
    """Load an agent from a Python file or YAML.
    
    Args:
        filepath: Path to agent definition
        
    Returns:
        Loaded agent instance
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"Agent file not found: {filepath}")
    
    if path.suffix in (".yaml", ".yml"):
        # Load from YAML
        try:
            import yaml
            with open(path) as f:
                config = yaml.safe_load(f)
            
            # Import Agent and create from config
            from ..core import Agent
            return Agent.from_dict(config) if hasattr(Agent, "from_dict") else Agent(**config)
        except ImportError:
            raise ImportError("YAML support requires pyyaml: pip install pyyaml")
    
    elif path.suffix == ".py":
        # Load from Python module
        spec = importlib.util.spec_from_file_location("agent_module", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["agent_module"] = module
        spec.loader.exec_module(module)
        
        # Look for agent in module
        if hasattr(module, "agent"):
            return module.agent
        elif hasattr(module, "Agent"):
            return module.Agent()
        elif hasattr(module, "main"):
            return module.main
        else:
            # Find any Agent-like object
            from ..core import Agent
            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, Agent):
                    return obj
            
            raise ValueError(f"No agent found in {filepath}. Define 'agent = Agent(...)' in your file.")
    
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def cmd_run(args: argparse.Namespace) -> int:
    """Run an agent interactively.
    
    Example:
        pyagent run my_agent.py
        pyagent run config.yaml --input "Hello"
    """
    print(f"🚀 PyAgent CLI v{__version__}")
    
    try:
        agent = load_agent_from_file(args.agent)
        print(f"✅ Loaded agent from {args.agent}")
        
        if args.input:
            # Single input mode
            result = agent.run(args.input) if hasattr(agent, "run") else agent(args.input)
            output = result.output if hasattr(result, "output") else str(result)
            print(f"\n📤 Output:\n{output}")
        else:
            # Interactive REPL mode
            print("💬 Interactive mode (type 'exit' to quit)")
            print("-" * 40)
            
            while True:
                try:
                    user_input = input("\n🧑 You: ").strip()
                    
                    if user_input.lower() in ("exit", "quit", "q"):
                        print("👋 Goodbye!")
                        break
                    
                    if not user_input:
                        continue
                    
                    result = agent.run(user_input) if hasattr(agent, "run") else agent(user_input)
                    output = result.output if hasattr(result, "output") else str(result)
                    
                    print(f"\n🤖 Agent: {output}")
                    
                except KeyboardInterrupt:
                    print("\n👋 Interrupted. Goodbye!")
                    break
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate an agent on test cases.
    
    Example:
        pyagent eval tests.evalset.json --agent my_agent.py
        pyagent eval tests/*.json --parallel
    """
    print(f"🧪 PyAgent Evaluation CLI v{__version__}")
    
    try:
        # Load eval set
        from ..evaluation import EvalSet, Evaluator, EvalConfig
        
        eval_set = EvalSet.from_file(args.evalset)
        print(f"📋 Loaded {len(eval_set)} test cases from {args.evalset}")
        
        # Load agent
        agent = load_agent_from_file(args.agent)
        print(f"✅ Loaded agent from {args.agent}")
        
        # Configure evaluation
        config = EvalConfig(
            parallel=not args.sequential,
            max_workers=args.workers,
            verbose=args.verbose,
            fail_fast=args.fail_fast,
        )
        
        # Run evaluation
        evaluator = Evaluator(agent, config)
        metrics = evaluator.evaluate(eval_set, tags=args.tags.split(",") if args.tags else None)
        
        # Output results
        print("\n" + "=" * 50)
        print(metrics.summary())
        
        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(metrics.to_dict(), f, indent=2)
            print(f"\n💾 Results saved to {args.output}")
        
        # Return exit code based on results
        return 0 if metrics.failed == 0 and metrics.errors == 0 else 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_serve(args: argparse.Namespace) -> int:
    """Start an API server for the agent.
    
    Example:
        pyagent serve my_agent.py --port 8000
    """
    print(f"🌐 PyAgent API Server v{__version__}")
    
    try:
        agent = load_agent_from_file(args.agent)
        print(f"✅ Loaded agent from {args.agent}")
        
        # Try FastAPI first
        try:
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
            import uvicorn
            
            app = FastAPI(
                title="PyAgent API",
                version=__version__,
                description="AI Agent API powered by PyAgent"
            )
            
            class ChatRequest(BaseModel):
                message: str
                session_id: str = None
            
            class ChatResponse(BaseModel):
                response: str
                session_id: str = None
            
            @app.post("/chat", response_model=ChatResponse)
            async def chat(request: ChatRequest):
                try:
                    result = agent.run(request.message) if hasattr(agent, "run") else agent(request.message)
                    output = result.output if hasattr(result, "output") else str(result)
                    return ChatResponse(response=output, session_id=request.session_id)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
            
            @app.get("/health")
            async def health():
                return {"status": "healthy", "version": __version__}
            
            print(f"🚀 Starting server at http://{args.host}:{args.port}")
            print(f"📖 API docs at http://{args.host}:{args.port}/docs")
            
            uvicorn.run(app, host=args.host, port=args.port)
            return 0
            
        except ImportError:
            # Fall back to Flask
            try:
                from flask import Flask, request, jsonify
                
                app = Flask(__name__)
                
                @app.route("/chat", methods=["POST"])
                def chat():
                    data = request.json
                    result = agent.run(data["message"]) if hasattr(agent, "run") else agent(data["message"])
                    output = result.output if hasattr(result, "output") else str(result)
                    return jsonify({"response": output})
                
                @app.route("/health")
                def health():
                    return jsonify({"status": "healthy", "version": __version__})
                
                print(f"🚀 Starting server at http://{args.host}:{args.port}")
                app.run(host=args.host, port=args.port)
                return 0
                
            except ImportError:
                print("❌ Server requires FastAPI or Flask. Install with:")
                print("   pip install fastapi uvicorn")
                print("   # or")
                print("   pip install flask")
                return 1
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize a new PyAgent project.
    
    Example:
        pyagent init my-agent
        pyagent init . --template research
    """
    print(f"🎉 PyAgent Project Initializer v{__version__}")
    
    project_dir = Path(args.name)
    
    if project_dir.exists() and any(project_dir.iterdir()):
        if not args.force:
            print(f"❌ Directory {args.name} already exists and is not empty")
            print("   Use --force to overwrite")
            return 1
    
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Create project structure
    templates = {
        "basic": {
            "agent.py": '''"""My PyAgent Agent"""
from pyagent import Agent

agent = Agent(
    name="my-agent",
    instructions="You are a helpful AI assistant.",
)

if __name__ == "__main__":
    result = agent.run("Hello!")
    print(result.output)
''',
            "requirements.txt": "pyagent>=0.2.0\n",
            "README.md": f"# {args.name}\n\nAI Agent built with PyAgent.\n\n## Usage\n\n```bash\npyagent run agent.py\n```\n",
        },
        "research": {
            "agent.py": '''"""Research Agent with Tools"""
from pyagent import Agent
from pyagent.easy import research, fetch, summarize

agent = Agent(
    name="research-agent",
    instructions="You are a research assistant that can search and analyze information.",
)

# Add research capabilities
agent.add_skill(research)
agent.add_skill(fetch)
agent.add_skill(summarize)

if __name__ == "__main__":
    result = agent.run("Research the latest AI developments")
    print(result.output)
''',
            "tests.evalset.json": json.dumps({
                "name": "research-tests",
                "test_cases": [
                    {"input": "What is machine learning?", "expected_contains": ["algorithm", "data", "learn"]}
                ]
            }, indent=2),
            "requirements.txt": "pyagent>=0.2.0\nrequests\n",
            "README.md": f"# {args.name}\n\nResearch Agent built with PyAgent.\n",
        },
    }
    
    template = templates.get(args.template, templates["basic"])
    
    for filename, content in template.items():
        filepath = project_dir / filename
        filepath.write_text(content)
        print(f"  📄 Created {filepath}")
    
    print(f"\n✅ Project initialized in {project_dir}")
    print(f"\n📝 Next steps:")
    print(f"   cd {args.name}")
    print(f"   pip install -r requirements.txt")
    print(f"   pyagent run agent.py")
    
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show PyAgent information."""
    print(f"PyAgent v{__version__}")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Show installed dependencies
    deps = ["openai", "azure-identity", "anthropic", "google-generativeai", "litellm", "httpx"]
    print("\nInstalled dependencies:")
    for dep in deps:
        try:
            mod = importlib.import_module(dep.replace("-", "_"))
            version = getattr(mod, "__version__", "installed")
            print(f"  ✅ {dep}: {version}")
        except ImportError:
            print(f"  ⬜ {dep}: not installed")
    
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="pyagent",
        description="PyAgent CLI - Build and run AI agents",
        epilog="For more info: https://github.com/pyagent/pyagent"
    )
    parser.add_argument("-V", "--version", action="version", version=f"PyAgent {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run an agent interactively")
    run_parser.add_argument("agent", help="Path to agent file (.py or .yaml)")
    run_parser.add_argument("-i", "--input", help="Single input (non-interactive)")
    run_parser.add_argument("-v", "--verbose", action="store_true")
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate agent on test cases")
    eval_parser.add_argument("evalset", help="Path to .evalset.json file")
    eval_parser.add_argument("-a", "--agent", required=True, help="Path to agent file")
    eval_parser.add_argument("-o", "--output", help="Save results to JSON file")
    eval_parser.add_argument("-t", "--tags", help="Filter tests by tags (comma-separated)")
    eval_parser.add_argument("-w", "--workers", type=int, default=4, help="Parallel workers")
    eval_parser.add_argument("--sequential", action="store_true", help="Run sequentially")
    eval_parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    eval_parser.add_argument("-v", "--verbose", action="store_true")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("agent", help="Path to agent file")
    serve_parser.add_argument("-H", "--host", default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("-p", "--port", type=int, default=8000, help="Port number")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize new project")
    init_parser.add_argument("name", help="Project name or directory")
    init_parser.add_argument("-t", "--template", default="basic", 
                            choices=["basic", "research"], help="Project template")
    init_parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing")
    
    # Info command
    subparsers.add_parser("info", help="Show PyAgent info")
    
    return parser


def main(argv: list = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return 0
    
    commands = {
        "run": cmd_run,
        "eval": cmd_eval,
        "serve": cmd_serve,
        "init": cmd_init,
        "info": cmd_info,
    }
    
    return commands[args.command](args)


# For entry point
app = main


if __name__ == "__main__":
    sys.exit(main())
