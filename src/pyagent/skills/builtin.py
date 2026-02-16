"""
Built-in Skills - Common skill implementations
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from pyagent.skills.skill import Skill, SkillResult, SkillParameter
from pyagent.skills.action_skill import ActionSkill, action, ActionType


class SearchSkill(Skill):
    """
    SearchSkill - Search for information.
    
    A general-purpose search skill that can be extended
    with different search backends.
    
    Example:
        >>> skill = SearchSkill()
        >>> result = await skill.execute(query="python tutorials")
    """
    
    name = "search"
    description = "Search for information on a topic"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameters = [
            SkillParameter(
                name="query",
                description="The search query",
                type="string",
                required=True,
            ),
            SkillParameter(
                name="limit",
                description="Maximum number of results",
                type="integer",
                required=False,
                default=5,
            ),
        ]
    
    async def execute(self, query: str, limit: int = 5, **kwargs) -> SkillResult:
        """Execute a search"""
        # This is a placeholder - real implementation would call a search API
        return SkillResult.ok(
            data={"results": [], "query": query},
            message=f"Search for '{query}' (implement search backend)"
        )


class CodeSkill(ActionSkill):
    """
    CodeSkill - Code-related actions.
    
    Provides actions for code execution, analysis, and generation.
    """
    
    name = "code"
    description = "Execute code-related actions"
    
    @action("execute", description="Execute Python code", action_type=ActionType.EXECUTE)
    async def execute_code(self, code: str) -> str:
        """Execute Python code and return the result"""
        # Safety: In production, this should use sandboxing
        try:
            # Create a restricted namespace
            namespace = {"__builtins__": {}}
            exec(code, namespace)
            return str(namespace.get("result", "Code executed"))
        except Exception as e:
            return f"Error: {e}"
    
    @action("analyze", description="Analyze code structure", action_type=ActionType.ANALYZE)
    async def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze Python code and return structure info"""
        import ast
        
        try:
            tree = ast.parse(code)
            
            functions = []
            classes = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(ast.dump(node))
            
            return {
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "line_count": len(code.split('\n')),
            }
        except SyntaxError as e:
            return {"error": str(e)}
    
    @action("format", description="Format Python code", action_type=ActionType.TRANSFORM)
    async def format_code(self, code: str) -> str:
        """Format Python code"""
        try:
            import black
            return black.format_str(code, mode=black.Mode())
        except ImportError:
            return code  # Return unformatted if black not installed
        except Exception as e:
            return code


class FileSkill(ActionSkill):
    """
    FileSkill - File system operations.
    
    Provides actions for reading, writing, and managing files.
    """
    
    name = "file"
    description = "Perform file system operations"
    
    @action("read", description="Read file contents", action_type=ActionType.READ)
    async def read_file(self, path: str, encoding: str = "utf-8") -> str:
        """Read and return file contents"""
        try:
            with open(path, 'r', encoding=encoding) as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {path}")
        except Exception as e:
            raise Exception(f"Error reading file: {e}")
    
    @action("write", description="Write content to file", action_type=ActionType.WRITE)
    async def write_file(self, path: str, content: str, encoding: str = "utf-8") -> bool:
        """Write content to a file"""
        try:
            with open(path, 'w', encoding=encoding) as f:
                f.write(content)
            return True
        except Exception as e:
            raise Exception(f"Error writing file: {e}")
    
    @action("list", description="List directory contents", action_type=ActionType.READ)
    async def list_directory(self, path: str = ".") -> List[str]:
        """List files in a directory"""
        import os
        try:
            return os.listdir(path)
        except Exception as e:
            raise Exception(f"Error listing directory: {e}")
    
    @action("exists", description="Check if path exists", action_type=ActionType.READ)
    async def path_exists(self, path: str) -> bool:
        """Check if a path exists"""
        import os
        return os.path.exists(path)


class WebSkill(ActionSkill):
    """
    WebSkill - Web-related operations.
    
    Provides actions for fetching web content and making HTTP requests.
    """
    
    name = "web"
    description = "Perform web operations"
    
    @action("fetch", description="Fetch content from URL", action_type=ActionType.READ)
    async def fetch(self, url: str, method: str = "GET") -> Dict[str, Any]:
        """Fetch content from a URL"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url) as response:
                    content = await response.text()
                    return {
                        "status": response.status,
                        "content": content[:10000],  # Limit content size
                        "headers": dict(response.headers),
                    }
        except ImportError:
            # Fallback to urllib
            import urllib.request
            with urllib.request.urlopen(url) as response:
                return {
                    "status": response.status,
                    "content": response.read().decode()[:10000],
                }
    
    @action("parse_html", description="Parse HTML content", action_type=ActionType.ANALYZE)
    async def parse_html(self, html: str) -> Dict[str, Any]:
        """Parse HTML and extract structure"""
        try:
            from html.parser import HTMLParser
            
            class SimpleHTMLParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.links = []
                    self.text = []
                
                def handle_starttag(self, tag, attrs):
                    if tag == 'a':
                        for attr, value in attrs:
                            if attr == 'href':
                                self.links.append(value)
                
                def handle_data(self, data):
                    text = data.strip()
                    if text:
                        self.text.append(text)
            
            parser = SimpleHTMLParser()
            parser.feed(html)
            
            return {
                "links": parser.links[:50],  # Limit
                "text_preview": " ".join(parser.text)[:1000],
            }
        except Exception as e:
            return {"error": str(e)}


class MathSkill(Skill):
    """
    MathSkill - Mathematical computations.
    
    Provides safe mathematical expression evaluation.
    """
    
    name = "math"
    description = "Evaluate mathematical expressions"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameters = [
            SkillParameter(
                name="expression",
                description="The mathematical expression to evaluate",
                type="string",
                required=True,
            ),
        ]
        
        # Safe math functions
        self._safe_functions = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'pow': pow,
            'sqrt': lambda x: x ** 0.5,
            'pi': 3.141592653589793,
            'e': 2.718281828459045,
        }
    
    async def execute(self, expression: str, **kwargs) -> SkillResult:
        """Evaluate a math expression safely"""
        try:
            # Use a restricted eval
            result = eval(
                expression,
                {"__builtins__": {}},
                self._safe_functions
            )
            return SkillResult.ok(
                data={"result": result, "expression": expression}
            )
        except Exception as e:
            return SkillResult.fail(f"Math error: {e}")


# Skill factory functions
def create_search_skill(**config) -> SearchSkill:
    """Factory function for creating search skills"""
    return SearchSkill(**config)


def create_code_skill(**config) -> CodeSkill:
    """Factory function for creating code skills"""
    return CodeSkill(**config)


def create_file_skill(**config) -> FileSkill:
    """Factory function for creating file skills"""
    return FileSkill(**config)


def create_web_skill(**config) -> WebSkill:
    """Factory function for creating web skills"""
    return WebSkill(**config)


def create_math_skill(**config) -> MathSkill:
    """Factory function for creating math skills"""
    return MathSkill(**config)
