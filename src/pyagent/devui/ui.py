# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
Development UI - Web Interface

Provides a browser-based interface for testing and debugging agents.
"""

import json
import threading
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse, parse_qs
import html


@dataclass
class ChatMessage:
    """A message in the chat interface."""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DevUI:
    """Development UI for testing agents.
    
    Provides a web-based chat interface with:
    - Real-time conversation
    - Message history
    - Tool call visualization
    - Token usage tracking
    - Configuration panel
    
    Example:
        from pyagent import Agent
        from pyagent.devui import DevUI
        
        agent = Agent(model="gpt-4")
        ui = DevUI(agent)
        ui.launch()  # Opens http://localhost:7860
    """
    
    def __init__(
        self,
        agent: Optional[Any] = None,
        handler: Optional[Callable[[str], str]] = None,
        title: str = "PyAgent Dev UI",
        description: str = "Test and debug your agent",
    ):
        """Initialize Dev UI.
        
        Args:
            agent: PyAgent Agent instance
            handler: Custom message handler
            title: UI title
            description: UI description
        """
        self.agent = agent
        self.handler = handler
        self.title = title
        self.description = description
        
        self._messages: List[ChatMessage] = []
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
    
    def _handle_message(self, content: str) -> str:
        """Handle an incoming message.
        
        Args:
            content: User message
            
        Returns:
            Agent response
        """
        # Add user message
        self._messages.append(ChatMessage(role="user", content=content))
        
        try:
            if self.handler:
                response = self.handler(content)
            elif self.agent:
                result = self.agent.run(content)
                if hasattr(result, "output"):
                    response = result.output
                elif isinstance(result, str):
                    response = result
                else:
                    response = str(result)
            else:
                response = "No agent or handler configured."
        except Exception as e:
            response = f"Error: {str(e)}"
        
        # Add assistant message
        self._messages.append(ChatMessage(role="assistant", content=response))
        
        return response
    
    def _generate_html(self) -> str:
        """Generate the HTML interface."""
        messages_html = ""
        for msg in self._messages:
            role_class = "user" if msg.role == "user" else "assistant"
            content_escaped = html.escape(msg.content).replace("\n", "<br>")
            messages_html += f'''
            <div class="message {role_class}">
                <div class="role">{msg.role.title()}</div>
                <div class="content">{content_escaped}</div>
            </div>
            '''
        
        return f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(self.title)}</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #1a1a2e;
            color: #eee;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        header {{
            background: #16213e;
            padding: 20px;
            border-bottom: 1px solid #0f3460;
        }}
        h1 {{
            font-size: 1.5rem;
            color: #e94560;
        }}
        .description {{
            color: #888;
            font-size: 0.9rem;
            margin-top: 5px;
        }}
        .chat-container {{
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }}
        .message {{
            max-width: 80%;
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 10px;
        }}
        .message.user {{
            background: #0f3460;
            margin-left: auto;
        }}
        .message.assistant {{
            background: #16213e;
        }}
        .role {{
            font-size: 0.75rem;
            color: #e94560;
            margin-bottom: 5px;
            text-transform: uppercase;
        }}
        .content {{
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .input-container {{
            padding: 20px;
            background: #16213e;
            border-top: 1px solid #0f3460;
            display: flex;
            gap: 10px;
        }}
        input[type="text"] {{
            flex: 1;
            padding: 15px;
            border: 1px solid #0f3460;
            border-radius: 8px;
            background: #1a1a2e;
            color: #eee;
            font-size: 1rem;
        }}
        input[type="text"]:focus {{
            outline: none;
            border-color: #e94560;
        }}
        button {{
            padding: 15px 30px;
            background: #e94560;
            border: none;
            border-radius: 8px;
            color: white;
            font-size: 1rem;
            cursor: pointer;
            transition: background 0.2s;
        }}
        button:hover {{
            background: #ff6b6b;
        }}
        button:disabled {{
            background: #555;
            cursor: not-allowed;
        }}
        .loading {{
            display: none;
            color: #888;
            padding: 10px;
        }}
        .empty-state {{
            text-align: center;
            color: #666;
            padding: 50px;
        }}
    </style>
</head>
<body>
    <header>
        <h1>{html.escape(self.title)}</h1>
        <p class="description">{html.escape(self.description)}</p>
    </header>
    
    <div class="chat-container" id="chat">
        {messages_html if messages_html else '<div class="empty-state">Send a message to start chatting</div>'}
    </div>
    
    <div class="input-container">
        <input type="text" id="input" placeholder="Type your message..." autofocus>
        <button onclick="sendMessage()" id="sendBtn">Send</button>
    </div>
    
    <div class="loading" id="loading">Thinking...</div>
    
    <script>
        const input = document.getElementById('input');
        const chat = document.getElementById('chat');
        const sendBtn = document.getElementById('sendBtn');
        const loading = document.getElementById('loading');
        
        input.addEventListener('keypress', function(e) {{
            if (e.key === 'Enter') {{
                sendMessage();
            }}
        }});
        
        async function sendMessage() {{
            const message = input.value.trim();
            if (!message) return;
            
            input.value = '';
            sendBtn.disabled = true;
            loading.style.display = 'block';
            
            // Add user message immediately
            addMessage('user', message);
            
            try {{
                const response = await fetch('/chat', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{message: message}})
                }});
                
                const data = await response.json();
                addMessage('assistant', data.response);
            }} catch (error) {{
                addMessage('assistant', 'Error: ' + error.message);
            }}
            
            sendBtn.disabled = false;
            loading.style.display = 'none';
            input.focus();
        }}
        
        function addMessage(role, content) {{
            const div = document.createElement('div');
            div.className = 'message ' + role;
            div.innerHTML = `
                <div class="role">${{role.charAt(0).toUpperCase() + role.slice(1)}}</div>
                <div class="content">${{escapeHtml(content).replace(/\\n/g, '<br>')}}</div>
            `;
            
            // Remove empty state if present
            const empty = chat.querySelector('.empty-state');
            if (empty) empty.remove();
            
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }}
        
        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}
    </script>
</body>
</html>
'''
    
    def _create_handler(self) -> type:
        """Create request handler class."""
        ui = self
        
        class DevUIHandler(SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress logging
            
            def do_GET(self):
                if self.path == "/" or self.path == "/index.html":
                    html_content = ui._generate_html()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(html_content.encode("utf-8"))
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def do_POST(self):
                if self.path == "/chat":
                    content_length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(content_length)
                    
                    try:
                        data = json.loads(body)
                        message = data.get("message", "")
                        response = ui._handle_message(message)
                        
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({
                            "response": response
                        }).encode("utf-8"))
                    except Exception as e:
                        self.send_response(500)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({
                            "error": str(e)
                        }).encode("utf-8"))
                else:
                    self.send_response(404)
                    self.end_headers()
        
        return DevUIHandler
    
    def launch(
        self,
        host: str = "127.0.0.1",
        port: int = 7860,
        open_browser: bool = True,
        block: bool = True,
    ):
        """Launch the development UI.
        
        Args:
            host: Host to bind to
            port: Port to listen on
            open_browser: Open browser automatically
            block: Block until server stops
        """
        handler = self._create_handler()
        self._server = HTTPServer((host, port), handler)
        
        url = f"http://{host}:{port}"
        print(f"🚀 PyAgent Dev UI running at {url}")
        
        if open_browser:
            webbrowser.open(url)
        
        if block:
            try:
                self._server.serve_forever()
            except KeyboardInterrupt:
                print("\n👋 Shutting down...")
                self._server.shutdown()
        else:
            self._thread = threading.Thread(target=self._server.serve_forever)
            self._thread.daemon = True
            self._thread.start()
    
    def stop(self):
        """Stop the UI server."""
        if self._server:
            self._server.shutdown()
    
    def clear_history(self):
        """Clear chat history."""
        self._messages.clear()


def launch_ui(
    agent: Optional[Any] = None,
    handler: Optional[Callable[[str], str]] = None,
    title: str = "PyAgent Dev UI",
    port: int = 7860,
    **kwargs
):
    """Launch development UI quickly.
    
    Args:
        agent: PyAgent Agent instance
        handler: Custom message handler
        title: UI title
        port: Port number
        **kwargs: Additional DevUI arguments
        
    Example:
        from pyagent import Agent
        from pyagent.devui import launch_ui
        
        agent = Agent(model="gpt-4")
        launch_ui(agent)
    """
    ui = DevUI(agent=agent, handler=handler, title=title, **kwargs)
    ui.launch(port=port)
