# Copyright (c) 2026 PyAgent Contributors
# Licensed under the MIT License

"""
A2A Agent Registry

Discover and register agents in a distributed system.
"""

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .protocol import AgentCard
from .client import A2AClient


@dataclass
class RegisteredAgent:
    """A registered agent in the registry."""
    card: AgentCard
    url: str
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    healthy: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentRegistry:
    """Registry for discovering and managing agents.
    
    Supports multiple discovery mechanisms:
    - Static registration
    - File-based discovery
    - Environment variable discovery
    - Well-known endpoint scanning
    
    Example:
        registry = AgentRegistry()
        
        # Register an agent
        registry.register("http://agent1:8080")
        
        # Discover agents
        agents = registry.list()
        
        # Find by skill
        research_agents = registry.find_by_skill("research")
    """
    
    _default_instance: "AgentRegistry" = None
    
    def __init__(self):
        self._agents: Dict[str, RegisteredAgent] = {}
        self._lock = threading.RLock()
    
    def register(
        self,
        url: str,
        card: Optional[AgentCard] = None,
        **metadata
    ) -> RegisteredAgent:
        """Register an agent.
        
        Args:
            url: Agent URL
            card: Optional pre-fetched card
            **metadata: Additional metadata
            
        Returns:
            Registered agent
        """
        with self._lock:
            url = url.rstrip("/")
            
            # Fetch card if not provided
            if card is None:
                client = A2AClient(url)
                try:
                    card = client.get_card()
                except:
                    # Create minimal card
                    card = AgentCard(
                        name=url.split("/")[-1] or "unknown",
                        url=url,
                    )
            
            registered = RegisteredAgent(
                card=card,
                url=url,
                metadata=metadata,
            )
            
            self._agents[url] = registered
            return registered
    
    def unregister(self, url: str) -> bool:
        """Unregister an agent.
        
        Args:
            url: Agent URL
            
        Returns:
            True if unregistered
        """
        with self._lock:
            url = url.rstrip("/")
            if url in self._agents:
                del self._agents[url]
                return True
            return False
    
    def get(self, url: str) -> Optional[RegisteredAgent]:
        """Get a registered agent by URL."""
        with self._lock:
            return self._agents.get(url.rstrip("/"))
    
    def list(self, healthy_only: bool = False) -> List[RegisteredAgent]:
        """List all registered agents.
        
        Args:
            healthy_only: Only return healthy agents
            
        Returns:
            List of registered agents
        """
        with self._lock:
            agents = list(self._agents.values())
            if healthy_only:
                agents = [a for a in agents if a.healthy]
            return agents
    
    def find_by_name(self, name: str) -> List[RegisteredAgent]:
        """Find agents by name.
        
        Args:
            name: Agent name (partial match)
            
        Returns:
            Matching agents
        """
        with self._lock:
            name_lower = name.lower()
            return [
                a for a in self._agents.values()
                if name_lower in a.card.name.lower()
            ]
    
    def find_by_skill(self, skill: str) -> List[RegisteredAgent]:
        """Find agents by skill.
        
        Args:
            skill: Skill name (partial match)
            
        Returns:
            Matching agents
        """
        with self._lock:
            skill_lower = skill.lower()
            return [
                a for a in self._agents.values()
                if any(skill_lower in s.lower() for s in a.card.skills)
            ]
    
    def health_check(self, url: Optional[str] = None) -> Dict[str, bool]:
        """Check health of agents.
        
        Args:
            url: Specific URL to check, or None for all
            
        Returns:
            Map of URL to health status
        """
        with self._lock:
            if url:
                urls = [url.rstrip("/")]
            else:
                urls = list(self._agents.keys())
        
        results = {}
        for agent_url in urls:
            client = A2AClient(agent_url)
            healthy = client.health_check()
            results[agent_url] = healthy
            
            with self._lock:
                if agent_url in self._agents:
                    self._agents[agent_url].healthy = healthy
                    self._agents[agent_url].last_seen = datetime.utcnow()
        
        return results
    
    def discover_from_env(self, env_var: str = "PYAGENT_AGENTS") -> int:
        """Discover agents from environment variable.
        
        Environment variable should contain comma-separated URLs.
        
        Args:
            env_var: Environment variable name
            
        Returns:
            Number of agents discovered
        """
        urls_str = os.environ.get(env_var, "")
        if not urls_str:
            return 0
        
        urls = [u.strip() for u in urls_str.split(",") if u.strip()]
        count = 0
        
        for url in urls:
            try:
                self.register(url)
                count += 1
            except:
                pass
        
        return count
    
    def discover_from_file(self, path: str) -> int:
        """Discover agents from a JSON file.
        
        File format:
        {
            "agents": [
                {"url": "http://agent1:8080"},
                {"url": "http://agent2:8080", "name": "Research Agent"}
            ]
        }
        
        Args:
            path: Path to JSON file
            
        Returns:
            Number of agents discovered
        """
        if not os.path.exists(path):
            return 0
        
        with open(path, "r") as f:
            data = json.load(f)
        
        agents = data.get("agents", [])
        count = 0
        
        for agent_data in agents:
            url = agent_data.get("url")
            if not url:
                continue
            
            try:
                card = None
                if "name" in agent_data:
                    card = AgentCard(
                        name=agent_data.get("name", ""),
                        description=agent_data.get("description", ""),
                        url=url,
                        skills=agent_data.get("skills", []),
                    )
                
                self.register(url, card=card)
                count += 1
            except:
                pass
        
        return count
    
    def clear(self):
        """Clear all registered agents."""
        with self._lock:
            self._agents.clear()
    
    @classmethod
    def get_default(cls) -> "AgentRegistry":
        """Get the default registry instance."""
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance
    
    @classmethod
    def set_default(cls, registry: "AgentRegistry"):
        """Set the default registry instance."""
        cls._default_instance = registry


# Convenience functions
def register_agent(
    url: str,
    **metadata
) -> RegisteredAgent:
    """Register an agent with the default registry.
    
    Args:
        url: Agent URL
        **metadata: Additional metadata
        
    Returns:
        Registered agent
    """
    return AgentRegistry.get_default().register(url, **metadata)


def discover_agents(
    urls: Optional[List[str]] = None,
    env_var: Optional[str] = None,
    config_file: Optional[str] = None,
) -> List[RegisteredAgent]:
    """Discover agents from multiple sources.
    
    Args:
        urls: List of URLs to register
        env_var: Environment variable with URLs
        config_file: Path to config file
        
    Returns:
        List of discovered agents
    """
    registry = AgentRegistry.get_default()
    
    if urls:
        for url in urls:
            try:
                registry.register(url)
            except:
                pass
    
    if env_var:
        registry.discover_from_env(env_var)
    
    if config_file:
        registry.discover_from_file(config_file)
    
    return registry.list()
