"""
Memory - Context and conversation history management for agents
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque


@dataclass
class Message:
    """A single message in the conversation"""
    
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class Memory(ABC):
    """
    Abstract base class for agent memory systems.
    
    Memory provides context persistence across agent interactions,
    enabling stateful conversations and long-term knowledge retention.
    """
    
    @abstractmethod
    def add_message(self, role: str, content: str, **kwargs) -> None:
        """Add a message to memory"""
        pass
    
    @abstractmethod
    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Retrieve messages from memory"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all messages from memory"""
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[Message]:
        """Search memory for relevant messages"""
        pass


class ConversationMemory(Memory):
    """
    Simple conversation memory that maintains message history.
    
    Stores recent messages in a sliding window, suitable for
    short to medium conversation contexts.
    
    Example:
        >>> memory = ConversationMemory(max_messages=100)
        >>> memory.add_message("user", "Hello!")
        >>> memory.add_message("assistant", "Hi there!")
        >>> messages = memory.get_messages()
    """
    
    def __init__(
        self,
        max_messages: int = 100,
        include_system: bool = True,
    ):
        self.max_messages = max_messages
        self.include_system = include_system
        self._messages: deque = deque(maxlen=max_messages)
        self._system_message: Optional[Message] = None
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Add a message to the conversation history"""
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        
        if role == "system":
            self._system_message = message
        else:
            self._messages.append(message)
    
    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Get conversation messages in OpenAI format"""
        messages = []
        
        # Add system message first if exists
        if self.include_system and self._system_message:
            messages.append({
                "role": "system",
                "content": self._system_message.content,
            })
        
        # Get recent messages
        recent = list(self._messages)
        if limit:
            recent = recent[-limit:]
        
        for msg in recent:
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })
        
        return messages
    
    def clear(self) -> None:
        """Clear conversation history"""
        self._messages.clear()
        self._system_message = None
    
    def search(self, query: str, limit: int = 5) -> List[Message]:
        """Simple keyword search in message history"""
        query_lower = query.lower()
        matches = []
        
        for msg in self._messages:
            if query_lower in msg.content.lower():
                matches.append(msg)
                if len(matches) >= limit:
                    break
        
        return matches
    
    @property
    def message_count(self) -> int:
        """Number of messages in memory"""
        return len(self._messages)
    
    def __len__(self) -> int:
        return self.message_count


class BufferMemory(Memory):
    """
    Buffer memory with summarization support.
    
    When the buffer exceeds a threshold, older messages are
    summarized to maintain context while reducing token usage.
    """
    
    def __init__(
        self,
        max_tokens: int = 4000,
        summary_threshold: int = 3000,
    ):
        self.max_tokens = max_tokens
        self.summary_threshold = summary_threshold
        self._messages: List[Message] = []
        self._summary: Optional[str] = None
    
    def add_message(self, role: str, content: str, **kwargs) -> None:
        """Add message and trigger summarization if needed"""
        self._messages.append(Message(role=role, content=content))
        # Could trigger summarization here based on token count
    
    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Get messages, prepending summary if exists"""
        messages = []
        
        if self._summary:
            messages.append({
                "role": "system",
                "content": f"Previous conversation summary: {self._summary}",
            })
        
        recent = self._messages[-limit:] if limit else self._messages
        for msg in recent:
            messages.append({"role": msg.role, "content": msg.content})
        
        return messages
    
    def clear(self) -> None:
        self._messages.clear()
        self._summary = None
    
    def search(self, query: str, limit: int = 5) -> List[Message]:
        query_lower = query.lower()
        return [
            msg for msg in self._messages
            if query_lower in msg.content.lower()
        ][:limit]


class VectorMemory(Memory):
    """
    Vector-based memory for semantic search and retrieval.
    
    Uses embeddings to store and retrieve relevant context
    based on semantic similarity rather than keyword matching.
    
    Requires an embedding provider to be configured.
    """
    
    def __init__(
        self,
        embedding_provider: Optional[Any] = None,
        similarity_threshold: float = 0.7,
    ):
        self.embedding_provider = embedding_provider
        self.similarity_threshold = similarity_threshold
        self._messages: List[Message] = []
        self._embeddings: List[List[float]] = []
    
    def add_message(self, role: str, content: str, **kwargs) -> None:
        """Add message with embedding"""
        message = Message(role=role, content=content)
        self._messages.append(message)
        
        # Generate embedding if provider available
        if self.embedding_provider:
            embedding = self.embedding_provider.embed(content)
            self._embeddings.append(embedding)
    
    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        recent = self._messages[-limit:] if limit else self._messages
        return [{"role": m.role, "content": m.content} for m in recent]
    
    def clear(self) -> None:
        self._messages.clear()
        self._embeddings.clear()
    
    def search(self, query: str, limit: int = 5) -> List[Message]:
        """Semantic search using embeddings"""
        if not self.embedding_provider or not self._embeddings:
            # Fallback to keyword search
            return [
                m for m in self._messages
                if query.lower() in m.content.lower()
            ][:limit]
        
        # Generate query embedding
        query_embedding = self.embedding_provider.embed(query)
        
        # Calculate similarities
        similarities = []
        for i, emb in enumerate(self._embeddings):
            sim = self._cosine_similarity(query_embedding, emb)
            if sim >= self.similarity_threshold:
                similarities.append((sim, self._messages[i]))
        
        # Sort by similarity and return top results
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [msg for _, msg in similarities[:limit]]
    
    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0
