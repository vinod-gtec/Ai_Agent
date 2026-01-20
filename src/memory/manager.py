from typing import List, Dict, Optional
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory


class MemoryManager:
    """
    Unified memory management combining short and long-term memory.
    
    This class provides a single interface to both types of memory,
    abstracting away the complexity of managing two separate systems.
    
    Short-term: Recent conversation context
    Long-term: Persistent semantic memory
    
    Attributes:
        short_term: ShortTermMemory instance for conversation history.
        long_term: LongTermMemory instance for persistent storage.
    
    Usage:
        >>> memory = MemoryManager()
        >>> memory.add_to_short_term("user", "Hello")
        >>> memory.add_to_long_term("fact_1", "The sky is blue")
        >>> relevant = memory.retrieve_relevant("sky color")
    """
    
    def __init__(self):
        """
        Initialize both short-term and long-term memory systems.
        
        Creates instances of both memory types and prepares them for use.
        
        Note:
            Long-term memory may take a few seconds on first run to
            download the embedding model.
        """
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()
    
    def add_to_short_term(self, role: str, content: str) -> None:
        """
        Add a message to short-term conversation history.
        
        Args:
            role: Speaker role (user, assistant, agent name).
            content: Message content.
        
        Example:
            >>> memory.add_to_short_term("user", "What's 2+2?")
            >>> memory.add_to_short_term("assistant", "4")
        """
        self.short_term.add_message(role, content)
    
    def add_to_long_term(
        self,
        key: str,
        value: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Store information in long-term memory.
        
        Args:
            key: Unique identifier for this memory.
            value: The content to store.
            metadata: Optional metadata dictionary.
        
        Example:
            >>> memory.add_to_long_term(
            ...     "project_goal",
            ...     "Build a multi-agent AI system",
            ...     {"priority": "high"}
            ... )
        """
        self.long_term.store(key, value, metadata)
    
    def retrieve_relevant(self, query: str, k: int = 3) -> List[str]:
        """
        Search long-term memory for relevant information.
        
        Args:
            query: Search query.
            k: Number of results to return.
        
        Returns:
            List of relevant text snippets.
        
        Example:
            >>> results = memory.retrieve_relevant("project goals", k=5)
        """
        return self.long_term.retrieve(query, k)
    
    def get_short_term_history(self) -> List[Dict]:
        """
        Get recent conversation history.
        
        Returns:
            List of message dictionaries.
        
        Example:
            >>> history = memory.get_short_term_history()
            >>> print(f"Last message: {history[-1]['content']}")
        """
        return self.short_term.get_history()