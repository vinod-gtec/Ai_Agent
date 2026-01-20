from typing import List, Dict
from langchain.memory import ConversationBufferMemory


class ShortTermMemory:
    """
    Manages short-term conversation memory using LangChain's buffer.
    
    This class maintains a rolling window of recent messages for context
    in ongoing conversations. It's "short-term" because it's not persisted
    and has a size limit.
    
    Attributes:
        memory: LangChain ConversationBufferMemory instance.
        max_messages: Maximum number of messages to retain.
    
    Usage:
        >>> stm = ShortTermMemory(max_messages=50)
        >>> stm.add_message("user", "Hello")
        >>> stm.add_message("assistant", "Hi there!")
        >>> history = stm.get_history()
    """
    
    def __init__(self, max_messages: int = 50):
        """
        Initialize short-term memory with a message limit.
        
        Args:
            max_messages: Maximum number of messages to keep in memory.
                         Older messages are discarded when limit is reached.
        
        Note:
            The memory is transient and cleared when the application restarts.
        """
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        self.max_messages = max_messages
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a new message to the conversation history.
        
        Args:
            role: The speaker role (e.g., "user", "assistant", "planner").
            content: The message content.
        
        Example:
            >>> stm.add_message("user", "What's the weather?")
            >>> stm.add_message("assistant", "It's sunny today.")
        
        Note:
            Messages are automatically trimmed if max_messages is exceeded.
        """
        self.memory.chat_memory.add_message({
            "role": role,
            "content": content
        })
        
        # Trim oldest messages if we exceed the limit
        if len(self.memory.chat_memory.messages) > self.max_messages:
            self.memory.chat_memory.messages = \
                self.memory.chat_memory.messages[-self.max_messages:]
    
    def get_history(self) -> List[Dict]:
        """
        Retrieve the full conversation history.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys.
        
        Example:
            >>> history = stm.get_history()
            >>> print(history[-1]['content'])  # Last message
        """
        return self.memory.chat_memory.messages
    
    def clear(self) -> None:
        """
        Clear all messages from memory.
        
        Useful for starting fresh conversations or freeing memory.
        
        Example:
            >>> stm.clear()
            >>> assert len(stm.get_history()) == 0
        """
        self.memory.clear()