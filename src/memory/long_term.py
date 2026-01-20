from typing import List, Dict, Optional
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os


class LongTermMemory:
    """
    Long-term memory using ChromaDB vector database with local embeddings.
    
    This class provides semantic search capabilities over stored information.
    Unlike short-term memory, this persists to disk and can retrieve
    relevant information based on semantic similarity.
    
    Key Features:
        - Persistent storage (survives application restarts)
        - Semantic search using embeddings
        - FREE - uses local HuggingFace embeddings
        - No API calls or cloud dependencies
    
    Attributes:
        embeddings: HuggingFace embedding model for text vectorization.
        vector_store: ChromaDB instance for storing and searching vectors.
    
    Usage:
        >>> ltm = LongTermMemory()
        >>> ltm.store("meeting_1", "Discussed Q4 sales strategy")
        >>> relevant = ltm.retrieve("sales planning", k=3)
    """
    
    def __init__(self):
        """
        Initialize long-term memory with local embeddings and ChromaDB.
        
        Uses HuggingFace's sentence-transformers model for embeddings.
        This model is downloaded once and cached locally - completely free.
        
        Note:
            First run will download the embedding model (~90MB).
            Subsequent runs use the cached model.
        """
        # Use FREE local embeddings from HuggingFace
        # Model: all-MiniLM-L6-v2 (fast, lightweight, good quality)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv(
                "EMBEDDINGS_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        )
        
        # Use local ChromaDB for persistence
        # No cloud connection needed - all data stays on your machine
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
        
        self.vector_store = Chroma(
            collection_name="agent_memory",
            embedding_function=self.embeddings,
            persist_directory=persist_dir
        )
    
    def store(self, key: str, value: str, metadata: Optional[Dict] = None) -> None:
        """
        Store a piece of information in long-term memory.
        
        The text is automatically converted to embeddings and stored
        in the vector database for later semantic retrieval.
        
        Args:
            key: Unique identifier for this piece of information.
            value: The text content to store.
            metadata: Optional metadata to attach (e.g., timestamp, source).
        
        Example:
            >>> ltm.store(
            ...     key="sales_report_q4",
            ...     value="Q4 sales were strong with 15% growth",
            ...     metadata={"date": "2024-12-31", "type": "report"}
            ... )
        
        Note:
            The data is persisted to disk immediately.
        """
        self.vector_store.add_texts(
            texts=[value],
            metadatas=[metadata or {}],
            ids=[key]
        )
        # Persist changes to disk
        self.vector_store.persist()
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve the most semantically similar memories.
        
        Uses cosine similarity between query embedding and stored embeddings
        to find the most relevant information.
        
        Args:
            query: The search query (natural language).
            k: Number of results to return.
        
        Returns:
            List of text snippets most similar to the query.
        
        Example:
            >>> results = ltm.retrieve("sales performance", k=3)
            >>> for result in results:
            ...     print(result)
        
        Note:
            Results are ordered by relevance (most similar first).
        """
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    def delete(self, key: str) -> None:
        """
        Delete a specific memory by its key.
        
        Args:
            key: The unique identifier of the memory to delete.
        
        Example:
            >>> ltm.delete("old_report_q1")
        
        Note:
            This permanently removes the data from the vector store.
        """
        self.vector_store.delete(ids=[key])
        self.vector_store.persist()
