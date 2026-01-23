import os
from typing import Any
from langchain_core.language_models import BaseChatModel


class FreeLLMFactory:
    """
    Factory for creating free LLM instances.
    
    This factory supports multiple FREE LLM providers:
    - Groq: Fast, free API (RECOMMENDED)
    - Ollama: 100% local, unlimited
    - Together AI: Free credits
    - HuggingFace: Completely free
    
    Usage:
        >>> llm = FreeLLMFactory.create_llm()
        >>> response = llm.invoke("Hello!")
    
    Configuration:
        Set LLM_PROVIDER in .env to choose provider:
        - "groq" (default, recommended)
        - "ollama" (for local/private)
        - "together" (for free credits)
        - "huggingface" (for completely free)
    """
    
    @staticmethod
    def create_llm() -> BaseChatModel:
        """
        Create an LLM instance based on environment configuration.
        
        Reads LLM_PROVIDER from environment and creates the appropriate
        LLM instance with proper configuration.
        
        Returns:
            Configured LLM instance ready to use.
        
        Raises:
            ValueError: If provider is unknown.
            ImportError: If required package is not installed.
        
        Environment Variables:
            LLM_PROVIDER: Which provider to use (groq/ollama/together/huggingface)
            LLM_MODEL: Model name to use
            GROQ_API_KEY: API key for Groq
            TOGETHER_API_KEY: API key for Together
            HUGGINGFACE_API_KEY: API key for HuggingFace
            OLLAMA_BASE_URL: URL for Ollama server
        
        Example:
            >>> os.environ["LLM_PROVIDER"] = "groq"
            >>> os.environ["GROQ_API_KEY"] = "gsk_..."
            >>> llm = FreeLLMFactory.create_llm()
        """
        provider = os.getenv("LLM_PROVIDER", "groq").lower()
        
        if provider == "groq":
            return FreeLLMFactory._create_groq()
        elif provider == "together":
            return FreeLLMFactory._create_together()
        elif provider == "huggingface":
            return FreeLLMFactory._create_huggingface()
        elif provider == "ollama":
            return FreeLLMFactory._create_ollama()
        else:
            raise ValueError(
                f"Unknown LLM provider: {provider}. "
                f"Must be one of: groq, together, huggingface, ollama"
            )
    
    @staticmethod
    def _create_groq():
        """
        Create Groq LLM instance.
        
        Groq offers:
        - Fast inference (0.5-1 second)
        - Llama 3.1 70B model
        - 14,400 free requests/day
        - Professional API
        
        Returns:
            Configured ChatGroq instance.
        
        Raises:
            ImportError: If langchain-groq not installed.
        
        Note:
            Get free API key from: https://console.groq.com
        """
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError(
                "Groq provider requires langchain-groq. "
                "Install with: pip install langchain-groq"
            )
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment. "
                "Get free key from: https://console.groq.com"
            )
        
        return ChatGroq(
            groq_api_key=api_key,
            model_name=os.getenv("LLM_MODEL", "llama-3.1-70b-versatile"),
            temperature=0
        )
    
    @staticmethod
    def _create_together():
        """
        Create Together AI LLM instance.
        
        Together AI offers:
        - $25 free credits on signup
        - Multiple model options
        - Good performance
        
        Returns:
            Configured ChatTogether instance.
        """
        try:
            from langchain_together import ChatTogether
        except ImportError:
            raise ImportError(
                "Together provider requires langchain-together. "
                "Install with: pip install langchain-together"
            )
        
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError(
                "TOGETHER_API_KEY not found. "
                "Get free credits from: https://api.together.xyz"
            )
        
        return ChatTogether(
            together_api_key=api_key,
            model=os.getenv(
                "LLM_MODEL",
                "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
            ),
            temperature=0
        )
    
    @staticmethod
    def _create_huggingface():
        """
        Create HuggingFace LLM instance.
        
        HuggingFace offers:
        - Completely free inference API
        - Multiple models
        - Rate limited but usable
        
        Returns:
            Configured ChatHuggingFace instance.
        """
        try:
            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        except ImportError:
            raise ImportError(
                "HuggingFace provider requires langchain-huggingface. "
                "Install with: pip install langchain-huggingface"
            )
        
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError(
                "HUGGINGFACE_API_KEY not found. "
                "Get free key from: https://huggingface.co/settings/tokens"
            )
        
        llm = HuggingFaceEndpoint(
            repo_id=os.getenv(
                "LLM_MODEL",
                "meta-llama/Meta-Llama-3-70B-Instruct"
            ),
            huggingfacehub_api_token=api_key,
            temperature=0
        )
        return ChatHuggingFace(llm=llm)
    
    @staticmethod
    def _create_ollama():
        """
        Create local Ollama LLM instance.
        
        Ollama offers:
        - 100% free, runs locally
        - Complete privacy
        - Unlimited requests
        - Works offline
        
        Returns:
            Configured ChatOllama instance.
        
        Note:
            Requires Ollama installed: https://ollama.ai
            Pull model first: ollama pull llama3.1:8b
        """
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "Ollama provider requires langchain-ollama. "
                "Install with: pip install langchain-ollama"
            )
        
        return ChatOllama(
            model=os.getenv("LLM_MODEL", "llama3.1:8b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0
        )
