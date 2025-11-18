"""
Configuration Management for Advanced RAG System
Centralized configuration for LLM selection, chunking parameters, and retrieval settings.
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv
_ = load_dotenv()
class RAGConfig:
    """Configuration class for the RAG system."""
    
    # LLM Configuration
    DEFAULT_LLM_PROVIDER = "ollama"  # "ollama" or "openai"
    DEFAULT_OLLAMA_MODEL = "deepseek-r1"
    DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
    DEFAULT_TEMPERATURE = 0.7
    
    # Smart Chunking Configuration
    CHUNKING_SIMILARITY_THRESHOLD = 0.75  # 0.75-0.85 recommended
    CHUNKING_MIN_SIZE = 400  # Minimum chunk size in characters
    CHUNKING_MAX_SIZE = 1200  # Maximum chunk size in characters
    CHUNKING_INITIAL_SEGMENT_SIZE = 300  # Initial segment size before merging
    
    # Retrieval Configuration
    DEFAULT_RETRIEVAL_K = 7  # Number of documents to retrieve
    AGENT_MAX_ITERATIONS = 10  # Maximum iterations for agent
    AGENT_VERBOSE = True  # Whether to show agent reasoning
    
    # Embedding Configuration
    CHUNKING_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers model
    
    # Memory Configuration
    ENABLE_MEMORY = True  # Enable long-term memory across conversations
    ENABLE_CHECKPOINTING = False  # Enable conversation checkpointing
    CHECKPOINT_DB_PATH = "checkpoints.db"  # Path to checkpoint database
    
    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """
        Get LLM configuration from environment variables or defaults.
        
        Environment variables:
        - LLM_PROVIDER: "ollama" or "openai"
        - LLM_MODEL: Specific model name
        - LLM_TEMPERATURE: Temperature for generation
        - OPENAI_API_KEY: OpenAI API key (if using OpenAI)
        
        Returns:
            Dictionary with LLM configuration
        """
        provider = os.getenv("LLM_PROVIDER", cls.DEFAULT_LLM_PROVIDER).lower()
        
        # Determine default model based on provider
        if provider == "openai":
            default_model = cls.DEFAULT_OPENAI_MODEL
        else:
            default_model = cls.DEFAULT_OLLAMA_MODEL
        
        model = os.getenv("LLM_MODEL", default_model)
        temperature = float(os.getenv("LLM_TEMPERATURE", cls.DEFAULT_TEMPERATURE))
        
        config = {
            "provider": provider,
            "model": model,
            "temperature": temperature
        }
        
        # Add API key if using OpenAI
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                config["api_key"] = api_key
        
        return config
    
    @classmethod
    def get_chunking_config(cls) -> Dict[str, Any]:
        """
        Get smart chunking configuration.
        
        Environment variables:
        - CHUNKING_SIMILARITY_THRESHOLD: Cosine similarity threshold
        - CHUNKING_MIN_SIZE: Minimum chunk size
        - CHUNKING_MAX_SIZE: Maximum chunk size
        - CHUNKING_INITIAL_SEGMENT_SIZE: Initial segment size
        
        Returns:
            Dictionary with chunking configuration
        """
        return {
            "similarity_threshold": float(
                os.getenv("CHUNKING_SIMILARITY_THRESHOLD", cls.CHUNKING_SIMILARITY_THRESHOLD)
            ),
            "min_chunk_size": int(
                os.getenv("CHUNKING_MIN_SIZE", cls.CHUNKING_MIN_SIZE)
            ),
            "max_chunk_size": int(
                os.getenv("CHUNKING_MAX_SIZE", cls.CHUNKING_MAX_SIZE)
            ),
            "initial_segment_size": int(
                os.getenv("CHUNKING_INITIAL_SEGMENT_SIZE", cls.CHUNKING_INITIAL_SEGMENT_SIZE)
            )
        }
    
    @classmethod
    def get_retrieval_config(cls) -> Dict[str, Any]:
        """
        Get retrieval configuration.
        
        Environment variables:
        - RETRIEVAL_K: Number of documents to retrieve
        - AGENT_MAX_ITERATIONS: Max agent iterations
        - AGENT_VERBOSE: Show agent reasoning (true/false)
        
        Returns:
            Dictionary with retrieval configuration
        """
        return {
            "k": int(os.getenv("RETRIEVAL_K", cls.DEFAULT_RETRIEVAL_K)),
            "max_iterations": int(
                os.getenv("AGENT_MAX_ITERATIONS", cls.AGENT_MAX_ITERATIONS)
            ),
            "verbose": os.getenv("AGENT_VERBOSE", str(cls.AGENT_VERBOSE)).lower() == "true"
        }
    
    @classmethod
    def get_memory_config(cls) -> Dict[str, Any]:
        """
        Get memory configuration.
        
        Environment variables:
        - ENABLE_MEMORY: Enable long-term memory (true/false)
        - ENABLE_CHECKPOINTING: Enable conversation checkpointing (true/false)
        - CHECKPOINT_DB_PATH: Path to checkpoint database
        
        Returns:
            Dictionary with memory configuration
        """
        return {
            "enable_memory": os.getenv("ENABLE_MEMORY", str(cls.ENABLE_MEMORY)).lower() == "true",
            "enable_checkpointing": os.getenv("ENABLE_CHECKPOINTING", str(cls.ENABLE_CHECKPOINTING)).lower() == "true",
            "checkpoint_db_path": os.getenv("CHECKPOINT_DB_PATH", cls.CHECKPOINT_DB_PATH)
        }
    
    @classmethod
    def get_all_config(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get all configuration settings.
        
        Returns:
            Dictionary with all configuration sections
        """
        return {
            "llm": cls.get_llm_config(),
            "chunking": cls.get_chunking_config(),
            "retrieval": cls.get_retrieval_config(),
            "memory": cls.get_memory_config()
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        config = cls.get_all_config()
        
        print("=" * 60)
        print("RAG SYSTEM CONFIGURATION")
        print("=" * 60)
        
        print("\n[LLM Configuration]")
        for key, value in config["llm"].items():
            if key != "api_key":  # Don't print API key
                print(f"  {key}: {value}")
        
        print("\n[Smart Chunking Configuration]")
        for key, value in config["chunking"].items():
            print(f"  {key}: {value}")
        
        print("\n[Retrieval Configuration]")
        for key, value in config["retrieval"].items():
            print(f"  {key}: {value}")
        
        print("\n[Memory Configuration]")
        for key, value in config["memory"].items():
            print(f"  {key}: {value}")
        
        print("=" * 60)


# Convenience function to get configuration
def get_config() -> Dict[str, Dict[str, Any]]:
    """Get all RAG configuration."""
    return RAGConfig.get_all_config()


if __name__ == "__main__":
    # Print current configuration
    RAGConfig.print_config()

