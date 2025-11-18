"""
Agents module for the Bible Commentary Bot.

This module contains:
- Agent creation and configuration
- Agent tools (RAG tools for query processing)
- LLM interface (unified interface for OpenAI and Ollama)
- Embeddings (vector embeddings for semantic search)
- Memory management (long-term memory and checkpointing)
- Prompts (system prompts for agent behavior)
"""

from src.agents.agent import create_rag_agent
from src.agents.tools import get_rag_tools, set_llm
from src.agents.llm import get_llm, get_llm_from_config
from src.agents.embeddings import get_embedding_function, get_chunking_embedding_function
from src.agents.memory import (
    create_memory_backend,
    create_checkpointer,
    MemoryConfig,
    get_memory_config
)

__all__ = [
    "create_rag_agent",
    "get_rag_tools",
    "set_llm",
    "get_llm",
    "get_llm_from_config",
    "get_embedding_function",
    "get_chunking_embedding_function",
    "create_memory_backend",
    "create_checkpointer",
    "MemoryConfig",
    "get_memory_config",
]
