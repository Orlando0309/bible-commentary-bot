"""
Memory Backend Configuration for DeepAgents
Implements long-term memory for Bible commentary agent using LangGraph's Store.
"""
from typing import Optional
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore
import os

# Try to import SqliteSaver - it's in a separate package
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    SQLITE_CHECKPOINT_AVAILABLE = True
except ImportError:
    SQLITE_CHECKPOINT_AVAILABLE = False
    SqliteSaver = None


def create_memory_backend(runtime, persist: bool = True, db_path: Optional[str] = None):
    """
    Create a memory backend for DeepAgents with long-term memory support.
    
    NOTE: Currently, memory persistence via StoreBackend may not work with all
    DeepAgents versions. This function will fall back to StateBackend if
    StoreBackend initialization fails.
    
    This configures a hybrid storage system where:
    - `/memories/` path is routed to persistent storage (survives across sessions)
    - Other paths use ephemeral storage (cleared after each session)
    
    Args:
        runtime: The runtime context for StateBackend
        persist: Whether to use persistent memory (default: True)
        db_path: Path to SQLite database for persistent storage (optional)
    
    Returns:
        CompositeBackend or StateBackend configured with long-term memory
    
    Example:
        ```python
        agent = create_deep_agent(
            backend=lambda runtime: create_memory_backend(runtime, persist=True)
        )
        ```
    
    Memory Usage in Prompts:
        The agent can save information to memory for later retrieval:
        - Save: Write to `/memories/conversation_context.txt`
        - Retrieve: Read from `/memories/conversation_context.txt`
        - List memories: List files in `/memories/`
    """
    if persist:
        try:
            # Try to create persistent memory storage
            memory_store = InMemoryStore()
            
            # Try different StoreBackend initialization methods
            try:
                # Method 1: Direct initialization
                store_backend = StoreBackend(store=memory_store)
            except (TypeError, AttributeError):
                try:
                    # Method 2: Without store parameter
                    store_backend = StoreBackend(memory_store)
                except (TypeError, AttributeError):
                    # If both fail, fall back to StateBackend
                    import warnings
                    warnings.warn(
                        "StoreBackend initialization failed. Using StateBackend only. "
                        "Memory will be ephemeral (cleared after session).",
                        UserWarning
                    )
                    return StateBackend(runtime)
            
            # Create CompositeBackend with memory routing
            return CompositeBackend(
                default=StateBackend(runtime),  # Ephemeral storage for general files
                routes={
                    "/memories/": store_backend  # Persistent storage for memories
                }
            )
            
        except Exception as e:
            # If anything fails, fall back to StateBackend
            import warnings
            warnings.warn(
                f"Memory backend initialization failed: {e}. "
                "Using StateBackend only. Memory will be ephemeral.",
                UserWarning
            )
            return StateBackend(runtime)
    else:
        # Use only ephemeral storage (no persistence)
        return StateBackend(runtime)


def create_checkpointer(db_path: Optional[str] = None):
    """
    Create a checkpointer for conversation state persistence.
    
    This allows the agent to resume conversations and maintain state
    across different sessions.
    
    Args:
        db_path: Path to SQLite database (default: "checkpoints.db")
    
    Returns:
        SqliteSaver instance for checkpointing, or None if not available
    
    Raises:
        ImportError: If langgraph-checkpoint-sqlite is not installed
    
    Example:
        ```python
        agent = create_deep_agent(
            checkpointer=create_checkpointer("checkpoints.db")
        )
        ```
    
    Note:
        Requires the `langgraph-checkpoint-sqlite` package to be installed:
        ```bash
        pip install langgraph-checkpoint-sqlite
        ```
    """
    if not SQLITE_CHECKPOINT_AVAILABLE:
        raise ImportError(
            "SQLite checkpointing requires the 'langgraph-checkpoint-sqlite' package. "
            "Install it with: pip install langgraph-checkpoint-sqlite"
        )
    
    if db_path is None:
        db_path = os.path.join(os.getcwd(), "checkpoints.db")
    
    return SqliteSaver.from_conn_string(db_path)


class MemoryConfig:
    """Configuration for agent memory settings."""
    
    # Memory paths
    MEMORY_ROOT = "/memories/"
    CONVERSATION_MEMORY = "/memories/conversation_context.txt"
    USER_PREFERENCES = "/memories/user_preferences.json"
    QUERY_HISTORY = "/memories/query_history.txt"
    
    # Database paths
    DEFAULT_CHECKPOINT_DB = "checkpoints.db"
    
    @classmethod
    def get_memory_instructions(cls) -> str:
        """
        Get instructions for the agent on how to use memory.
        
        Returns:
            String with memory usage instructions
        """
        return f"""
## Memory Management

You have access to a persistent memory system that survives across conversations:

### Memory Files:
- `{cls.CONVERSATION_MEMORY}` - Store important context from conversations
- `{cls.USER_PREFERENCES}` - Remember user preferences and interests
- `{cls.QUERY_HISTORY}` - Track previously answered questions

### Memory Operations:
1. **Save to Memory**: Write important information to memory files
   - Use: `write_file("{cls.CONVERSATION_MEMORY}", "content")`
   
2. **Retrieve from Memory**: Read previous conversation context
   - Use: `read_file("{cls.CONVERSATION_MEMORY}")`
   
3. **Check Memory**: List available memories
   - Use: `list_directory("{cls.MEMORY_ROOT}")`

### When to Use Memory:
- User asks you to remember something specific
- Important theological discussions to reference later
- User preferences about Bible translations or topics
- Previously answered complex questions for quick reference

### Memory Best Practices:
- Keep memory files concise and well-organized
- Update memory when new important information is discussed
- Check memory at the start of conversations for context
- Summarize long conversations before storing
"""


# Convenience function
def get_memory_config() -> dict:
    """
    Get memory configuration as a dictionary.
    
    Returns:
        Dictionary with memory configuration
    """
    return {
        "persist": True,
        "checkpoint_db": MemoryConfig.DEFAULT_CHECKPOINT_DB,
        "memory_root": MemoryConfig.MEMORY_ROOT,
        "enable_checkpointing": True
    }

