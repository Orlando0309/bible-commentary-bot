from src.agents.tools import get_rag_tools, set_llm
from src.agents.prompt import DEEP_AGENT_SYSTEM_PROMPT
from src.agents.memory import create_memory_backend, create_checkpointer, MemoryConfig
from deepagents import create_deep_agent
import warnings

def create_rag_agent(llm, enable_memory: bool = True, enable_checkpointing: bool = False):
    """
    Create a RAG agent using DeepAgents framework with optional memory support.
    
    Args:
        llm: The language model to use
        enable_memory: Whether to enable long-term memory (default: True)
        enable_checkpointing: Whether to enable conversation checkpointing (default: False)
    
    Returns:
        DeepAgent instance
    
    Memory Features:
        - When enabled, agent can save information to /memories/ path
        - Memories persist across conversations
        - Useful for remembering user preferences, previous discussions, etc.
    
    Note:
        Checkpointing requires the 'langgraph-checkpoint-sqlite' package.
        Install with: pip install langgraph-checkpoint-sqlite
        Or: pip install bible-commentary-bot[checkpointing]
    """
    # Set the LLM for tools
    set_llm(llm)
    
    # Get all RAG tools
    tools = get_rag_tools()
    
    # Enhance system prompt with memory instructions if enabled
    system_prompt = DEEP_AGENT_SYSTEM_PROMPT
    if enable_memory:
        system_prompt = DEEP_AGENT_SYSTEM_PROMPT + "\n" + MemoryConfig.get_memory_instructions()
    
    # Configure agent creation parameters
    agent_params = {
        "tools": tools,
        "system_prompt": system_prompt,
        "model": llm
    }
    
    # Add memory backend if enabled
    if enable_memory:
        agent_params["backend"] = create_memory_backend
    
    # Add checkpointing if enabled
    if enable_checkpointing:
        try:
            agent_params["checkpointer"] = create_checkpointer()
        except ImportError as e:
            warnings.warn(
                f"Checkpointing requested but not available: {e}\n"
                "Install with: pip install langgraph-checkpoint-sqlite\n"
                "Or: pip install bible-commentary-bot[checkpointing]\n"
                "Continuing without checkpointing...",
                UserWarning
            )
            # Continue without checkpointing
    
    # Create deep agent with configured parameters
    agent = create_deep_agent(**agent_params)
    
    return agent