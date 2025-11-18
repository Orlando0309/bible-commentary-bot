"""
Advanced RAG System with DeepAgents
Supports query rephrasing, subqueries, and intelligent retrieval.
"""
import argparse
from langchain_core.prompts import ChatPromptTemplate
from src.agents.agent import create_rag_agent
from src.agents.llm import get_llm
from src.agents.tools import get_rag_tools, set_llm
from src.agents.prompt import DEEP_AGENT_SYSTEM_PROMPT
from src.config import RAGConfig
from src.ingest.ingest import ingest_documents


# Simple prompt for direct answer generation (fallback)
SIMPLE_PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""





def query_with_agent(
    query_text: str,
    provider: str = None,
    model: str = None,
    use_agent: bool = True,
    enable_memory: bool = None,
    enable_checkpointing: bool = None
):
    """
    Query the Bible commentary database using advanced RAG with agents.
    
    Args:
        query_text: The user's question
        provider: LLM provider ("openai" or "ollama") - uses config default if None
        model: Specific model name (optional) - uses config default if None
        use_agent: Whether to use agent workflow (True) or simple retrieval (False)
        enable_memory: Enable long-term memory (uses config default if None)
        enable_checkpointing: Enable conversation checkpointing (uses config default if None)
    
    Returns:
        Response text and sources
    """
    # Get LLM config
    llm_config = RAGConfig.get_llm_config()
    if provider is None:
        provider = llm_config.get("provider", "ollama")
    if model is None:
        model = llm_config.get("model")
    
    temperature = llm_config.get("temperature", 0.7)
    
    # Get memory config
    memory_config = RAGConfig.get_memory_config()
    if enable_memory is None:
        enable_memory = memory_config.get("enable_memory", True)
    if enable_checkpointing is None:
        enable_checkpointing = memory_config.get("enable_checkpointing", False)
    
    # Get the LLM
    llm = get_llm(provider=provider, model=model, temperature=temperature)
    
    if use_agent:
        # Use DeepAgent-based approach with memory
        agent = create_rag_agent(
            llm,
            enable_memory=enable_memory,
            enable_checkpointing=enable_checkpointing
        )
        
        try:
            # DeepAgent expects messages format
            result = agent.invoke({
                "messages": [
                    {"role": "user", "content": query_text}
                ]
            })
            
            # Extract the response from the last message
            response_text = result["messages"][-1].content
            
            memory_status = " with memory" if enable_memory else ""
            formatted_response = f"\nResponse: {response_text}\n\nThis response was generated using DeepAgents workflow{memory_status} with planning and tool orchestration."
            print(formatted_response)
            return response_text
        
        except Exception as e:
            print(f"DeepAgent execution failed: {e}")
            print("Falling back to simple retrieval...")
            use_agent = False
    
    if not use_agent:
        # Fallback to simple retrieval (original approach)
        from src.agents.embeddings import get_embedding_function
        from src.config.secret import CHROMA_PATH
        from langchain_chroma import Chroma
        
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Get retrieval config
        retrieval_config = RAGConfig.get_retrieval_config()
        k = retrieval_config.get("k", 7)
        
        results = db.similarity_search_with_score(query_text, k=k)
        
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(SIMPLE_PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        response_text = llm.invoke(prompt)
        
        # Handle different LLM response types
        if hasattr(response_text, 'content'):
            response_text = response_text.content
        
        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        print(formatted_response)
        return response_text


# Legacy function for backward compatibility
def query_tag(query_text: str, k: int = 7):
    """Legacy function - use query_with_agent instead."""
    return query_with_agent(query_text, provider="ollama", model="deepseek-r1", use_agent=False)


def main():
    """Main function with command-line argument support."""
    parser = argparse.ArgumentParser(
        description="Advanced RAG System for Bible Commentary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a single PDF file
  python main.py --ingest "document.pdf"
  
  # Ingest all PDFs from a directory
  python main.py --ingest "data/"
  
  # Query the database
  python main.py --query "What are the miracles of Jesus?"
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--ingest",
        type=str,
        metavar="PATH",
        help="Path to a PDF file or directory containing PDF files to ingest"
    )
    mode_group.add_argument(
        "--query",
        type=str,
        metavar="QUERY",
        help="Query the Bible commentary database"
    )
    
    # Memory options
    parser.add_argument(
        "--memory",
        action="store_true",
        default=None,
        help="Enable long-term memory (agent can remember across sessions)"
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable long-term memory"
    )
    parser.add_argument(
        "--checkpointing",
        action="store_true",
        help="Enable conversation checkpointing"
    )
    
    args = parser.parse_args()
    
    # Determine memory settings
    enable_memory = None
    if args.memory:
        enable_memory = True
    elif hasattr(args, 'no_memory') and args.no_memory:
        enable_memory = False
    
    enable_checkpointing = args.checkpointing if hasattr(args, 'checkpointing') else None
    
    # Handle ingestion mode
    if args.ingest:
        ingest_documents(args.ingest)
    
    # Handle query mode
    elif args.query:
        # Print config for query mode
        RAGConfig.print_config()
        
        print("\n" + "=" * 80)
        print("ADVANCED RAG WITH DEEPAGENTS")
        print("=" * 80)
        print(f"\nQuery: {args.query}\n")
        
        # Query with default parameters from config
        response = query_with_agent(
            args.query,
            provider=None,  # Uses config default
            model=None,  # Uses config default
            use_agent=True,  # Always use agent-based workflow
            enable_memory=enable_memory,  # Uses config default if None
            enable_checkpointing=enable_checkpointing  # Uses config default if None
        )


if __name__ == "__main__":
    main()