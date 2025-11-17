"""
Advanced RAG System with DeepAgents
Supports query rephrasing, subqueries, and intelligent retrieval.
"""
import argparse
from langchain_core.prompts import ChatPromptTemplate
from deepagents import create_deep_agent
from llm import get_llm
from agent_tools import get_rag_tools, set_llm
from config import RAGConfig
from ingest import ingest_documents


# System prompt for DeepAgent
DEEP_AGENT_SYSTEM_PROMPT = """You are an expert Bible commentary researcher. Your job is to conduct thorough research on biblical questions and provide well-researched, accurate answers.

You have access to the following specialized tools:

## `rephrase_query`
Use this to reformulate a user's query to make it more effective for retrieval. Helpful when the original query is ambiguous or could be expressed more clearly.

## `generate_subqueries`
Use this to break down complex, multi-part questions into 2-4 simpler, focused subqueries. This is essential for handling questions that ask about multiple aspects or concepts.

## `vector_search`
Your primary research tool. Use this to search the Bible commentary database for relevant passages. You can specify the number of results to retrieve (default is 5).

## `synthesize_context`
Use this to combine and synthesize information from multiple search results, especially when you've conducted multiple searches via subqueries.

## Your Approach:
1. For complex questions, break them down using `generate_subqueries`
2. If a query is unclear, use `rephrase_query` to optimize it
3. Conduct thorough research using `vector_search`
4. Synthesize findings from multiple searches when needed
5. Base all answers ONLY on retrieved context - never make up information
6. Cite sources when available

Be methodical, thorough, and always ground your answers in the retrieved biblical commentary.
Always mention Sources in last part of your answer.
"""


# Simple prompt for direct answer generation (fallback)
SIMPLE_PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def create_rag_agent(llm):
    """
    Create a RAG agent using DeepAgents framework.
    
    Args:
        llm: The language model to use
    
    Returns:
        DeepAgent instance
    """
    # Set the LLM for tools
    set_llm(llm)
    
    # Get all RAG tools
    tools = get_rag_tools()
    
    # Create deep agent with tools and system prompt
    agent = create_deep_agent(
        tools=tools,
        system_prompt=DEEP_AGENT_SYSTEM_PROMPT,
        model=llm
    )
    
    return agent


def query_with_agent(
    query_text: str,
    provider: str = None,
    model: str = None,
    use_agent: bool = True
):
    """
    Query the Bible commentary database using advanced RAG with agents.
    
    Args:
        query_text: The user's question
        provider: LLM provider ("openai" or "ollama") - uses config default if None
        model: Specific model name (optional) - uses config default if None
        use_agent: Whether to use agent workflow (True) or simple retrieval (False)
    
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
    
    # Get the LLM
    llm = get_llm(provider=provider, model=model, temperature=temperature)
    
    if use_agent:
        # Use DeepAgent-based approach
        agent = create_rag_agent(llm)
        
        try:
            # DeepAgent expects messages format
            result = agent.invoke({
                "messages": [
                    {"role": "user", "content": query_text}
                ]
            })
            
            # Extract the response from the last message
            response_text = result["messages"][-1].content
            
            formatted_response = f"\nResponse: {response_text}\n\nThis response was generated using DeepAgents workflow with planning and tool orchestration."
            print(formatted_response)
            return response_text
        
        except Exception as e:
            print(f"DeepAgent execution failed: {e}")
            print("Falling back to simple retrieval...")
            use_agent = False
    
    if not use_agent:
        # Fallback to simple retrieval (original approach)
        from embeddings import get_embedding_function
        from secret import CHROMA_PATH
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
    
    args = parser.parse_args()
    
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
            use_agent=True  # Always use agent-based workflow
        )


if __name__ == "__main__":
    main()