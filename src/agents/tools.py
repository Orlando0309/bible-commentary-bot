"""
Agent Tools for Advanced RAG with DeepAgents
Contains tools for query rephrasing, subquery generation, vector search, and synthesis.
"""
from typing import List, Dict, Any
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

from src.agents.embeddings import get_embedding_function
from src.config.secret import CHROMA_PATH


# Global variable to store the LLM (will be set when creating the agent)
_llm = None


def set_llm(llm):
    """Set the LLM instance to be used by tools."""
    global _llm
    _llm = llm


@tool
def rephrase_query(query: str) -> str:
    """
    Rephrase the user query to improve retrieval quality.
    Useful when the original query is ambiguous or could be expressed better.
    
    Args:
        query: The original user query
    
    Returns:
        A rephrased version of the query optimized for retrieval
    """
    if _llm is None:
        return query
    
    rephrase_prompt = PromptTemplate(
        input_variables=["query"],
        template="""You are a query optimization expert. Rephrase the following query to make it more precise and better suited for semantic search in a Bible commentary database.
        
Original Query: {query}

Rephrased Query (be concise and focused):"""
    )
    
    prompt = rephrase_prompt.format(query=query)
    rephrased = _llm.invoke(prompt)
    
    # Handle different LLM response types
    if hasattr(rephrased, 'content'):
        return rephrased.content.strip()
    return str(rephrased).strip()


@tool
def generate_subqueries(query: str) -> List[str]:
    """
    Break down a complex query into 2-4 simpler subqueries.
    Useful when the user asks multiple questions or needs multi-faceted information.
    
    Args:
        query: The complex user query
    
    Returns:
        A list of 2-4 simpler subqueries
    """
    if _llm is None:
        return [query]
    
    subquery_prompt = PromptTemplate(
        input_variables=["query"],
        template="""You are a query decomposition expert. Break down the following complex query into 2-4 simpler, focused subqueries that together answer the original question.

Original Query: {query}

Return the subqueries as a numbered list (1., 2., 3., etc.), one per line. Be concise.

Subqueries:"""
    )
    
    prompt = subquery_prompt.format(query=query)
    response = _llm.invoke(prompt)
    
    # Handle different LLM response types
    if hasattr(response, 'content'):
        response_text = response.content
    else:
        response_text = str(response)
    
    # Parse the numbered list
    subqueries = []
    for line in response_text.strip().split('\n'):
        line = line.strip()
        # Remove numbering (1., 2., etc.) and clean up
        if line and (line[0].isdigit() or line.startswith('-')):
            # Remove leading numbers, dots, dashes, etc.
            clean_query = line.lstrip('0123456789.-) ').strip()
            if clean_query:
                subqueries.append(clean_query)
    
    # If parsing failed, return the original query
    if not subqueries:
        subqueries = [query]
    
    return subqueries[:4]  # Limit to 4 subqueries


@tool
def vector_search(query: str, k: int = 5) -> str:
    """
    Search the Bible commentary vector database for relevant passages.
    
    Args:
        query: The search query
        k: Number of results to retrieve (default: 5)
    
    Returns:
        A formatted string containing the retrieved passages and their sources
    """
    try:
        # Initialize the vector database
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Perform similarity search
        results = db.similarity_search_with_score(query, k=k)
        
        if not results:
            return "No relevant passages found."
        
        # Format the results
        formatted_results = []
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("id", "Unknown")
            content = doc.page_content
            formatted_results.append(f"[Result {i}] (Score: {score:.3f}, Source: {source})\n{content}")
        
        return "\n\n---\n\n".join(formatted_results)
    
    except Exception as e:
        return f"Error during vector search: {str(e)}"


@tool
def synthesize_context(contexts: List[str], original_query: str) -> str:
    """
    Synthesize multiple retrieved contexts into a coherent summary.
    Useful when dealing with results from multiple subqueries.
    
    Args:
        contexts: List of context strings from different searches
        original_query: The original user query for context
    
    Returns:
        A synthesized summary of all contexts
    """
    if not contexts:
        return "No context available to synthesize."
    
    if len(contexts) == 1:
        return contexts[0]
    
    # If LLM is available, use it for smart synthesis
    if _llm is not None:
        synthesis_prompt = PromptTemplate(
            input_variables=["query", "contexts"],
            template="""You are a context synthesis expert. Combine the following retrieved passages into a coherent summary that addresses the user's query. Remove redundancy and highlight the most relevant information.

User Query: {query}

Retrieved Passages:
{contexts}

Synthesized Summary:"""
        )
        
        # Combine contexts with separators
        combined_contexts = "\n\n---\n\n".join([f"Passage {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = synthesis_prompt.format(query=original_query, contexts=combined_contexts)
        response = _llm.invoke(prompt)
        
        # Handle different LLM response types
        if hasattr(response, 'content'):
            return response.content.strip()
        return str(response).strip()
    
    # Fallback: just concatenate the contexts
    return "\n\n---\n\n".join(contexts)


# Helper function to get all tools
def get_rag_tools():
    """Get all RAG agent tools."""
    return [
        rephrase_query,
        generate_subqueries,
        vector_search,
        synthesize_context
    ]

