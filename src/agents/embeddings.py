from langchain_ollama import OllamaEmbeddings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
_ = load_dotenv()

def get_embedding_function():
    """Get the main embedding function for vector store."""
    embedings = OllamaEmbeddings(
        model="qwen3-embedding:4b",
        base_url=os.getenv("OLLAMA_BASE_URL")
    )
    return embedings


def get_chunking_embedding_function():
    """
    Get a lightweight embedding function for smart chunking.
    Uses sentence-transformers for fast cosine similarity calculations.
    
    Returns:
        SentenceTransformer model
    """
    # Using a lightweight and fast model for chunking
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

