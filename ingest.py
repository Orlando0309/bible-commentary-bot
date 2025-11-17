"""
Data Ingestion Module with Smart Chunking
Handles loading PDFs and creating smart semantic chunks for the vector database.
"""
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from create_database import add_to_chroma
from embeddings import get_chunking_embedding_function
from config import RAGConfig


def load_documents(data_path: str):
    """
    Load PDF documents from a file or directory.
    
    Args:
        data_path: Path to a single PDF file or directory containing PDF files
    
    Returns:
        List of loaded documents
    """
    if os.path.isfile(data_path):
        # Single PDF file
        if not data_path.lower().endswith('.pdf'):
            raise ValueError(f"File must be a PDF: {data_path}")
        document_loader = PyPDFLoader(data_path)
        return document_loader.load()
    elif os.path.isdir(data_path):
        # Directory of PDFs
        document_loader = PyPDFDirectoryLoader(data_path)
        return document_loader.load()
    else:
        raise ValueError(f"Path does not exist: {data_path}")


def split_documents_basic(documents: list[Document], chunk: int):
    """
    Basic splitting function (kept for backward compatibility).
    
    Args:
        documents: List of documents to split
        chunk: Chunk size in characters
    
    Returns:
        List of Document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk,
        chunk_overlap=chunk // 10,
        is_separator_regex=False,
        length_function=len,
    )
    return text_splitter.split_documents(documents)


def smart_split_documents(
    documents: list[Document],
    similarity_threshold: float = 0.8,
    min_chunk_size: int = 400,
    max_chunk_size: int = 1200,
    initial_segment_size: int = 300
):
    """
    Smart document splitting using cosine similarity to merge semantically similar segments.
    
    Args:
        documents: List of documents to split
        similarity_threshold: Threshold above which consecutive segments are merged (0.75-0.85)
        min_chunk_size: Minimum size for a chunk (characters)
        max_chunk_size: Maximum size for a chunk (characters)
        initial_segment_size: Size of initial segments before merging (characters)
    
    Returns:
        List of Document chunks with smart semantic boundaries
    """
    # Get the embedding model for chunking
    embedding_model = get_chunking_embedding_function()
    
    all_chunks = []
    
    for doc in documents:
        text = doc.page_content
        
        # Skip very short documents
        if len(text) < min_chunk_size:
            all_chunks.append(doc)
            continue
        
        # Step 1: Create initial segments
        segments = []
        for i in range(0, len(text), initial_segment_size):
            segment = text[i:i + initial_segment_size]
            if segment.strip():  # Only add non-empty segments
                segments.append(segment)
        
        if len(segments) <= 1:
            all_chunks.append(doc)
            continue
        
        # Step 2: Generate embeddings for all segments
        try:
            embeddings = embedding_model.encode(segments, show_progress_bar=False)
        except Exception as e:
            print(f"Warning: Failed to generate embeddings for document. Using basic split. Error: {e}")
            # Fallback to basic splitting
            basic_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=80,
                length_function=len,
            )
            all_chunks.extend(basic_splitter.split_documents([doc]))
            continue
        
        # Step 3: Calculate cosine similarity and merge similar consecutive segments
        merged_chunks = []
        current_chunk = segments[0]
        current_embedding = embeddings[0]
        
        for i in range(1, len(segments)):
            # Calculate similarity between current chunk and next segment
            similarity = cosine_similarity(
                [current_embedding],
                [embeddings[i]]
            )[0][0]
            
            # Check if we should merge
            should_merge = (
                similarity > similarity_threshold and
                len(current_chunk) + len(segments[i]) <= max_chunk_size
            )
            
            if should_merge:
                # Merge the segment into current chunk
                current_chunk += " " + segments[i]
                # Update the embedding to be the average of merged segments
                current_embedding = np.mean([current_embedding, embeddings[i]], axis=0)
            else:
                # Save current chunk if it meets minimum size
                if len(current_chunk) >= min_chunk_size:
                    merged_chunks.append(current_chunk)
                elif merged_chunks:
                    # If too small, append to previous chunk
                    merged_chunks[-1] += " " + current_chunk
                else:
                    # First chunk, keep it even if small
                    merged_chunks.append(current_chunk)
                
                # Start new chunk
                current_chunk = segments[i]
                current_embedding = embeddings[i]
        
        # Add the last chunk
        if len(current_chunk) >= min_chunk_size:
            merged_chunks.append(current_chunk)
        elif merged_chunks:
            merged_chunks[-1] += " " + current_chunk
        else:
            merged_chunks.append(current_chunk)
        
        # Step 4: Create Document objects with metadata
        for chunk_text in merged_chunks:
            chunk_doc = Document(
                page_content=chunk_text,
                metadata=doc.metadata.copy()
            )
            all_chunks.append(chunk_doc)
    
    print(f"Smart splitting: {len(documents)} documents -> {len(all_chunks)} semantic chunks")
    return all_chunks


def ingest_documents(data_path: str):
    """
    Main ingestion function: Load, chunk, and add documents to vector database.
    Uses smart semantic chunking by default (configured in config.py).
    
    Args:
        data_path: Path to the directory containing PDF files to ingest
    """
    print("=" * 60)
    print("DOCUMENT INGESTION")
    print("=" * 60)
    
    # Get configuration
    chunking_config = RAGConfig.get_chunking_config()
    
    # Determine if it's a file or directory
    if os.path.isfile(data_path):
        print(f"\nPDF file: {data_path}")
    else:
        print(f"\nData directory: {data_path}")
    print(f"Chunking method: Smart (semantic)")
    print("Configuration:")
    for key, value in chunking_config.items():
        print(f"  {key}: {value}")
    print()
    
    # Load documents
    print("Loading documents...")
    documents = load_documents(data_path)
    print(f"Loaded {len(documents)} documents\n")
    
    # Split documents using smart chunking
    print("Splitting documents into chunks...")
    chunks = smart_split_documents(documents, **chunking_config)
    
    print()
    
    # Add to database
    print("Adding chunks to vector database...")
    add_to_chroma(chunks)
    
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    # Example: Run ingestion with default path
    from secret import DATA_PATH
    ingest_documents(DATA_PATH)

