from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.ingest.create_database import add_to_chroma
from src.config.secret import DATA_PATH
from src.agents.embeddings import get_chunking_embedding_function


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document], chunk: int):
    """Basic splitting function (kept for backward compatibility)."""
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


if __name__ == "__main__":
    from src.config import RAGConfig
    
    # Print configuration
    print("Using smart chunking with the following configuration:")
    chunking_config = RAGConfig.get_chunking_config()
    for key, value in chunking_config.items():
        print(f"  {key}: {value}")
    print()
    
    # Load documents
    documents = load_documents()
    
    # Use smart splitting with config values
    chunks = smart_split_documents(
        documents,
        **chunking_config
    )
    
    # Add to database
    add_to_chroma(chunks)