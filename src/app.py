"""
Streamlit Web Interface for Bible Commentary Bot
Provides a user-friendly interface for document ingestion and querying.
"""
import streamlit as st
import sys
import os
import time
import traceback
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.agent import create_rag_agent
from src.agents.llm import get_llm
from src.config import RAGConfig
from src.ingest.ingest import ingest_documents
from langchain_core.messages import HumanMessage, AIMessage


# Page configuration
st.set_page_config(
    page_title="Bible Commentary Bot",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "llm_initialized" not in st.session_state:
        st.session_state.llm_initialized = False


def initialize_agent():
    """Initialize the RAG agent if not already done."""
    if st.session_state.llm_initialized and st.session_state.agent is not None:
        return
    
    try:
        with st.spinner("Initializing agent..."):
            # Get LLM configuration
            llm_config = RAGConfig.get_llm_config()
            memory_config = RAGConfig.get_memory_config()
            
            # Get the LLM
            llm = get_llm(
                provider=llm_config.get("provider", "ollama"),
                model=llm_config.get("model"),
                temperature=llm_config.get("temperature", 0.7)
            )
            
            # Create agent with memory
            st.session_state.agent = create_rag_agent(
                llm,
                enable_memory=memory_config.get("enable_memory", True),
                enable_checkpointing=memory_config.get("enable_checkpointing", False)
            )
            st.session_state.llm_initialized = True
            return True
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return False


def convert_messages_to_langchain(messages):
    """Convert session state messages to LangChain message format."""
    langchain_messages = []
    for msg in messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))
    return langchain_messages


def stream_agent_response(query: str, conversation_history: list):
    """Stream the agent's response in real-time with full conversation context."""
    if st.session_state.agent is None:
        if not initialize_agent():
            return
    
    try:
        # Create message container for streaming
        message_placeholder = st.empty()
        full_response = ""
        
        # Convert conversation history to LangChain format
        history_messages = convert_messages_to_langchain(conversation_history)
        
        # Add the current query
        current_message = HumanMessage(content=query)
        all_messages = history_messages + [current_message]
        
        # Try to use streaming if available
        try:
            # Use astream for real-time streaming with full conversation history
            stream = st.session_state.agent.astream({
                "messages": all_messages
            })
            
            # Stream chunks as they arrive
            for chunk in stream:
                if "messages" in chunk and len(chunk["messages"]) > 0:
                    # Get the latest message content
                    latest_message = chunk["messages"][-1]
                    if hasattr(latest_message, 'content') and latest_message.content:
                        # Update with new content
                        new_content = latest_message.content
                        if len(new_content) > len(full_response):
                            # Only show new content
                            new_chunk = new_content[len(full_response):]
                            full_response = new_content
                            message_placeholder.markdown(full_response + "▌")
            
            # Final response without cursor
            message_placeholder.markdown(full_response)
            return full_response
            
        except (AttributeError, TypeError):
            # Fallback to non-streaming if astream is not available
            result = st.session_state.agent.invoke({
                "messages": all_messages
            })
            
            # Extract response from the last message
            if result and "messages" in result and len(result["messages"]) > 0:
                response_text = result["messages"][-1].content
                
                # Simulate streaming character by character
                for chunk in response_text:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)  # Small delay for visual effect
                
                # Final response without cursor
                message_placeholder.markdown(full_response)
                return full_response
            else:
                st.error("No response received from agent")
                return None
            
    except Exception as e:
        st.error(f"Error during query: {str(e)}")
        st.code(traceback.format_exc())
        return None


def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📖 Bible Commentary Bot</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Display current configuration
        with st.expander("📋 Current Configuration", expanded=False):
            config = RAGConfig.get_all_config()
            
            st.write("**LLM Settings:**")
            st.json({
                "provider": config["llm"].get("provider"),
                "model": config["llm"].get("model"),
                "temperature": config["llm"].get("temperature")
            })
            
            st.write("**Memory Settings:**")
            st.json(config["memory"])
            
            st.write("**Retrieval Settings:**")
            st.json(config["retrieval"])
        
        # Reset button
        if st.button("🔄 Reset Agent", help="Reinitialize the agent"):
            st.session_state.agent = None
            st.session_state.llm_initialized = False
            st.session_state.messages = []
            st.rerun()
    
    # Main tabs
    tab1, tab2 = st.tabs(["💬 Query", "📥 Ingest Documents"])
    
    # Query Tab
    with tab1:
        st.markdown('<h2 class="sub-header">Ask Questions About Bible Commentary</h2>', unsafe_allow_html=True)
        
        # Initialize agent on first use
        if not st.session_state.llm_initialized:
            if initialize_agent():
                st.success("✅ Agent initialized successfully!")
            else:
                st.error("❌ Failed to initialize agent. Please check your configuration.")
                return
        
        # Display chat history (messages appear in chronological order, oldest first)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input - Streamlit automatically places this at the bottom
        if prompt := st.chat_input("Ask a question about the Bible commentary..."):
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and stream response with full conversation history
            # Pass the current history (before adding the new message)
            with st.chat_message("assistant"):
                # Pass the conversation history and the new query
                response = stream_agent_response(prompt, st.session_state.messages)
                
                if response:
                    # Add both user message and assistant response to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to refresh the display with new messages
            st.rerun()
    
    # Ingest Tab
    with tab2:
        st.markdown('<h2 class="sub-header">Ingest PDF Documents</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>📚 Document Ingestion</strong><br>
            Upload PDF files or specify a directory path to ingest Bible commentary documents 
            into the vector database. The system will automatically chunk and index the documents 
            for semantic search.
        </div>
        """, unsafe_allow_html=True)
        
        # File upload option
        st.subheader("📤 Upload PDF Files")
        uploaded_files = st.file_uploader(
            "Choose PDF files to ingest",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select one or more PDF files to add to the database"
        )
        
        # Directory path option
        st.subheader("📁 Or Specify Directory Path")
        directory_path = st.text_input(
            "Enter directory path containing PDF files",
            value="data/",
            help="Path to a directory containing PDF files"
        )
        
        # Ingest button
        col1, col2 = st.columns([1, 4])
        with col1:
            ingest_button = st.button("🚀 Start Ingestion", type="primary", use_container_width=True)
        
        # Process ingestion
        if ingest_button:
            if uploaded_files:
                # Save uploaded files temporarily and ingest
                temp_dir = Path("temp_uploads")
                temp_dir.mkdir(exist_ok=True)
                
                try:
                    with st.spinner("Saving uploaded files..."):
                        saved_paths = []
                        for uploaded_file in uploaded_files:
                            file_path = temp_dir / uploaded_file.name
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            saved_paths.append(str(file_path))
                    
                    # Ingest each file
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, file_path in enumerate(saved_paths):
                        status_text.text(f"Processing {Path(file_path).name}... ({i+1}/{len(saved_paths)})")
                        try:
                            ingest_documents(file_path)
                            st.success(f"✅ Successfully ingested: {Path(file_path).name}")
                        except Exception as e:
                            st.error(f"❌ Error ingesting {Path(file_path).name}: {str(e)}")
                        
                        progress_bar.progress((i + 1) / len(saved_paths))
                    
                    # Cleanup
                    for file_path in saved_paths:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    if temp_dir.exists() and not any(temp_dir.iterdir()):
                        temp_dir.rmdir()
                    
                    status_text.text("✅ Ingestion complete!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error during ingestion: {str(e)}")
            
            elif directory_path and os.path.exists(directory_path):
                # Ingest from directory
                with st.spinner(f"Ingesting documents from {directory_path}..."):
                    try:
                        ingest_documents(directory_path)
                        st.success(f"✅ Successfully ingested documents from {directory_path}")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error during ingestion: {str(e)}")
            
            else:
                st.warning("⚠️ Please upload PDF files or provide a valid directory path")
        
        # Show ingestion info
        with st.expander("ℹ️ Ingestion Information", expanded=False):
            st.markdown("""
            **How it works:**
            1. Documents are loaded and split into semantic chunks
            2. Chunks are embedded using the configured embedding model
            3. Chunks are stored in the Chroma vector database
            4. Documents are indexed for fast semantic search
            
            **Smart Chunking:**
            - Uses semantic similarity to create meaningful chunks
            - Preserves context across chunk boundaries
            - Configurable chunk sizes and similarity thresholds
            
            **Configuration:**
            - Chunking settings can be adjusted in `src/config/__init__.py`
            - Or via environment variables (CHUNKING_*)
            """)


if __name__ == "__main__":
    main()

