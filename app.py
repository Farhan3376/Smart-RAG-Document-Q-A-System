import streamlit as st
import time
from datetime import datetime
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from document_processor import DocumentProcessor
from groq_models import get_available_models, get_recommended_models

# Page configuration
st.set_page_config(
    page_title="3D Reconstruction Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Claude-inspired design
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f8f7f4;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e5e5;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        margin-left: 10%;
    }
    
    .assistant-message {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        margin-right: 10%;
    }
    
    .message-role {
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: #6b6b6b;
    }
    
    .message-content {
        line-height: 1.6;
        color: #2d2d2d;
    }
    
    /* Input box styling */
    .stTextInput input {
        border-radius: 1rem;
        border: 1px solid #d4d4d4;
        padding: 0.75rem 1rem;
        font-size: 1rem;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 0.5rem;
        background-color: #d97757;
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background-color: #c76847;
        box-shadow: 0 2px 8px rgba(217, 119, 87, 0.3);
    }
    
    /* Title styling */
    h1 {
        color: #2d2d2d;
        font-weight: 400;
        font-size: 2rem;
        text-align: center;
        margin-top: 2rem;
    }
    
    /* Greeting styling */
    .greeting-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        min-height: 50vh;
    }
    
    .greeting-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .greeting-text {
        font-size: 2.5rem;
        color: #2d2d2d;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    
    .greeting-subtitle {
        font-size: 1rem;
        color: #6b6b6b;
        margin-bottom: 3rem;
    }
    
    /* Source citation styling */
    .source-citation {
        background-color: #f0f0f0;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        font-size: 0.85rem;
        color: #555;
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    .status-ready {
        background-color: #22c55e;
    }
    
    .status-error {
        background-color: #ef4444;
    }
    
    .status-warning {
        background-color: #f59e0b;
    }
    
    /* Sidebar navigation */
    .nav-item {
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 0.5rem;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .nav-item:hover {
        background-color: #f3f4f6;
    }
    
    .nav-item-active {
        background-color: #e5e7eb;
    }
    
    /* Document stats */
    .doc-stats {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
    }
    
    /* Loading animation */
    .loading-dots {
        display: inline-block;
    }
    
    .loading-dots::after {
        content: '...';
        animation: dots 1.5s steps(4, end) infinite;
    }
    
    @keyframes dots {
        0%, 20% { content: '.'; }
        40% { content: '..'; }
        60%, 100% { content: '...'; }
    }
    
    /* Model selector styling */
    .model-selector-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
        align-items: center;
    }
    
    .stSelectbox {
        flex: 1;
    }
    
    .stSlider {
        flex: 1;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = None
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = "chat"
    
    if "model_settings" not in st.session_state:
        st.session_state.model_settings = {
            "temperature": Config.TEMPERATURE,
            "top_k": Config.TOP_K_CHUNKS,
            "model_name": Config.LLM_MODEL
        }


def load_rag_system():
    """Initialize RAG system components"""
    try:
        # Initialize vector store
        if st.session_state.vector_store is None:
            st.session_state.vector_store = VectorStore()
            
            # Try to load from disk
            if st.session_state.vector_store.load_from_disk():
                st.session_state.documents_loaded = True
        
        # Initialize RAG pipeline
        if st.session_state.rag_pipeline is None:
            st.session_state.rag_pipeline = RAGPipeline(st.session_state.vector_store)
        
        return True
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return False


def render_sidebar():
    """Render the sidebar navigation and controls"""
    with st.sidebar:
        # Logo/Title
        st.markdown("### 🔬 RAG Research Assistant")
        st.markdown("---")
        
        # Navigation
        st.markdown("#### Navigation")
        
        if st.button("💬 Chat", use_container_width=True, 
                    type="primary" if st.session_state.current_page == "chat" else "secondary"):
            st.session_state.current_page = "chat"
            st.rerun()
        
        if st.button("📚 Documents", use_container_width=True,
                    type="primary" if st.session_state.current_page == "documents" else "secondary"):
            st.session_state.current_page = "documents"
            st.rerun()
        
        if st.button("⚙️ Settings", use_container_width=True,
                    type="primary" if st.session_state.current_page == "settings" else "secondary"):
            st.session_state.current_page = "settings"
            st.rerun()
        
        if st.button("ℹ️ About", use_container_width=True,
                    type="primary" if st.session_state.current_page == "about" else "secondary"):
            st.session_state.current_page = "about"
            st.rerun()
        
        st.markdown("---")
        
        # System status
        st.markdown("#### System Status")
        
        # Check API configuration
        api_status = "✅ Ready" if Config.API_KEY and Config.API_BASE_URL else "⚠️ Not Configured"
        st.markdown(f"**API:** {api_status}")
        
        # Check documents
        if st.session_state.documents_loaded:
            stats = st.session_state.vector_store.get_stats()
            doc_count = len(stats.get("documents", {}))
            chunk_count = stats.get("total_chunks", 0)
            st.markdown(f"**Documents:** {doc_count} loaded")
            st.markdown(f"**Chunks:** {chunk_count}")
        else:
            st.markdown("**Documents:** ⚠️ Not loaded")
        
        st.markdown("---")
        
        # Model Controls
        st.markdown("#### Model Controls")
        
        # Temperature
        temp_value = st.slider(
            "🌡️ Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.model_settings["temperature"],
            step=0.1,
            help="0.0 = precise/factual | 2.0 = creative/varied",
            key="sidebar_temp_slider"
        )
        st.session_state.model_settings["temperature"] = temp_value
        
        # Retrieval Sources
        sources_value = st.slider(
            "📚 Retrieval Sources",
            min_value=1,
            max_value=10,
            value=st.session_state.model_settings["top_k"],
            help="Number of document chunks to retrieve",
            key="sidebar_topk_slider"
        )
        st.session_state.model_settings["top_k"] = sources_value
        
        st.markdown("---")
        
        # New chat button
        if st.button("🔄 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Footer
        st.markdown("---")
        st.markdown("**CSE-848** • NUST SMME")
        st.markdown("*Generative AI & Applications*")


def render_greeting():
    """Render the initial greeting screen"""
    # Get current time for greeting
    current_hour = datetime.now().hour
    if current_hour < 12:
        greeting = "Good morning"
    elif current_hour < 18:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"
    
    st.markdown(f"""
    <div class="greeting-container">
        <div class="greeting-icon">🔬</div>
        <div class="greeting-text">{greeting}!</div>
        <div class="greeting-subtitle">Ask me anything about your research papers</div>
    </div>
    """, unsafe_allow_html=True)


def render_message(role, content, sources=None):
    """Render a chat message"""
    message_class = "user-message" if role == "user" else "assistant-message"
    role_display = "You" if role == "user" else "Assistant"
    
    html = f"""
    <div class="chat-message {message_class}">
        <div class="message-role">{role_display}</div>
        <div class="message-content">{content}</div>
    """
    
    # Add sources if provided
    if sources and role == "assistant":
        sources_html = "<div class='source-citation'><strong>Sources:</strong><br>"
        for source in sources:
            sources_html += f"• {source['document']}, Page {source['page']}<br>"
        sources_html += "</div>"
        html += sources_html
    
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def on_model_change():
    """Callback when model is changed - syncs selectbox value to session state"""
    st.session_state.model_settings["model_name"] = st.session_state.model_selector_key


def on_send_button_click():
    """Callback when send button is clicked"""
    user_input = st.session_state.user_input
    if user_input and user_input.strip():
        # Process the query
        process_user_query(user_input)
        # Clear the input field after processing
        st.session_state.user_input = ""


def render_chat_page():
    """Render the main chat interface with model selector directly above input"""
    # Display chat history
    if not st.session_state.messages:
        render_greeting()
    else:
        for message in st.session_state.messages:
            render_message(
                message["role"],
                message["content"],
                message.get("sources")
            )
    
    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Dynamically fetch available Groq models
    try:
        available_models = get_available_models(include_preview=False)
    except Exception as e:
        # Fallback models if API is unavailable
        available_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma-7b-it"
        ]
    
    # Model Selector - DIRECTLY ABOVE INPUT (Claude-style)
    st.markdown("**Model:**")
    
    current_model = st.session_state.model_settings["model_name"]
    model_index = available_models.index(current_model) if current_model in available_models else 0
    
    # Selectbox with proper state management
    st.selectbox(
        "Select Model",
        options=available_models,
        index=model_index,
        key="model_selector_key",
        on_change=on_model_change,
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Input and Send Button
    col1, col2 = st.columns([6, 1])
    
    with col1:
        st.text_input(
            "Message",
            placeholder="Ask a question about your research papers...",
            key="user_input",
            label_visibility="collapsed"
        )
    
    with col2:
        st.button("Send", use_container_width=True, on_click=on_send_button_click)


def process_user_query(query):
    """Process user query through RAG pipeline"""
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })
    
    # Check if system is ready
    is_ready, message = st.session_state.rag_pipeline.is_ready()
    
    if not is_ready:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ System not ready: {message}\n\nPlease configure your API keys and load documents first.",
            "sources": None
        })
        return
    
    # Show processing indicator
    with st.spinner("Thinking..."):
        try:
            # Get response from RAG pipeline
            response = st.session_state.rag_pipeline.process_query(
                query,
                temperature=st.session_state.model_settings["temperature"],
                model_name=st.session_state.model_settings["model_name"],
                top_k=st.session_state.model_settings["top_k"]
            )
            
            # Add assistant response to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources
            })
            
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Error: {str(e)}\n\nPlease check your configuration and try again.",
                "sources": None
            })


def render_documents_page():
    """Render the documents management page"""
    st.title("📚 Document Management")
    
    # Document statistics
    if st.session_state.documents_loaded:
        stats = st.session_state.vector_store.get_stats()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Documents", len(stats.get("documents", {})))
        
        with col2:
            st.metric("Total Chunks", stats.get("total_chunks", 0))
        
        with col3:
            st.metric("Embedding Dimension", stats.get("dimension", 0))
        
        st.markdown("---")
        
        # Document list
        st.subheader("Loaded Documents")
        for doc_name, chunk_count in stats.get("documents", {}).items():
            st.markdown(f"**{doc_name}**")
            st.progress(min(chunk_count / stats.get("total_chunks", 1), 1.0))
            st.caption(f"{chunk_count} chunks")
            st.markdown("<br>", unsafe_allow_html=True)
    
    else:
        st.warning("No documents loaded. Please process documents first.")
    
    st.markdown("---")
    
    # Process documents button
    st.subheader("Process New Documents")
    st.info(f"📁 Document directory: `{Config.DATA_DIR}`")
    
    if st.button("🔄 Process Documents", use_container_width=True):
        process_documents()


def process_documents():
    """Process documents and create vector store"""
    with st.spinner("Processing documents..."):
        try:
            # Initialize processor
            processor = DocumentProcessor()
            
            # Process all documents
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Loading documents...")
            progress_bar.progress(0.2)
            
            chunks, embeddings = processor.process_all_documents()
            
            if not chunks or not embeddings:
                st.error("Failed to process documents!")
                return
            
            status_text.text("Creating vector store...")
            progress_bar.progress(0.6)
            
            # Create vector store
            vector_store = VectorStore()
            vector_store.add_documents(chunks, embeddings)
            
            status_text.text("Saving to disk...")
            progress_bar.progress(0.8)
            
            # Save to disk
            vector_store.save_to_disk()
            
            progress_bar.progress(1.0)
            status_text.text("Complete!")
            
            # Update session state
            st.session_state.vector_store = vector_store
            st.session_state.documents_loaded = True
            
            # Reinitialize RAG pipeline
            st.session_state.rag_pipeline = RAGPipeline(vector_store)
            
            st.success(f"✅ Successfully processed {len(chunks)} chunks!")
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")


def render_settings_page():
    """Render the settings page"""
    st.title("⚙️ Settings")
    
    st.info("""
    ℹ️ **Note:** Model settings are now configurable directly on the chat page for easier access. 
    You can adjust them anytime while chatting.
    """)
    
    st.markdown("---")
    
    # API Configuration
    st.subheader("API Configuration")
    
    api_key_status = "✅ Configured" if Config.API_KEY else "❌ Not Set"
    api_url_status = "✅ Configured" if Config.API_BASE_URL else "❌ Not Set"
    
    st.info(f"""
    **API Key:** {api_key_status}  
    **Base URL:** {api_url_status}
    
    Update these in your `.env` file or `config.py`
    """)
    
    st.markdown("---")
    
    # System Info
    st.subheader("System Information")
    
    st.code(Config.get_summary(), language=None)


def render_about_page():
    """Render the about page"""
    st.title("ℹ️ About")
    
    st.markdown("""
    ### 🔬 3D Reconstruction Research Assistant
    
    An intelligent document Q&A system powered by Retrieval-Augmented Generation (RAG) 
    for analyzing research papers on 3D reconstruction and computer vision.
    
    #### Features
    - **📚 Document Processing:** Automatically chunks and indexes research papers
    - **🔍 Semantic Search:** Finds relevant information using vector similarity
    - **🤖 AI-Powered Answers:** Generates accurate responses grounded in your documents
    - **📊 Source Citations:** Always provides references to source papers
    - **⚡ Anti-Hallucination:** Configured to only answer from provided documents
    - **🎛️ Easy Model Selection:** Switch models directly from the chat interface
    
    #### Technology Stack
    - **Frontend:** Streamlit
    - **Vector Database:** FAISS
    - **Embeddings:** Local (Sentence Transformers) or API-based
    - **LLM:** Groq Llama 3.3 70B (or custom)
    - **Document Processing:** PyMuPDF, LangChain
    
    #### Course Information
    - **Course:** CSE-848 - Generative AI and Applications
    - **University:** NUST (SMME)
    - **Project:** Smart RAG Document Q&A System
    
    ---
    
    **Version:** 1.1.0  
    **Last Updated:** December 2024
    """)


def main():
    """Main application entry point"""
    # Initialize session state
    initialize_session_state()
    
    # Load RAG system
    if not load_rag_system():
        st.error("Failed to initialize RAG system. Please check your configuration.")
        return
    
    # Render sidebar
    render_sidebar()
    
    # Render current page
    if st.session_state.current_page == "chat":
        render_chat_page()
    elif st.session_state.current_page == "documents":
        render_documents_page()
    elif st.session_state.current_page == "settings":
        render_settings_page()
    elif st.session_state.current_page == "about":
        render_about_page()


if __name__ == "__main__":
    main()