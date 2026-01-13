import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration class for the RAG system"""
    
    # ==================== API SETTINGS ====================
    # API Key (loaded from .env file)
    API_KEY = os.getenv("GROQ_API_KEY", "")  # Your custom LLM API key
    
    # Custom LLM API Configuration
    API_BASE_URL = os.getenv("API_BASE_URL", "")  # Base URL for your LLM API
    
    # LLM Model Configuration
    LLM_MODEL = "llama-3.3-70b-versatile"  # Groq's Llama 3.3 70B (recommended for RAG)
    
    # Embedding Configuration
    # Options: "openai" or "custom"
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "custom")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v1.5")  # Groq embedding model
    EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")  # For embeddings
    EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "")  # For custom embeddings
    
    # Anti-Hallucination Settings
    TEMPERATURE = 0.3  # Low temperature for grounded, factual responses (prevents hallucination)
    MAX_TOKENS = 500  # Maximum tokens in generated response
    
    # ==================== DOCUMENT SETTINGS ====================
    # Project root directory
    PROJECT_ROOT = Path(__file__).parent
    
    # PDF Documents Directory
    DATA_DIR = PROJECT_ROOT / "data"
    
    # PDF Document Paths (CONFIGURE THESE)
    PDF_DOCUMENTS = [
        DATA_DIR / "FAST3R.pdf",
        DATA_DIR / "MASt3R-SLAM.pdf",
    ]
    
    # ==================== CHUNKING PARAMETERS ====================
    # Chunk size for text splitting (in characters)
    CHUNK_SIZE = 1000  # Optimal for preserving semantic meaning
    
    # Overlap between chunks (in characters)
    CHUNK_OVERLAP = 200  # Ensures context continuity across chunks
    
    # Separator for text splitting
    CHUNK_SEPARATOR = "\n\n"  # Split on paragraph boundaries
    
    # ==================== RETRIEVAL SETTINGS ====================
    # Number of top chunks to retrieve for each query
    TOP_K_CHUNKS = 4  # Balance between context and relevance
    
    # Similarity score threshold (0-1, higher = more strict)
    SIMILARITY_THRESHOLD = 0.3  # Minimum similarity to consider a chunk relevant for hallucination prevention
    
    # ==================== VECTOR STORE SETTINGS ====================
    # Vector database directory
    VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
    
    # FAISS index filename
    FAISS_INDEX_FILE = VECTOR_DB_DIR / "faiss_index.bin"
    
    # Metadata storage filename
    METADATA_FILE = VECTOR_DB_DIR / "metadata.pkl"
    
    # ==================== UI SETTINGS ====================
    # Streamlit page configuration
    PAGE_TITLE = "3D Reconstruction Research Assistant"
    PAGE_ICON = "🔬"
    LAYOUT = "wide"
    
    # UI Text
    APP_TITLE = "🔬 3D Reconstruction Research Assistant"
    APP_SUBTITLE = "AI-Powered Research Paper Analysis | CSE-848"
    
    # ==================== PROMPT TEMPLATES ====================
    # System prompt for anti-hallucination
    SYSTEM_PROMPT = """You are an AI assistant specialized in analyzing 3D reconstruction and computer vision research papers.

⚠️ CRITICAL INSTRUCTIONS - STRICTLY FOLLOW:
1. ONLY answer questions using EXPLICITLY stated content from the provided research papers.
2. If the requested information is NOT in the papers, respond with:
   "This information is not available in the provided research papers."
3. NEVER use external knowledge, assumptions, or general AI knowledge.
4. NEVER infer or extrapolate beyond what is written in the papers.
5. NEVER generate plausible-sounding but unverified information.
6. Always cite exact source: paper name and page number.
7. If unsure whether information is in the papers, err on the side of saying it's not available.

REQUIREMENTS:
- Answer only from provided papers
- Be technically precise
- Always cite sources (Paper Name, Page #)
- Reject questions about topics not covered in papers
- If multiple papers cover topic, mention all relevant sources
- Never answer hypothetically or from general knowledge

Remember: Your credibility depends on accuracy. It's better to say information is unavailable than to hallucinate.
"""
    
    # User prompt template
    USER_PROMPT_TEMPLATE = """Context from documents:
{context}

Question: {question}

Answer (with source citations):"""
    
    # ==================== VALIDATION ====================
    @classmethod
    def validate(cls):
        """Validate configuration settings"""
        errors = []
        
        # Check if API key is set
        if not cls.API_KEY:
            errors.append("⚠️ API_KEY not set. Please add it to the .env file.")
        
        # Check if API base URL is set
        if not cls.API_BASE_URL:
            errors.append("⚠️ API_BASE_URL not set. Please add it to the .env file.")
        
        # Check if data directory exists
        if not cls.DATA_DIR.exists():
            cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
            errors.append(f"ℹ️ Created data directory: {cls.DATA_DIR}")
        
        # Check if PDF documents exist
        for pdf_path in cls.PDF_DOCUMENTS:
            if not pdf_path.exists():
                errors.append(f"⚠️ PDF not found: {pdf_path.name}")
        
        # Check if vector DB directory exists
        if not cls.VECTOR_DB_DIR.exists():
            cls.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
            errors.append(f"ℹ️ Created vector database directory: {cls.VECTOR_DB_DIR}")
        
        return errors
    
    @classmethod
    def get_summary(cls):
        """Get a summary of current configuration"""
        return f"""
╔══════════════════════════════════════════════════════════╗
║          SMART RAG SYSTEM CONFIGURATION                  ║
╚══════════════════════════════════════════════════════════╝

📄 Documents:
   • Total PDFs: {len(cls.PDF_DOCUMENTS)}
   • Location: {cls.DATA_DIR}

🤖 LLM Settings:
   • Model: {cls.LLM_MODEL}
   • Temperature: {cls.TEMPERATURE} (anti-hallucination)
   • Max Tokens: {cls.MAX_TOKENS}

✂️ Chunking:
   • Chunk Size: {cls.CHUNK_SIZE} characters
   • Overlap: {cls.CHUNK_OVERLAP} characters

🔍 Retrieval:
   • Top-K Chunks: {cls.TOP_K_CHUNKS}
   • Similarity Threshold: {cls.SIMILARITY_THRESHOLD}

🔑 API Configuration:
   • API Key: {'✅ Configured' if cls.API_KEY else '❌ Not Set'}
   • Base URL: {'✅ Configured' if cls.API_BASE_URL else '❌ Not Set'}
   • Embedding Provider: {cls.EMBEDDING_PROVIDER}
"""


# Validate configuration on import
if __name__ == "__main__":
    print(Config.get_summary())
    validation_errors = Config.validate()
    if validation_errors:
        print("\n⚠️ Configuration Issues:")
        for error in validation_errors:
            print(f"  {error}")
