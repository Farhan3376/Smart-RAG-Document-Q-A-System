from typing import List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers not installed. Install with: pip install sentence-transformers")


class LocalEmbeddings:
    """Local embedding generation using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize local embeddings
        
        Args:
            model_name: Sentence transformer model name
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers package not installed")
        
        print(f"📥 Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"✅ Model loaded successfully")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode([text])[0]
        return embedding.tolist()


# Test the module
if __name__ == "__main__":
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        embeddings = LocalEmbeddings()
        
        # Test embedding
        test_texts = ["Hello world", "This is a test"]
        result = embeddings.embed_documents(test_texts)
        print(f"\n✅ Generated {len(result)} embeddings")
        print(f"   Embedding dimension: {len(result[0])}")
    else:
        print("\n❌ Please install sentence-transformers:")
        print("   pip install sentence-transformers")
