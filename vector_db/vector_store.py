import pickle
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np

import faiss
from langchain_openai import OpenAIEmbeddings

from config import Config
from document_processor import DocumentChunk


class VectorStore:
    """FAISS-based vector database for document retrieval"""
    
    def __init__(self):
        """Initialize the vector store"""
        self.config = Config
        self.index: Optional[faiss.Index] = None
        self.chunks: List[DocumentChunk] = []
        self.embeddings_model = None
        self.embedding_client = None
        
        # Force local embeddings first (Groq doesn't support embeddings)
        try:
            from local_embeddings import LocalEmbeddings
            self.embeddings_model = LocalEmbeddings()
            print("✅ Vector store using local embeddings")
            # Create vector DB directory if it doesn't exist
            self.config.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
            return  # Exit early if local works
        except Exception as e:
            print(f"⚠️ Could not initialize local embeddings: {e}")
        
        # Fallback to API-based embeddings
        if self.config.EMBEDDING_PROVIDER == "openai":
            try:
                from langchain_openai import OpenAIEmbeddings
                if self.config.EMBEDDING_API_KEY:
                    self.embeddings_model = OpenAIEmbeddings(
                        model=self.config.EMBEDDING_MODEL,
                        openai_api_key=self.config.EMBEDDING_API_KEY
                    )
            except ImportError:
                print("⚠️ langchain_openai not installed")
        
        elif self.config.EMBEDDING_PROVIDER == "custom":
            try:
                from openai import OpenAI
                if self.config.EMBEDDING_API_KEY and self.config.EMBEDDING_BASE_URL:
                    self.embedding_client = OpenAI(
                        api_key=self.config.EMBEDDING_API_KEY,
                        base_url=self.config.EMBEDDING_BASE_URL
                    )
                    self.embeddings_model = "custom"
            except ImportError:
                print("⚠️ openai package not installed")
        
        # Create vector DB directory if it doesn't exist
        self.config.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    
    def create_index(self, dimension: int):
        """
        Create a new FAISS index
        
        Args:
            dimension: Dimension of embedding vectors
        """
        # Use L2 (Euclidean) distance for similarity
        self.index = faiss.IndexFlatL2(dimension)
        print(f"✅ Created FAISS index with dimension {dimension}")
    
    def add_documents(
        self, 
        chunks: List[DocumentChunk], 
        embeddings: List[List[float]]
    ):
        """
        Add documents to the vector store
        
        Args:
            chunks: List of DocumentChunk objects
            embeddings: List of embedding vectors
        """
        if not chunks or not embeddings:
            print("⚠️ No chunks or embeddings to add")
            return
        
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Create index if it doesn't exist
        if self.index is None:
            dimension = embeddings_array.shape[1]
            self.create_index(dimension)
        
        # Add embeddings to FAISS index
        self.index.add(embeddings_array)
        
        # Store chunks for metadata retrieval
        self.chunks.extend(chunks)
        
        print(f"✅ Added {len(chunks)} documents to vector store")
        print(f"   Total documents in store: {len(self.chunks)}")
    
    def similarity_search(
        self, 
        query: str, 
        k: Optional[int] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar documents using semantic similarity
        
        Args:
            query: Search query text
            k: Number of top results to return (default: from config)
            
        Returns:
            List of (DocumentChunk, distance_score) tuples
        """
        if self.index is None or len(self.chunks) == 0:
            print("⚠️ Vector store is empty. Please add documents first.")
            return []
        
        if not self.embeddings_model:
            raise ValueError("Embedding API not configured. Cannot perform search.")
        
        # Use config default if k not specified
        if k is None:
            k = self.config.TOP_K_CHUNKS
        
        # Ensure k doesn't exceed available chunks
        k = min(k, len(self.chunks))
        
        # Generate query embedding based on provider
        if self.embeddings_model == "custom":
            # Use custom API
            try:
                response = self.embedding_client.embeddings.create(
                    model=self.config.EMBEDDING_MODEL,
                    input=query
                )
                query_embedding = response.data[0].embedding
            except Exception as e:
                print(f"❌ Error generating query embedding: {e}")
                return []
        else:
            # Use OpenAI LangChain embeddings
            query_embedding = self.embeddings_model.embed_query(query)
        
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Search FAISS index
        distances, indices = self.index.search(query_vector, k)
        
        # Retrieve chunks with their similarity scores
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):  # Safety check
                chunk = self.chunks[idx]
                results.append((chunk, float(distance)))
        
        return results
    
    def get_chunk_metadata(self, chunk: DocumentChunk) -> dict:
        """
        Get metadata for a document chunk
        
        Args:
            chunk: DocumentChunk object
            
        Returns:
            Dictionary with metadata
        """
        return {
            "document_name": chunk.document_name,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "text_preview": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
            **chunk.metadata
        }
    
    def save_to_disk(self):
        """Save the vector store to disk"""
        if self.index is None:
            print("⚠️ No index to save")
            return
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.config.FAISS_INDEX_FILE))
            
            # Save chunks metadata
            with open(self.config.METADATA_FILE, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            print(f"✅ Saved vector store to disk")
            print(f"   Index: {self.config.FAISS_INDEX_FILE}")
            print(f"   Metadata: {self.config.METADATA_FILE}")
            
        except Exception as e:
            print(f"❌ Error saving vector store: {e}")
    
    def load_from_disk(self) -> bool:
        """
        Load the vector store from disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if files exist
            if not self.config.FAISS_INDEX_FILE.exists():
                print("⚠️ No saved index found")
                return False
            
            if not self.config.METADATA_FILE.exists():
                print("⚠️ No saved metadata found")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.config.FAISS_INDEX_FILE))
            
            # Load chunks metadata
            with open(self.config.METADATA_FILE, 'rb') as f:
                self.chunks = pickle.load(f)
            
            print(f"✅ Loaded vector store from disk")
            print(f"   Total documents: {len(self.chunks)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store"""
        if self.index is None:
            return {"status": "empty"}
        
        # Count documents per source
        doc_counts = {}
        for chunk in self.chunks:
            doc_name = chunk.document_name
            doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1
        
        return {
            "status": "ready",
            "total_chunks": len(self.chunks),
            "index_size": self.index.ntotal,
            "dimension": self.index.d,
            "documents": doc_counts
        }


# Test the module
if __name__ == "__main__":
    from document_processor import DocumentProcessor
    
    # Initialize components
    processor = DocumentProcessor()
    vector_store = VectorStore()
    
    # Try to load existing vector store
    if not vector_store.load_from_disk():
        print("\n📚 No existing vector store found. Processing documents...")
        
        # Process all documents
        chunks, embeddings = processor.process_all_documents()
        
        if chunks and embeddings:
            # Add to vector store
            vector_store.add_documents(chunks, embeddings)
            
            # Save to disk
            vector_store.save_to_disk()
    
    # Display statistics
    stats = vector_store.get_stats()
    print("\n📊 Vector Store Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test search (if API key is configured)
    if vector_store.embeddings_model:
        print("\n🔍 Testing similarity search...")
        results = vector_store.similarity_search("What is artificial intelligence?", k=2)
        
        for i, (chunk, score) in enumerate(results, 1):
            print(f"\n   Result {i}:")
            print(f"   Document: {chunk.document_name}")
            print(f"   Page: {chunk.page_number}")
            print(f"   Score: {score:.4f}")
            print(f"   Text: {chunk.text[:150]}...")
