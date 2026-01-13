from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
import re

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from config import Config


@dataclass
class DocumentChunk:
    """Data class to store document chunk information"""
    text: str
    page_number: int
    document_name: str
    chunk_index: int
    metadata: Dict[str, Any]
    
    def __repr__(self):
        return f"DocumentChunk(doc={self.document_name}, page={self.page_number}, chunk={self.chunk_index})"


class DocumentProcessor:
    """Handles document processing pipeline: extraction, chunking, and embedding"""
    
    def __init__(self):
        """Initialize the document processor"""
        self.config = Config
        self.embeddings = None
        self.embedding_client = None
        
        # Force local embeddings for now (Groq doesn't support embeddings yet)
        # Try local embeddings first
        try:
            from local_embeddings import LocalEmbeddings
            self.embeddings = LocalEmbeddings()
            print("✅ Using local sentence-transformers for embeddings")
            return  # Exit early if local embeddings work
        except ImportError:
            print("⚠️ sentence-transformers not installed")
            print("   Install with: pip install sentence-transformers")
        except Exception as e:
            print(f"⚠️ Could not initialize local embeddings: {e}")
        
        # Fallback to API-based embeddings if local fails
        if self.config.EMBEDDING_PROVIDER == "openai":
            try:
                from langchain_openai import OpenAIEmbeddings
                if self.config.EMBEDDING_API_KEY:
                    self.embeddings = OpenAIEmbeddings(
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
                    self.embeddings = "custom"
                    print("✅ Using custom API for embeddings")
            except ImportError:
                print("⚠️ openai package not installed")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[int, str]:
        """
        Extract text from PDF with page-level metadata
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary mapping page numbers to extracted text
        """
        print(f"📄 Extracting text from: {pdf_path.name}")
        
        try:
            reader = PdfReader(str(pdf_path))
            page_texts = {}
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                
                # Clean extracted text
                text = self._clean_text(text)
                
                if text.strip():  # Only store non-empty pages
                    page_texts[page_num] = text
            
            print(f"✅ Extracted {len(page_texts)} pages from {pdf_path.name}")
            return page_texts
            
        except Exception as e:
            print(f"❌ Error extracting text from {pdf_path.name}: {e}")
            return {}
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and special characters
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def intelligent_chunking(
        self, 
        page_texts: Dict[int, str], 
        document_name: str
    ) -> List[DocumentChunk]:
        """
        Perform intelligent semantic chunking on extracted text
        
        Args:
            page_texts: Dictionary mapping page numbers to text
            document_name: Name of the source document
            
        Returns:
            List of DocumentChunk objects
        """
        print(f"✂️ Chunking document: {document_name}")
        
        # Initialize text splitter with semantic awareness
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],  # Semantic separators
            length_function=len,
        )
        
        chunks = []
        chunk_index = 0
        
        for page_num, page_text in page_texts.items():
            # Split page text into chunks
            page_chunks = text_splitter.split_text(page_text)
            
            for chunk_text in page_chunks:
                chunk = DocumentChunk(
                    text=chunk_text,
                    page_number=page_num,
                    document_name=document_name,
                    chunk_index=chunk_index,
                    metadata={
                        "source": document_name,
                        "page": page_num,
                        "chunk_id": chunk_index,
                        "char_count": len(chunk_text)
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
        
        print(f"✅ Created {len(chunks)} chunks from {document_name}")
        return chunks
    
    def create_embeddings(self, chunks: List[DocumentChunk]) -> List[List[float]]:
        """
        Generate vector embeddings for document chunks
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            List of embedding vectors
        """
        if not self.embeddings:
            raise ValueError("Embedding API not configured. Please check your .env file.")
        
        print(f"🔢 Generating embeddings for {len(chunks)} chunks...")
        
        # Extract text from chunks
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings based on provider
        if self.embeddings == "custom":
            # Use custom API
            embeddings = []
            for text in texts:
                try:
                    response = self.embedding_client.embeddings.create(
                        model=self.config.EMBEDDING_MODEL,
                        input=text
                    )
                    embeddings.append(response.data[0].embedding)
                except Exception as e:
                    print(f"⚠️ Error generating embedding: {e}")
                    # Return zero vector as fallback
                    embeddings.append([0.0] * 1536)  # Default embedding dimension
        else:
            # Use OpenAI LangChain embeddings
            embeddings = self.embeddings.embed_documents(texts)
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings
    
    def process_document(self, pdf_path: Path) -> tuple[List[DocumentChunk], List[List[float]]]:
        """
        Complete document processing pipeline
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (chunks, embeddings)
        """
        document_name = pdf_path.stem  # Get filename without extension
        
        # Step 1: Extract text from PDF
        page_texts = self.extract_text_from_pdf(pdf_path)
        
        if not page_texts:
            print(f"⚠️ No text extracted from {pdf_path.name}")
            return [], []
        
        # Step 2: Intelligent chunking
        chunks = self.intelligent_chunking(page_texts, document_name)
        
        if not chunks:
            print(f"⚠️ No chunks created from {pdf_path.name}")
            return [], []
        
        # Step 3: Generate embeddings
        try:
            embeddings = self.create_embeddings(chunks)
            return chunks, embeddings
        except ValueError as e:
            print(f"⚠️ {e}")
            return chunks, []
    
    def process_all_documents(self) -> tuple[List[DocumentChunk], List[List[float]]]:
        """
        Process all configured PDF documents
        
        Returns:
            Tuple of (all_chunks, all_embeddings)
        """
        all_chunks = []
        all_embeddings = []
        
        print("\n" + "="*60)
        print("📚 PROCESSING ALL DOCUMENTS")
        print("="*60 + "\n")
        
        for pdf_path in self.config.PDF_DOCUMENTS:
            if not pdf_path.exists():
                print(f"⚠️ PDF not found: {pdf_path}")
                continue
            
            chunks, embeddings = self.process_document(pdf_path)
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)
        
        print("\n" + "="*60)
        print(f"✅ PROCESSING COMPLETE")
        print(f"   Total Chunks: {len(all_chunks)}")
        print(f"   Total Embeddings: {len(all_embeddings)}")
        print("="*60 + "\n")
        
        return all_chunks, all_embeddings


# Test the module
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    # Validate configuration
    errors = Config.validate()
    if errors:
        print("\n⚠️ Configuration Issues:")
        for error in errors:
            print(f"  {error}")
    
    # Process all documents
    chunks, embeddings = processor.process_all_documents()
    
    # Display sample chunk
    if chunks:
        print("\n📋 Sample Chunk:")
        print(f"   {chunks[0]}")
        print(f"   Text Preview: {chunks[0].text[:200]}...")
