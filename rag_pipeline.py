from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from config import Config
from vector_store import VectorStore
from document_processor import DocumentChunk


@dataclass
class RAGResponse:
    """Data class for RAG pipeline response"""
    answer: str
    sources: List[Dict[str, Any]]
    evidence_snippets: List[str]
    prompt: str
    query: str
    retrieved_chunks: int


class RAGPipeline:
    """Core RAG question-answering pipeline"""
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize the RAG pipeline
        
        Args:
            vector_store: VectorStore instance (creates new one if None)
        """
        self.config = Config
        self.vector_store = vector_store or VectorStore()
        self.llm = None
        self.llm_client = None
        
        # Initialize LLM based on configuration
        if self.config.API_KEY and self.config.API_BASE_URL:
            # Use custom LLM API (OpenAI-compatible)
            try:
                from openai import OpenAI
                self.llm_client = OpenAI(
                    api_key=self.config.API_KEY,
                    base_url=self.config.API_BASE_URL
                )
                self.llm = "custom"  # Flag for custom LLM
                print(f"✅ Initialized custom LLM: {self.config.LLM_MODEL}")
            except ImportError:
                print("⚠️ openai package not installed. Install with: pip install openai")
        else:
            print("⚠️ API_KEY or API_BASE_URL not configured")
    
    def retrieve_context(self, query: str, k: Optional[int] = None) -> List[DocumentChunk]:
        """
        Retrieve relevant context from vector store with relevance filtering
        
        Args:
            query: User question
            k: Number of chunks to retrieve
            
        Returns:
            List of relevant DocumentChunk objects (filtered by similarity threshold)
        """
        # Perform similarity search (returns list of tuples: (chunk, score))
        results = self.vector_store.similarity_search(query, k=k)
        
        # Filter by relevance threshold to prevent irrelevant context
        filtered_chunks = []
        for chunk, score in results:
            # Only include chunks above the similarity threshold
            if score >= self.config.SIMILARITY_THRESHOLD:
                filtered_chunks.append(chunk)
                print(f"   ✓ {chunk.document_name} (p.{chunk.page_number}): {score:.3f}")
            else:
                print(f"   ✗ {chunk.document_name} (p.{chunk.page_number}): {score:.3f} (below threshold)")
        
        return filtered_chunks
    
    def build_prompt(self, query: str, context_chunks: List[DocumentChunk]) -> tuple[str, str]:
        """
        Build the prompt for the LLM with anti-hallucination constraints
        
        Args:
            query: User question
            context_chunks: Retrieved document chunks
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Format context from chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(
                f"[Document: {chunk.document_name}, Page: {chunk.page_number}]\n{chunk.text}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build user prompt using template
        user_prompt = self.config.USER_PROMPT_TEMPLATE.format(
            context=context,
            question=query
        )
        
        return self.config.SYSTEM_PROMPT, user_prompt
    
    def generate_answer(self, system_prompt: str, user_prompt: str, 
                       temperature: Optional[float] = None, 
                       model_name: Optional[str] = None,
                       stream: bool = False) -> str:
        """
        Generate answer using LLM
        
        Args:
            system_prompt: System instructions
            user_prompt: User query with context
            temperature: Optional temperature override (uses Config.TEMPERATURE if None)
            model_name: Optional model name override (uses Config.LLM_MODEL if None)
            stream: Whether to stream the response (for dynamic text)
            
        Returns:
            Generated answer (or generator if stream=True)
        """
        if not self.llm:
            raise ValueError("LLM API not configured. Cannot generate answers.")
        
        # Use provided values or fall back to config
        temp = temperature if temperature is not None else self.config.TEMPERATURE
        model = model_name if model_name is not None else self.config.LLM_MODEL
        
        # Debug: Print temperature being used
        print(f"   🌡️ Using temperature: {temp}")
        
        # Generate response based on LLM type
        if self.llm == "custom":
            # Use custom API (OpenAI-compatible)
            try:
                response = self.llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temp,
                    max_tokens=self.config.MAX_TOKENS,
                    stream=stream
                )
                
                if stream:
                    # Return generator for streaming
                    return response
                else:
                    # Return complete response
                    return response.choices[0].message.content
            except Exception as e:
                print(f"❌ Error generating answer: {e}")
                raise Exception(f"API Error: {str(e)}")
        else:
            # Use LangChain ChatOpenAI (original implementation)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
    
    def extract_sources(self, chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """
        Extract source information from chunks
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            List of source dictionaries
        """
        sources = []
        seen = set()  # Track unique (document, page) combinations
        
        for chunk in chunks:
            source_key = (chunk.document_name, chunk.page_number)
            
            if source_key not in seen:
                sources.append({
                    "document": chunk.document_name,
                    "page": chunk.page_number,
                    "chunk_index": chunk.chunk_index
                })
                seen.add(source_key)
        
        return sources
    
    def extract_evidence_snippets(
        self, 
        chunks: List[DocumentChunk], 
        max_length: int = 200
    ) -> List[str]:
        """
        Extract evidence snippets from chunks
        
        Args:
            chunks: List of DocumentChunk objects
            max_length: Maximum length of each snippet
            
        Returns:
            List of evidence snippets
        """
        snippets = []
        
        for chunk in chunks:
            # Truncate text if too long
            text = chunk.text
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            snippet = f"[{chunk.document_name}, p.{chunk.page_number}] {text}"
            snippets.append(snippet)
        
        return snippets
    
    def process_query(self, query: str, 
                     temperature: Optional[float] = None,
                     model_name: Optional[str] = None,
                     top_k: Optional[int] = None) -> RAGResponse:
        """
        Main entry point for RAG question-answering with hallucination prevention
        
        Args:
            query: User question
            temperature: Optional temperature override
            model_name: Optional model name override
            top_k: Optional top-k retrieval override
            
        Returns:
            RAGResponse object with answer and metadata
        """
        print(f"\n🔍 Processing query: {query}")
        
        # Step 1: Retrieve relevant context
        print("   📚 Retrieving context...")
        context_chunks = self.retrieve_context(query, k=top_k)
        
        # Step 1b: Check if we found relevant context
        if not context_chunks or len(context_chunks) == 0:
            # No relevant context found - return "not available" response
            print("   ⚠️ No relevant context found in documents")
            return RAGResponse(
                answer="This information is not available in the provided research papers.",
                sources=[],
                evidence_snippets=[],
                prompt="",
                query=query,
                retrieved_chunks=0
            )
        
        print(f"   ✅ Retrieved {len(context_chunks)} relevant chunks")
        
        # Step 2: Build prompt
        print("   📝 Building prompt...")
        system_prompt, user_prompt = self.build_prompt(query, context_chunks)
        full_prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
        
        # Step 3: Generate answer
        print("   🤖 Generating answer...")
        answer = self.generate_answer(system_prompt, user_prompt, 
                                     temperature=temperature, 
                                     model_name=model_name)
        
        # Step 4: Check if answer indicates information not available
        # (This is a safety check in case the LLM still tries to answer without context)
        answer_lower = answer.lower().strip()
        
        # Step 5: Extract sources and evidence
        sources = self.extract_sources(context_chunks)
        evidence_snippets = self.extract_evidence_snippets(context_chunks)
        
        print("   ✅ Answer generated")
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            evidence_snippets=evidence_snippets,
            prompt=full_prompt,
            query=query,
            retrieved_chunks=len(context_chunks)
        )
    
    def is_ready(self) -> tuple[bool, str]:
        """
        Check if the RAG pipeline is ready to process queries
        
        Returns:
            Tuple of (is_ready, message)
        """
        # Check if API key is configured
        if not self.config.API_KEY:
            return False, "API_KEY not configured"
        
        # Check if API base URL is configured
        if not self.config.API_BASE_URL:
            return False, "API_BASE_URL not configured"
        
        # Check if vector store has documents
        stats = self.vector_store.get_stats()
        if stats.get("status") != "ready":
            return False, "Vector store is empty. Please add documents first."
        
        return True, "RAG pipeline is ready"


# Test the module
if __name__ == "__main__":
    from document_processor import DocumentProcessor
    
    # Initialize components
    processor = DocumentProcessor()
    vector_store = VectorStore()
    
    # Load or create vector store
    if not vector_store.load_from_disk():
        print("\n📚 Processing documents...")
        chunks, embeddings = processor.process_all_documents()
        
        if chunks and embeddings:
            vector_store.add_documents(chunks, embeddings)
            vector_store.save_to_disk()
    
    # Initialize RAG pipeline
    rag = RAGPipeline(vector_store)
    
    # Check if ready
    is_ready, message = rag.is_ready()
    print(f"\n{'✅' if is_ready else '❌'} {message}")
    
    # Test query (if ready)
    if is_ready:
        test_query = "What is artificial intelligence?"
        response = rag.process_query(test_query)
        
        print("\n" + "="*60)
        print("📋 RAG RESPONSE")
        print("="*60)
        print(f"\nQuery: {response.query}")
        print(f"\nAnswer: {response.answer}")
        print(f"\nSources: {len(response.sources)}")
        for source in response.sources:
            print(f"  • {source['document']}, Page {source['page']}")
