from document_processor import DocumentProcessor
from vector_store import VectorStore

print("\n" + "="*60)
print("  PROCESSING DOCUMENTS")
print("="*60 + "\n")

# Initialize
print("Step 1: Initializing document processor...")
processor = DocumentProcessor()

print("\nStep 2: Processing PDFs...")
chunks, embeddings = processor.process_all_documents()

if not chunks or not embeddings:
    print("\n❌ Failed to process documents!")
    exit(1)

print(f"\n✅ Successfully created {len(chunks)} chunks with {len(embeddings)} embeddings")

# Create vector store
print("\nStep 3: Creating vector store...")
vector_store = VectorStore()
vector_store.add_documents(chunks, embeddings)

# Save to disk
print("\nStep 4: Saving to disk...")
vector_store.save_to_disk()

print("\n" + "="*60)
print("  ✅ PROCESSING COMPLETE!")
print("="*60)
print("\nYou can now:")
print("1. Refresh your Streamlit app (Ctrl+R in browser)")
print("2. Start asking questions!")
print("\n")
