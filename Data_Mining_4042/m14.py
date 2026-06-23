import ollama
import fitz  # PyMuPDF
import time
import numpy as np
from pathlib import Path

# ============================================================================
# SETUP: Connect to remote Ollama instance
# ============================================================================
client = ollama.Client(host='http://localhost:11434')
MODEL = 'llama3.1:8b'


# ============================================================================
# UTILITY: PDF text extraction
# ============================================================================
def extract_text_from_pdf(pdf_path):
    """Extract all text from PDF file"""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text


# ============================================================================
# UTILITY: Text chunking with overlap
# ============================================================================
def chunk_text(text, chunk_size=200, overlap=40):
    """
    Split text into overlapping chunks

    Args:
        text: Full document text
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks

    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)

        # Stop if we've reached the end
        if i + chunk_size >= len(words):
            break

    return chunks


# ============================================================================
# UTILITY: Cosine similarity for vector comparison
# ============================================================================
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# ============================================================================
# RAG SYSTEM: Retrieval-Augmented Generation
# ============================================================================
class RAGSystem:
    def __init__(self, model_name='llama3.1:8b'):
        self.model = model_name
        self.chunks = []
        self.embeddings = []

    def load_pdf(self, pdf_path, chunk_size=200, overlap=40):
        """Load PDF, chunk it, and compute embeddings"""
        print(f"\n[RAG] Loading PDF: {pdf_path}")
        start = time.time()

        # Step 1: Extract text
        text = extract_text_from_pdf(pdf_path)
        print(f"  ✓ Extracted {len(text)} characters")

        # Step 2: Chunk text
        self.chunks = chunk_text(text, chunk_size, overlap)
        print(f"  ✓ Created {len(self.chunks)} chunks")

        # Step 3: Compute embeddings for each chunk
        self.embeddings = []
        for i, chunk in enumerate(self.chunks):
            response = client.embeddings(model=self.model, prompt=chunk)
            self.embeddings.append(response['embedding'])

            if (i + 1) % 10 == 0:
                print(f"  → Processed {i + 1}/{len(self.chunks)} chunks")

        elapsed = time.time() - start
        print(f"  ✓ Indexing complete in {elapsed:.2f}s\n")

    def retrieve(self, query, top_k=3):
        """Retrieve top-k most relevant chunks for a query"""
        # Get query embedding
        query_response = client.embeddings(model=self.model, prompt=query)
        query_embedding = query_response['embedding']

        # Calculate similarity with all chunks
        similarities = []
        for chunk_embedding in self.embeddings:
            sim = cosine_similarity(query_embedding, chunk_embedding)
            similarities.append(sim)

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return top chunks
        retrieved_chunks = [self.chunks[i] for i in top_indices]
        return retrieved_chunks

    def ask(self, question):
        """Answer question using RAG pipeline"""
        start = time.time()

        # Step 1: Retrieve relevant chunks
        relevant_chunks = self.retrieve(question, top_k=3)
        context = "\n\n".join(relevant_chunks)

        # Step 2: Generate answer with retrieved context
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""

        response = client.generate(model=self.model, prompt=prompt)
        answer = response['response']

        elapsed = time.time() - start
        return answer, elapsed


# ============================================================================
# CAG SYSTEM: Cache-Augmented Generation
# ============================================================================
class CAGSystem:
    def __init__(self, model_name='llama3.1:8b'):
        self.model = model_name
        self.messages = []

    def load_pdf(self, pdf_path):
        """Load entire PDF into cache as system message"""
        print(f"\n[CAG] Loading PDF: {pdf_path}")
        start = time.time()

        # Extract full text
        text = extract_text_from_pdf(pdf_path)
        print(f"  ✓ Extracted {len(text)} characters")

        # Store as system message - this gets cached by Ollama
        self.messages = [
            {
                'role': 'system',
                'content': f'You have access to this document. Answer questions based only on this document:\n\n{text}'
            }
        ]

        elapsed = time.time() - start
        print(f"  ✓ Document loaded in {elapsed:.2f}s\n")

    def ask(self, question):
        """Answer question using cached document"""
        start = time.time()

        # Append question to conversation
        messages = self.messages.copy()
        messages.append({'role': 'user', 'content': question})

        # Generate answer - KV cache is reused for the document
        response = client.chat(
            model=self.model,
            messages=messages,
            options={'num_ctx': 8192}  # Adjust based on document size
        )

        answer = response['message']['content']

        # Update conversation history
        self.messages.append({'role': 'user', 'content': question})
        self.messages.append({'role': 'assistant', 'content': answer})

        elapsed = time.time() - start
        return answer, elapsed


# ============================================================================
# DEMO: Compare RAG vs CAG
# ============================================================================
def demo_comparison(pdf_path):
    """
    Full demonstration comparing RAG and CAG approaches
    """
    print("=" * 70)
    print("RAG vs CAG Comparison Demo")
    print("=" * 70)

    # Check if PDF exists
    if not Path(pdf_path).exists():
        print(f"Error: PDF file not found at {pdf_path}")
        return

    # ========================================================================
    # Initialize both systems
    # ========================================================================
    rag = RAGSystem(model_name=MODEL)
    cag = CAGSystem(model_name=MODEL)

    # ========================================================================
    # Load PDF into both systems
    # ========================================================================
    print("\n📄 LOADING PHASE")
    print("-" * 70)

    rag.load_pdf(pdf_path, chunk_size=200, overlap=40)
    cag.load_pdf(pdf_path)

    # ========================================================================
    # Test questions
    # ========================================================================
    questions = [
        "What is the main topic of this document?",
        "Summarize the key points in 2-3 sentences.",
        "What are the important technical details mentioned?"
    ]

    print("\n❓ QUESTION ANSWERING PHASE")
    print("-" * 70)

    total_rag_time = 0
    total_cag_time = 0

    for i, question in enumerate(questions, 1):
        print(f"\n🔹 Question {i}: {question}")
        print()

        # RAG answer
        print("[RAG]")
        rag_answer, rag_time = rag.ask(question)
        total_rag_time += rag_time
        print(f"  Answer: {rag_answer[:150]}...")
        print(f"  ⏱️  Time: {rag_time:.2f}s")

        print()

        # CAG answer
        print("[CAG]")
        cag_answer, cag_time = cag.ask(question)
        total_cag_time += cag_time
        print(f"  Answer: {cag_answer[:150]}...")
        print(f"  ⏱️  Time: {cag_time:.2f}s")

        print()
        print(f"  📊 Speed: CAG is {rag_time / cag_time:.1f}x faster for this query")
        print("-" * 70)

    # ========================================================================
    # Final comparison
    # ========================================================================
    print("\n📊 PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Total RAG time:  {total_rag_time:.2f}s")
    print(f"Total CAG time:  {total_cag_time:.2f}s")
    print(f"Speedup:         {total_rag_time / total_cag_time:.2f}x")
    print()
    print("Key differences:")
    print("  • RAG: Embedding + search + retrieval for each query")
    print("  • CAG: Direct inference with cached document context")
    print()
    print("When to use each:")
    print("  • RAG: Large documents (>100 pages), need precise retrieval")
    print("  • CAG: Small-medium docs (<50 pages), many repeated queries")
    print("=" * 70)


# ============================================================================
# RUN THE DEMO
# ============================================================================
if __name__ == "__main__":
    # Replace with your actual PDF path
    PDF_PATH = "mambratrack.pdf"

    demo_comparison(PDF_PATH)
