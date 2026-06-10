import ollama
import numpy as np
import fitz  # pymupdf  #pdfplumber   #LlamaParse

client = ollama.Client(host='http://127.0.0.1:11434')
MODEL = 'llama3.1:8b'

# --- Step 1: Extract text from PDF ---
def load_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

# --- Step 2: Chunk text ---
def chunk_text(text, chunk_size=200, overlap=40): #chunk=10%-20%
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# --- Step 3: Embed all chunks ---
def build_index(chunks):
    return [
        client.embeddings(model=MODEL, prompt=chunk)['embedding']
        for chunk in chunks
    ]

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- Step 4: Retrieve top-k relevant chunks ---
def retrieve(query, chunks, embeddings, top_k=3):
    query_emb = client.embeddings(model=MODEL, prompt=query)['embedding']
    scores = [cosine_similarity(query_emb, emb) for emb in embeddings]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

# --- Step 5: RAG pipeline ---
def rag(query, chunks, embeddings):
    context = "\n---\n".join(retrieve(query, chunks, embeddings))
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer based only on the context above."
    response = client.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# --- Run ---
text = load_pdf("mambratrack.pdf")
chunks = chunk_text(text, chunk_size=200, overlap=40)
embeddings = build_index(chunks)

print(f"Total chunks: {len(chunks)}")
# print(rag("what is the result of mota of mambatrack", chunks, embeddings))
# print(rag("what is the result of mota of mambatrack", chunks, embeddings))
print(rag("what is IDF1 score of mambatrack?", chunks, embeddings))
