import ollama
import numpy as np
import fitz
import chromadb

client = ollama.Client(host='http://127.0.0.1:11434')
MODEL = 'nomic-embed-text'
MODEL2 = 'llama3.1:8b'

# Persistent ChromaDB — saves to disk
db = chromadb.PersistentClient(path="./chroma_store")
collection = db.get_or_create_collection("rag_docs")

# --- PDF + chunking (same as before) ---
def load_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text, chunk_size=200, overlap=40):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return chunks

# --- Index: embed and store in ChromaDB ---
def build_index(chunks):
    # Skip if already indexed
    if collection.count() > 0:
        print("Index already exists, skipping embedding.")
        return

    embeddings = [
        client.embeddings(model=MODEL, prompt=chunk)['embedding']
        for chunk in chunks
    ]
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )
    print(f"Indexed {len(chunks)} chunks.")

# --- Retrieve from ChromaDB ---
def retrieve(query, top_k=3):
    query_emb = client.embeddings(model=MODEL, prompt=query)['embedding']
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    return results['documents'][0]  # list of top-k chunks

# --- RAG ---
def rag(query):
    context = "\n---\n".join(retrieve(query))
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer based only on the context above."
    response = client.chat(model=MODEL2, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# --- Run ---
text = load_pdf("mambratrack.pdf")
chunks = chunk_text(text, chunk_size=200, overlap=40)
build_index(chunks)  # only embeds once; skips on re-run

print(rag("how many phases are in mambatrack method?"))
