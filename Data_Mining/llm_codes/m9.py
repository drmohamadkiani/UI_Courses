import ollama
import numpy as np

client = ollama.Client(host='http://127.0.0.1:11434')
MODEL = 'llama3.1:8b'

# --- Knowledge Base ---
documents = [
    "RAG stands for Retrieval-Augmented Generation.",
    "In RAG, relevant information is retrieved first, then passed to the model.",
    "ollama is a tool for running language models locally.",
    "Embeddings are vector representations of text used for semantic search.",
    "Cosine similarity measures the angle between two vectors.",
]

# --- Embed all documents at startup ---
doc_embeddings = [
    client.embeddings(model=MODEL, prompt=doc)['embedding']
    for doc in documents
]

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, top_k=2):
    global doc_embeddings
    query_emb = client.embeddings(model=MODEL, prompt=query)['embedding']
    scores = [cosine_similarity(query_emb, emb) for emb in doc_embeddings]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [documents[i] for i in top_indices]

def rag(query):
    context = "\n".join(retrieve(query))
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    response = client.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

print(rag("What is RAG?"))
