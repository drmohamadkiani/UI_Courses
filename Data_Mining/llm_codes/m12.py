import ollama
import chromadb

MODEL = "llama3.1:8b"
client = ollama.Client(host='http://127.0.0.1:11434')

# Connect to existing DB - no PDF loading needed
db = chromadb.PersistentClient(path="./chroma_store")
collection = db.get_or_create_collection("rag_docs")


def retrieve(query, top_k=3):
    emb = client.embeddings(model=MODEL, prompt=query).embedding
    results = collection.query(query_embeddings=[emb], n_results=top_k)
    return results['documents'][0]


def rewrite_query(question, chat_history):
    if not chat_history:
        return question
    history_text = "\n".join(f"{role}: {msg}" for role, msg in chat_history)
    prompt = (
        f"Given this conversation:\n{history_text}\n\n"
        f"Rewrite this follow-up as a standalone question: '{question}'\n"
        "Return only the rewritten question."
    )
    return client.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])['message']['content'].strip()


def ask(question, chat_history):
    standalone_q = rewrite_query(question, chat_history)
    context = "\n---\n".join(retrieve(standalone_q))

    messages = [{"role": "system", "content": f"Answer based only on the context below.\n\nContext:\n{context}"}]
    for role, msg in chat_history:
        messages.append({"role": role, "content": msg})
    messages.append({"role": "user", "content": question})

    answer = client.chat(model=MODEL, messages=messages)['message']['content']
    chat_history.append(("user", question))
    chat_history.append(("assistant", answer))
    return answer


chat_history = []
print(f"DB has {collection.count()} chunks. Ready.\n")

while True:
    question = input("You: ").strip()
    if question.lower() == "exit":
        break
    print(f"AI: {ask(question, chat_history)}\n")
