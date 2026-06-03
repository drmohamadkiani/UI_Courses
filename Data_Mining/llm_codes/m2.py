# pip install langchain langchain-community langchain-ollama chromadb
# ollama pull nomic-embed-text

# from langchain_ollama import OllamaLLM
#
# llm = OllamaLLM(model="llama3")
from langchain.text_splitter import CharacterTextSplitter

text = """
Python is a programming language created by Guido van Rossum.
It is widely used for AI, web development and automation.
"""

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

docs = splitter.create_documents([text])