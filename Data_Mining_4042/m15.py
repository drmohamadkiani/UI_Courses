from langchain_ollama import OllamaLLM

# Initialize the Ollama LLM with a specific model
llm = OllamaLLM(
    model="llama3.1:8b",  # Change to your preferred model
    base_url="http://localhost:11434"  # Default Ollama URL
)

# Ask a simple question
response = llm.invoke("What is the capital of France?")
print(response)
