from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

llm = OllamaLLM(model="llama3.1:8b", base_url="http://localhost:11434")

prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer this question: {question}"
)

chain = prompt | llm

# First question
response1 = chain.invoke({"question": "My name is Ali. Remember it."})
print(f"Response 1: {response1}\n")

# Second question
response2 = chain.invoke({"question": "What is my name?"})
print(f"Response 2: {response2}")
