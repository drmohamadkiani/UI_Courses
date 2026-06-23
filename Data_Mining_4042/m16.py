from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Initialize the model
llm = OllamaLLM(
    model="llama3.1:8b",
    base_url="http://localhost:11434"
)

# Create a prompt template
template = """You are a helpful assistant. Answer the following question:

Question: {question}

Answer:"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template
)

# Format and invoke
formatted_prompt = prompt.format(question="What is Python?")
response = llm.invoke(formatted_prompt)
print(response)
