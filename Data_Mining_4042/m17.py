from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

# Initialize model
llm = OllamaLLM(
    model="llama3.1:8b",
    base_url="http://localhost:11434"
)

# Create prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms for beginners."
)

# Create output parser
output_parser = StrOutputParser()

# Create chain using LCEL (LangChain Expression Language)
chain = prompt | llm | output_parser

# Invoke the chain
response = chain.invoke({"topic": "machine learning"})
print(response)
