from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = OllamaLLM(model="llama3.1:8b", base_url="http://localhost:11434")
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic}"
)

# Without StrOutputParser
chain_without = prompt | llm
result_without = chain_without.invoke({"topic": "AI"})
print(f"Type without parser: {type(result_without)}")
print(f"Content: {result_without}\n")

# With StrOutputParser
chain_with = prompt | llm | StrOutputParser()
result_with = chain_with.invoke({"topic": "AI"})
print(f"Type with parser: {type(result_with)}")
print(f"Content: {result_with}")
