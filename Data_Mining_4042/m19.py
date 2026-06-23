from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

llm = OllamaLLM(model="llama3.1:8b", base_url="http://localhost:11434")

# First chain: Generate a topic
topic_prompt = PromptTemplate(
    input_variables=["subject"],
    template="Suggest one interesting subtopic about {subject}. Only output the subtopic name."
)

# Second chain: Explain the topic
explain_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain this topic briefly: {topic}"
)

# Create chains
topic_chain = topic_prompt | llm | StrOutputParser()
explain_chain = explain_prompt | llm | StrOutputParser()

# Execute sequentially
subject = "artificial intelligence"
subtopic = topic_chain.invoke({"subject": subject})
print(f"Generated subtopic: {subtopic}\n")

explanation = explain_chain.invoke({"topic": subtopic})
print(f"Explanation:\n{explanation}")
