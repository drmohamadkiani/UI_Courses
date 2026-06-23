from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage

# 1. Define the "State" (this replaces ConversationBufferMemory)
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. Setup the LLM
llm = OllamaLLM(model="llama3.1:8b", base_url="http://localhost:11434")

# 3. Define the node (the function that calls the LLM)
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# 4. Construct the Graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# --- Execution ---

# First interaction
state1 = {"messages": [HumanMessage(content="My name is Ali. Remember it.")]}
response1 = graph.invoke(state1)
print(f"Response 1: {response1['messages'][-1].content}\n")

# Second interaction
# We pass the history from the previous step back into the graph
state2 = {"messages": response1["messages"] + [HumanMessage(content="What is my name?")]}
response2 = graph.invoke(state2)
print(f"Response 2: {response2['messages'][-1].content}\n")

# Check memory content
print("Memory history:")
for msg in response2["messages"]:
    print(f"{msg.type}: {msg.content}")
