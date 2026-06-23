from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    text: str
    upper_text: str

def to_upper(state: State):
    return {"upper_text": state["text"].upper()}

graph_builder = StateGraph(State)
graph_builder.add_node("to_upper", to_upper)
graph_builder.add_edge(START, "to_upper")
graph_builder.add_edge("to_upper", END)

graph = graph_builder.compile()

result = graph.invoke({"text": "hello"})
print(result)
