from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    score: int
    result: str

def check_score(state: State):
    return {}

def pass_node(state: State):
    return {"result": "pass"}

def fail_node(state: State):
    return {"result": "fail"}

def route_fn(state: State):
    return "pass" if state["score"] >= 50 else "fail"

builder = StateGraph(State)
builder.add_node("check_score", check_score)
builder.add_node("pass_node", pass_node)
builder.add_node("fail_node", fail_node)

builder.add_edge(START, "check_score")
builder.add_conditional_edges(
    "check_score",
    route_fn,
    {
        "pass": "pass_node",
        "fail": "fail_node",
    }
)
builder.add_edge("pass_node", END)
builder.add_edge("fail_node", END)

graph = builder.compile()
