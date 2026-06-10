#Multiple Tool Calls / Agent Loop
import ollama
import datetime

# connect to remote ollama server
client = ollama.Client(host="http://127.0.0.1:11434")


# -------------------------
# tools implemented in python
# -------------------------

def get_time():
    """Return current hour"""
    return str(datetime.datetime.now().hour)


def calculator(expression: str):
    """Evaluate a math expression"""
    try:
        return str(eval(expression))
    except Exception:
        return "calculation error"


# map tool names to functions
tool_map = {
    "get_time": get_time,
    "calculator": calculator
}


# tool schema for the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current hour of the day (0-23)",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression like 5*5 or 12+3"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


messages = [
    {
        "role": "user",
        "content": "What is the square of the current hour? think step by step Never mix function names or variables (like 'get_time') into the calculator expression."
    }
]


# -------------------------
# agent loop
# -------------------------

while True:

    response = client.chat(
        model="llama3.1:8b",
        messages=messages,
        tools=tools
    )

    msg = response["message"]

    # if model wants to call a tool
    if msg.get("tool_calls"):

        tool_call = msg["tool_calls"][0]

        name = tool_call["function"]["name"]
        args = tool_call["function"]["arguments"]

        print("Model requested tool:", name)
        print("Arguments:", args)

        # execute the tool
        result = tool_map[name](**args)

        print("Tool result:", result)

        # add tool request to conversation
        messages.append(msg)

        # send tool result back to model
        messages.append({
            "role": "tool",
            "name": name,
            "content": result
        })

    else:
        # model produced final answer
        print("\nFinal answer:")
        print(msg["content"])
        break
