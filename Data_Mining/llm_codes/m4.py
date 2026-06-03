import ollama
import math
import datetime

# connect to remote ollama server
client = ollama.Client(host="http://127.0.0.1:11434")


# -------------------------
# tools implemented in python
# -------------------------

def calculator(expression: str):
    """Evaluate a mathematical expression"""
    try:
        return str(eval(expression))
    except Exception:
        return "Error evaluating expression"


def get_weather(city: str):
    """Dummy weather function"""
    return f"The weather in {city} is 22°C and partly cloudy."


def get_time():
    """Return current system time"""
    return str(datetime.datetime.now())


def wiki_summary(topic: str):
    """Fake wikipedia summary (for demo)"""
    data = {
        "python": "Python is a programming language created by Guido van Rossum.",
        "ollama": "Ollama allows running large language models locally.",
        "transformer": "Transformers are neural networks used in modern LLMs."
    }
    return data.get(topic.lower(), "No information found.")


# -------------------------
# tool schema for the LLM
# -------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current system time",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_summary",
            "description": "Get short explanation of a topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"}
                },
                "required": ["topic"]
            }
        }
    }
]


# map tool names to python functions
tool_map = {
    "calculator": calculator,
    "get_weather": get_weather,
    "get_time": get_time,
    "wiki_summary": wiki_summary
}


messages = []

while True:
    user_input = input("User: ")

    if user_input == "exit":
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat(
        model="llama3.1:8b",
        messages=messages,
        tools=tools
    )

    msg = response["message"]
    print('base result',msg)

    # if the model decides to call a tool
    if msg.get("tool_calls"):

        tool_call = msg["tool_calls"][0]

        name = tool_call["function"]["name"]
        args = tool_call["function"]["arguments"]

        print(f"[Model wants to call tool: {name}]")

        # execute tool locally
        result = tool_map[name](**args)
        print('result', result)


