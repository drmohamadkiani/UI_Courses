import ollama

# connect to remote ollama server
client = ollama.Client(host="http://127.0.0.1:11434")


# local python function that will be called by the model
def get_weather(city: str):
    return f"The weather in {city} is 25°C and sunny."


# tool schema that the model can use
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name of the city"
                    }
                },
                "required": ["city"]
            }
        }
    }
]


# conversation messages
messages = [
    {
        "role": "user",
        "content": "What is the weather in Paris?"
    }
]


# send request to the remote model
response = client.chat(
    model="llama3.1:8b",
    messages=messages,
    tools=tools
)

msg = response["message"]
print(msg)
# check if the model requested a tool
if msg.get("tool_calls"):

    tool_call = msg["tool_calls"][0]

    function_name = tool_call["function"]["name"]
    arguments = tool_call["function"]["arguments"]

    # execute the python function locally
    if function_name == "get_weather":
        result = get_weather(**arguments)
        #my result
        print('my result',result)
