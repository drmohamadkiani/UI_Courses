import ollama

# Connect to remote Ollama server
client = ollama.Client(host="http://127.0.0.1:11434")


# Tool 1
def get_weather(city: str):
    return f"The weather in {city} is sunny with light wind."

# Tool 2
def get_temperature(city: str):
    return f"The temperature in {city} is 24°C."


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get info about a city",
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
            "name": "get_temperature",
            "description": "Get info about a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }
]

messages = [
    # {"role": "user", "content": "What is the temperature in Paris?"}
    # {"role": "user", "content": "What is the weather like in Paris?"}
    {"role": "user", "content": "Is it sunny in Paris?"}
    # {"role": "user", "content": "How hot is Paris right now?"}
]

response = client.chat(
    model="llama3.1:8b",
    messages=messages,
    tools=tools
)

print(response["message"])
