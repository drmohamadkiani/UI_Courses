import ollama
import datetime
import ast
import operator
import json

client = ollama.Client(host="http://localhost:11434")

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}

def safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")

    elif isinstance(node, ast.BinOp):
        left = safe_eval(node.left)
        right = safe_eval(node.right)
        op_type = type(node.op)
        if op_type in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[op_type](left, right)
        raise ValueError(f"Operator {op_type} is not allowed")

    elif isinstance(node, ast.UnaryOp):
        operand = safe_eval(node.operand)
        op_type = type(node.op)
        if op_type in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[op_type](operand)
        raise ValueError(f"Unary operator {op_type} is not allowed")

    raise ValueError("Invalid expression")

def calculator(expression: str):
    try:
        parsed = ast.parse(expression, mode="eval")
        result = safe_eval(parsed.body)
        return str(result)
    except Exception as e:
        return f"calculation error: {e}"

def get_time():
    return str(datetime.datetime.now().hour)

tool_map = {
    "get_time": get_time,
    "calculator": calculator,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current local hour as a number from 0 to 23.",
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
            "description": (
                "Evaluate a numeric math expression. "
                "Expression must contain only numbers and operators like + - * / ** % parentheses. "
                "Do not use variable names or function names."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Pure numeric expression like 14**2 or (10+2)*3"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

messages = [
    {
        "role": "system",
        "content": (
            "You are a tool-using assistant. "
            "Use tools when needed. "
            "Do not pretend to call tools in normal text. "
            "Do not write fake tool outputs. "
            "If current hour is needed, call get_time first. "
            "If arithmetic is needed, call calculator with only numeric expressions."
        )
    },
    {
        "role": "user",
        "content": "What is the square of the current hour?"
    }
]

max_steps = 6

for step in range(1, max_steps + 1):
    response = client.chat(
        model="llama3.1:8b",
        messages=messages,
        tools=tools,
        options={"temperature": 0}
    )

    msg = response["message"]

    print(f"\n--- Step {step} ---")
    print("Assistant message:", msg)

    tool_calls = getattr(msg, "tool_calls", None)
    content = getattr(msg, "content", None)

    if not tool_calls:
        # If the model tried to fake a tool call in plain text, stop here
        if content and ("calculator" in content or "get_time" in content):
            print("\nModel produced text that looks like a fake tool call.")
            print("Stop this run and restart fresh.")
        else:
            print("\nFinal answer:")
            print(content)
        break

    # append assistant message
    messages.append({
        "role": "assistant",
        "content": content or "",
        "tool_calls": [
            {
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            }
            for tc in tool_calls
        ]
    })

    for tc in tool_calls:
        name = tc.function.name
        args = tc.function.arguments

        if isinstance(args, str):
            args = json.loads(args)

        print("Calling tool:", name)
        print("Arguments:", args)

        if name not in tool_map:
            result = f"error: unknown tool '{name}'"
        else:
            try:
                result = tool_map[name](**args)
            except Exception as e:
                result = f"error while executing tool '{name}': {e}"

        print("Tool result:", result)

        messages.append({
            "role": "tool",
            "name": name,
            "content": result
        })
