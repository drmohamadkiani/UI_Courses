

#Local Test----------------------------
#import ollama

# response = ollama.generate(
#     model='llama3',
#     prompt='hello'
# )


#On network Test-----------------------------
# import ollama
# client = ollama.Client(host='http://127.0.0.1:11434')
#
# print(client.list())
#
# response = client.generate(
#     model="llama3.1:8b",
#     prompt="explain transformers simply"
# )
#
# print(response['response'])


# #Parameteres--------------------
# import ollama
# client = ollama.Client(host='http://127.0.0.1:11434')
#
# # print(client.list())
#
# response = client.generate(
#     model="gemma3:1b",
#     prompt='Explain about ai',
#     system='You are a medical assistant.',
#     stream=True,
#     options={
#         "temperature": 0.7,
#         "top_p": 0.9,
#         "top_k": 40,
#         "num_predict": 200,
#         "stop": ["\nUser:"],
#         "repeat_penalty": 1.1
#     }
# )
# for chunk in response:
#     print(chunk)
# print(response['response'])

#Parameteres--------------------
import ollama
client = ollama.Client(host='http://127.0.0.1:11434')
response = client.generate(
    model="gemma3:1b",
    prompt='Explain about ai',
    system='You are a medical assistant.',
    stream=False,
    options={
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 40,
        "num_predict": 100,
        "stop": ["\nUser:"],
        "repeat_penalty": 0
    }
)
print(response['response'])
print("Input tokens  :", response["prompt_eval_count"])
print("Output tokens :", response["eval_count"])
print("total_duration :", response["total_duration"])
print("eval_duration  :", response["eval_duration"])




# from transformers import AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
#
# text = "Explain neural networks simply."
#
# tokens = tokenizer.tokenize(text)
# ids = tokenizer.encode(text)
#
# print("Tokens:", tokens)
# print("Token IDs:", ids)
# print("Token count:", len(ids))
