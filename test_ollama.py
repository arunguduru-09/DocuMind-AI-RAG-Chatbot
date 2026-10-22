
from ollama import chat

response = chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": "What is RAG? Explain in two sentences."
        }
    ]
)

print(response.message.content)
