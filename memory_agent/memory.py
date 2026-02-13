import json
from mem0 import Memory
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY not set")

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=gemini_api_key,
)

config = {
    "version": "v1.1",
    "embedder": {
        # currently mem0 only works with openai embeddings
        "provider": "google",
        "config": {
            "api_key": gemini_api_key,
            "model": "gemini_embed_text_1.0",
        },
    },
    "llm": {
        "provider": "google",
        "config": {
            "api_key": gemini_api_key,
            "model": "gemini/2.5-flash",
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {"host": "localhost", "port": 6333},
    },
}

memory_client = Memory.from_config(config)

user_query = input("> ")

search_memory = memory_client.search(query=user_query, user_id="vinitkumar")

memories = [
    f"ID: {mem.get('id')}\nMemory: {mem.get('memory')}"
    for mem in search_memory.get("results", [])
]

print("Found memories:", memories)

SYSTEM_PROMPT = f"""
Here is the context about the user:
{json.dumps(memories)}
"""

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ],
)

ai_response = response.choices[0].message.content

print("AI:", ai_response)

memory_client.add(
    user_id="vinitkumar",
    messages=[
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": ai_response},
    ],
)

print("Memory has been saved..")

