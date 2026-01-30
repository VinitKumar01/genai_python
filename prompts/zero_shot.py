from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    default_headers={
        "HTTP-Referer": "http://127.0.0.1",
        "X-Title": "hello-world-test",
    },
)

# Zero-Shot prompts are the prompts that are directly given to model without any prior examples so the model relies on its general knowledge to figure out the response.
SYSTEM_PROMPT = "You are an ai model developed by Vinit Kashwan. You are an expert in maths and only allowed to answer questions related to maths only, if any user asks for any other question then just say sorry i can only help you with maths."

response = client.chat.completions.create(
    model="openai/gpt-oss-120b:free",
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {"role": "user", "content": "What is the integration of log(x)dx"},
    ],
    extra_body={"reasoning": {"enabled": True}},
)

print(response.choices[0].message.content)
