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

response = client.chat.completions.create(
    model="openai/gpt-oss-120b:free",
    messages=[{"role": "user", "content": "Hello world"}],
    extra_body={"reasoning": {"enabled": True}},
)

print(response.choices[0].message.content)
