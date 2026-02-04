from openai.types.chat import ChatCompletionMessageParam
from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

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

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_rag",
    embedding=embedding_model,
)

user_query = input("Ask something: ")

search_result = vector_db.similarity_search(query=user_query, k=3)

context = "\n\n\n".join(
    [
        f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}"
        for result in search_result
    ]
)

SYSTEM_PROMPT = f"""
You are a helpfull AI assistant who answers user queries based on the available context returived from a PDF file along with page_contents and page number.

You should only answer the user based on the following context and navigate the user to open the right page number to know more.

Context:
{context}
"""

message_history: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

response = client.chat.completions.create(
    model="openai/gpt-oss-120b:free",
    messages=message_history,
)

print(f"{response.choices[0].message.content}")
