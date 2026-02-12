import os
from dotenv import load_dotenv
from langgraph.graph.state import END, START, RunnableConfig, StateGraph
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.mongodb import MongoDBSaver

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set")


llm = init_chat_model(
    model="gemini-2.5-flash", model_provider="google_genai", api_key=api_key
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    response = llm.invoke(state.get("messages"))
    return {"messages": [response]}


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


def compile_graph_with_checkpointer(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)


with MongoDBSaver.from_conn_string(
    "mongodb://admin:admin@localhost:27017"
) as checkpointer:
    graph_with_checkpointer = compile_graph_with_checkpointer(checkpointer=checkpointer)

    config: RunnableConfig = {
        "configurable": {"thread_id": "1"}
    }  # for different thread id the context/checkpoints will be different

    for chunk in graph_with_checkpointer.stream(
        State({"messages": ["what is my name?"]}), config=config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
