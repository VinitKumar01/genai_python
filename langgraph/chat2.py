import os
from dotenv import load_dotenv
from langgraph.graph.state import END, START, StateGraph
from typing_extensions import TypedDict
from typing import Literal, NotRequired, Optional, cast
from langchain.chat_models import init_chat_model

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set")


llm = init_chat_model(
    model="gemini-2.5-flash", model_provider="google_genai", api_key=api_key
)


class State(TypedDict):
    user_query: str
    llm_response: NotRequired[Optional[str]]
    is_good: NotRequired[Optional[bool]]


def chatbot(state: State):
    response = llm.invoke(state.get("user_query"))
    state["llm_response"] = cast(str, response.content)

    return state


def evaluate_response(state: State) -> Literal["chatbot", "end_node"]:
    # evaluate response here using another llm model
    if True:
        return "end_node"

    return "chatbot"


def end_node(state: State):
    return state


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("evaluate_response", evaluate_response)
graph_builder.add_node("end_node", end_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", evaluate_response)

graph_builder.add_edge("chatbot", "end_node")
graph_builder.add_edge("end_node", END)

graph = graph_builder.compile()

updated_state = graph.invoke({"user_query": "What is 2 + 2?"})
print("Updated state", updated_state)
