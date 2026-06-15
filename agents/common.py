from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    structured_intent: dict
    config: str
    validation_result: str

model = ChatOpenAI(
    model="qwen/qwen3-8b",
    base_url="http://127.0.0.1:1234/v1",
    api_key="sk-lm-EJjPzE1o:6W4624mlXHx6NbrnC10r"
)