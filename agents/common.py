from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    structured_intent: dict
    config: str
    validation_result: str
    retry_count: int
    failure_context: str
    repair_attempts: int  
 

model = ChatBedrock(
    model="amazon.nova-pro-v1:0"
   )