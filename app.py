import json

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from agents.common import AgentState
from agents.intent_agent import intent_agent, route_intent, clarification_node
from agents.config_agent import config_agent

graph = StateGraph(AgentState)

graph.add_node("IntentAgent", intent_agent)
graph.add_node("Clarification", clarification_node)
graph.add_node("ConfigAgent", config_agent)

graph.set_entry_point("IntentAgent")

graph.add_conditional_edges("IntentAgent", route_intent, {
    "clarification": "Clarification",
    "ready": "ConfigAgent"
})

graph.add_edge("Clarification", "IntentAgent")
graph.add_edge("ConfigAgent", END)

app = graph.compile()

result = app.invoke({
    'messages': [HumanMessage(content=input("what u like to create : "))]
})

print("\n-------- Intent --------\n")
print(json.dumps(result['structured_intent'], indent=2))

print("\n-------- Generated Config --------\n")
print(result['config'])

print("\n-------- Verification Result --------\n")
print(result['validation_result'])