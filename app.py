import json

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from agents.common import AgentState
from agents.intent_agent import intent_agent, route_intent, clarification_node
from agents.config_agent import config_agent
from agents.validation_agent import validation_agent, route_after_validation, MAX_RETRIES
from agents.repair_agent import repair_agent


def human_review_node(state: dict) -> dict:
    print("\n[Escalated to human review]")
    print(f"Reason: {state.get('failure_context', 'unknown')}")
    return state


def deploy_node(state: dict) -> dict:
    print("\n[Deploying config — placeholder, wire up Ansible/Nornir here]")
    return state


graph = StateGraph(AgentState)

graph.add_node("IntentAgent", intent_agent)
graph.add_node("Clarification", clarification_node)
graph.add_node("ConfigAgent", config_agent)
graph.add_node("ValidationAgent", validation_agent)
graph.add_node("RepairAgent", repair_agent)
graph.add_node("HumanReview", human_review_node)
graph.add_node("Deploy", deploy_node)

graph.set_entry_point("IntentAgent")

graph.add_conditional_edges("IntentAgent", route_intent, {
    "clarification": "Clarification",
    "ready": "ConfigAgent"
})

graph.add_edge("Clarification", "IntentAgent")
graph.add_edge("ConfigAgent", "ValidationAgent")

graph.add_conditional_edges("ValidationAgent", route_after_validation, {
    "deploy": "Deploy",
    "retry": "RepairAgent",
    "human_review": "HumanReview"
})

graph.add_edge("RepairAgent", "ValidationAgent")

graph.add_edge("Deploy", END)
graph.add_edge("HumanReview", END)

app = graph.compile()

result = app.invoke({
    'messages': [HumanMessage(content=input("what u like to create : "))],
    'structured_intent': {},
    'config': "",
    'validation_result': "",
    'retry_count': 0,
    'failure_context': "",
    'repair_attempts': 0,
})

print("\n-------- Intent --------\n")
print(json.dumps(result['structured_intent'], indent=2))

print("\n-------- Generated Config --------\n")
print(result['config'])

print("\n-------- Validation Result --------\n")
print(result['validation_result'])

if result.get('failure_context'):
    print("\n-------- Failure Context --------\n")
    print(result['failure_context'])