import json

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from agents.common import AgentState
from agents.intent_agent import intent_agent, route_intent, clarification_node
from agents.config_agent import config_agent
from agents.validation_agent import validation_agent, route_after_validation
from agents.repair_agent import repair_agent
from agents.deployment_agent import deployment_agent, route_after_deployment
from memory.network_memory import load_network_memory_node, update_network_memory_node


def human_review_node(state: dict) -> dict:
    print("\n[Escalated to human review]")
    reason = state.get("failure_context") or state.get("deployment_result", "unknown")
    print(f"Reason: {reason}")
    state["deployment_result"] = f"escalated to human review: {reason}"
    return state


graph = StateGraph(AgentState)

graph.add_node("NetworkMemoryLoad", load_network_memory_node)
graph.add_node("IntentAgent",    intent_agent)
graph.add_node("Clarification",  clarification_node)
graph.add_node("ConfigAgent",    config_agent)
graph.add_node("ValidationAgent", validation_agent)
graph.add_node("RepairAgent",    repair_agent)
graph.add_node("HumanReview",    human_review_node)
graph.add_node("DeploymentAgent", deployment_agent)
graph.add_node("NetworkMemoryUpdate", update_network_memory_node)

graph.set_entry_point("NetworkMemoryLoad")
graph.add_edge("NetworkMemoryLoad", "IntentAgent")

graph.add_conditional_edges("IntentAgent", route_intent, {
    "clarification": "Clarification",
    "ready":         "ConfigAgent"
})

graph.add_edge("Clarification",  "IntentAgent")
graph.add_edge("ConfigAgent",    "ValidationAgent")

graph.add_conditional_edges("ValidationAgent", route_after_validation, {
    "deploy":       "DeploymentAgent",
    "retry":        "RepairAgent",
    "human_review": "HumanReview"
})

graph.add_edge("RepairAgent", "ValidationAgent")

graph.add_conditional_edges("DeploymentAgent", route_after_deployment, {
    "done":         "NetworkMemoryUpdate",
    "human_review": "HumanReview"
})

graph.add_edge("NetworkMemoryUpdate", END)
graph.add_edge("HumanReview", END)

app = graph.compile()


if __name__ == "__main__":
    try:
        result = app.invoke({
            'messages':          [HumanMessage(content=input("what u like to create : "))],
            'structured_intent': {},
            'config':            "",
            'validation_result': "",
            'retry_count':       0,
            'failure_context':   "",
            'repair_attempts':   0,
            'deployment_result': "",
            'network_memory':    {},
            'network_memory_summary': "",
        })
    except RuntimeError as exc:
        print(f"\nStartup failed: {exc}")
        raise SystemExit(1)

    print("\n-------- Intent --------\n")
    print(json.dumps(result['structured_intent'], indent=2))

    print("\n-------- Generated Config --------\n")
    print(result['config'])

    print("\n-------- Validation Result --------\n")
    print(result['validation_result'])

    print("\n-------- Deployment Result --------\n")
    print(result.get('deployment_result', '(not reached)'))

    print("\n-------- Network Memory Summary --------\n")
    print(result.get('network_memory_summary', '(not available)'))

    if result.get('failure_context'):
        print("\n-------- Failure Context --------\n")
        print(result['failure_context'])
