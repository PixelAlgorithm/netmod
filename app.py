import os
import json
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END
from langgraph.types import Command

from agents.common import AgentState
from agents.intent_agent import intent_agent, route_intent, clarification_node
from agents.config_agent import config_agent
from agents.validation_agent import validation_agent, route_after_validation
from agents.repair_agent import repair_agent
from agents.deployment_agent import deployment_agent, route_after_deployment
from memory.network_memory import load_network_memory_node, update_network_memory_node
from memory.network_store import build_memory_summary, get_latest_snapshot, init_db
from settings import MissingEnvironmentError, get_device_settings

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ModuleNotFoundError:
    SqliteSaver = None

DB_PATH = "memory/ibn_checkpoints.db"
THREAD_ID = f"ibn-session-{uuid.uuid4().hex[:8]}"


def human_review_node(state: dict) -> dict:
    print("\n[Escalated to human review]")
    reason = state.get("failure_context") or state.get("deployment_result", "unknown")
    print(f"Reason: {reason}")
    state["deployment_result"] = f"escalated to human review: {reason}"
    return state


def query_node(state: dict) -> dict:
    """
    Read-only node — displays current network state from memory.
    Does NOT push any config to the device.
    """
    print("\n[Query — read-only, no config changes]")
    summary = state.get("network_memory_summary", "")
    if summary:
        print("\n-------- Current Network State --------\n")
        print(summary)
    else:
        print("\n[No network memory available yet — run a deployment first]")
    state["deployment_result"] = "query_complete: read-only state displayed"
    return state


def _print_banner(device_host: str, snapshot_time: str | None) -> None:
    width = 60
    print("\n" + "─" * width)
    print(" IBN Orchestration System")
    print(" Autonomous Intent-Based Networking")
    print("─" * width)
    print(f" Connected to: {device_host}")
    if snapshot_time:
        print(f" Last snapshot: {snapshot_time}")
    else:
        print(" Last snapshot: No prior snapshot")
    print()
    print(" What would you like to configure?")
    print(" Examples:")
    print("   • Create VLAN 30 for finance with subnet 192.168.30.0/24")
    print("   • Block traffic from 10.0.20.0/24 to 10.0.30.0/24")
    print("   • Show current network state")
    print("   • Add a static route to 172.20.0.0/16 via 192.168.1.254")
    print()
    print(" Type your intent below:")


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("NetworkMemoryLoad", load_network_memory_node)
    graph.add_node("IntentAgent",    intent_agent)
    graph.add_node("Clarification",  clarification_node)
    graph.add_node("ConfigAgent",    config_agent)
    graph.add_node("ValidationAgent", validation_agent)
    graph.add_node("RepairAgent",    repair_agent)
    graph.add_node("HumanReview",    human_review_node)
    graph.add_node("QueryNode",      query_node)
    graph.add_node("DeploymentAgent", deployment_agent)
    graph.add_node("NetworkMemoryUpdate", update_network_memory_node)

    graph.set_entry_point("NetworkMemoryLoad")
    graph.add_edge("NetworkMemoryLoad", "IntentAgent")

    graph.add_conditional_edges("IntentAgent", route_intent, {
        "clarification": "Clarification",
        "query":         "QueryNode",
        "ready":         "ConfigAgent"
    })

    graph.add_edge("Clarification",  "IntentAgent")
    graph.add_edge("QueryNode", END)
    graph.add_edge("ConfigAgent",    "ValidationAgent")

    graph.add_conditional_edges("ValidationAgent", route_after_validation, {
        "deploy":       "DeploymentAgent",
        "retry":        "RepairAgent",
        "human_review": "HumanReview"
    })

    graph.add_edge("RepairAgent", "ValidationAgent")

    graph.add_conditional_edges("DeploymentAgent", route_after_deployment, {
        "done":             "NetworkMemoryUpdate",
        "repair_and_retry": "RepairAgent",
        "human_review":     "HumanReview"
    })

    graph.add_edge("NetworkMemoryUpdate", END)
    graph.add_edge("HumanReview", END)
    return graph


def compile_app(checkpointer=None):
    return build_graph().compile(checkpointer=checkpointer)


app = compile_app(checkpointer=InMemorySaver())


def _load_initial_memory() -> tuple[dict, str]:
    try:
        device_settings = get_device_settings()
    except MissingEnvironmentError:
        return {}, ""

    latest = get_latest_snapshot(device_settings.host)
    initial_network_memory = {"devices": [latest]} if latest else {}
    initial_memory_summary = build_memory_summary(device_settings.host) if latest else ""
    return initial_network_memory, initial_memory_summary


def _initial_state(user_input: str) -> dict:
    initial_network_memory, initial_memory_summary = _load_initial_memory()
    return {
        'messages':                [HumanMessage(content=user_input)],
        'structured_intent':       {},
        'config':                  "",
        'validation_result':       "",
        'retry_count':             0,
        'failure_context':         "",
        'repair_attempts':         0,
        'deployment_result':       "",
        'network_memory':          initial_network_memory,
        'network_memory_summary':  initial_memory_summary,
        'current_question_index':  0,
        'total_attempts':          0,
    }


def _run_with_interrupts(compiled_app, initial_state: dict, config: dict):
    try:
        result = compiled_app.invoke(initial_state, config=config)
    except RuntimeError as exc:
        print(f"\nStartup failed: {exc}")
        raise SystemExit(1)

    while "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0].value
        if isinstance(interrupt_data, dict):
            q_num = interrupt_data.get("question_number", "?")
            total = interrupt_data.get("total_questions", "?")
            question = interrupt_data.get("question", str(interrupt_data))
            print(f"\n  [{q_num}/{total}] {question}")
        elif isinstance(interrupt_data, str) and interrupt_data.startswith("Device unreachable"):
            print(f"\n  ⚠  {interrupt_data}")
            print("     (r) retry   (c) continue with cached   (a) abort")
        else:
            print(f"\n  {interrupt_data}")
        answer = input("  > ").strip()
        try:
            result = compiled_app.invoke(Command(resume=answer), config=config)
        except RuntimeError as exc:
            print(f"\nStartup failed: {exc}")
            raise SystemExit(1)

    return result


if __name__ == "__main__":
    os.makedirs("memory", exist_ok=True)
    init_db()
    try:
        config = {"configurable": {"thread_id": THREAD_ID}}

        if SqliteSaver is None:
            raise RuntimeError(
                "SqliteSaver is unavailable. Install the 'langgraph-checkpoint-sqlite' package."
            )

        with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
            app = compile_app(checkpointer=checkpointer)
            try:
                device_settings = get_device_settings()
                latest = get_latest_snapshot(device_settings.host)
                _print_banner(
                    device_host=device_settings.host,
                    snapshot_time=latest.get("snapshot_time") if latest else None,
                )
            except MissingEnvironmentError:
                _print_banner(
                    device_host="unknown",
                    snapshot_time=None,
                )
            user_input = input("> ").strip()
            result = _run_with_interrupts(app, _initial_state(user_input), config=config)
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
