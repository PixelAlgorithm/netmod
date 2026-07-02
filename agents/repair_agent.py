"""
agents/repair_agent.py

Sits between ValidationAgent (on failure) and the retry loop back to
ValidationAgent. Rather than re-rendering the whole config from scratch
(which ConfigAgent would do, producing the same broken output again),
RepairAgent receives the exact failure_context from validation and
surgically patches the config text.

Flow:
    ValidationAgent (fail) → RepairAgent → ValidationAgent (re-check)

RepairAgent does NOT re-render from the Jinja template. It takes the
already-rendered config and fixes only what validation said was wrong.
This means:
  - Syntax failures (missing '!', unrecognized command) → LLM patches
    the specific broken lines
  - Batfish parse warnings → LLM rewrites the flagged lines to valid
    Cisco IOS syntax
  - Semantic failures → LLM adds missing objects/fields that intent
    required but config didn't contain

MAX_REPAIR_ATTEMPTS caps how many times RepairAgent can try before
giving up and letting the escalation path take over.
"""

import json
from langchain_core.messages import HumanMessage, SystemMessage
from agents.common import MAX_TOTAL_ATTEMPTS, invoke_model

REPAIR_PROMPT = """
You are a Network Configuration Repair Agent for an Autonomous IBN system.

You will be given:
1. A network configuration (Cisco IOS CLI) that failed validation.
2. The exact validation errors or Batfish parse warnings that caused the failure.
3. The structured intent (JSON) that the config is supposed to implement.
4. The current network memory summary.

YOUR TASK:
Fix ONLY the specific issues identified in the failure_context.
Return the repaired config as a valid Cisco IOS configuration.

STRICT RULES:
- Fix ONLY what failure_context identifies. Do not rewrite unrelated sections.
- Do NOT add interfaces, commands, or objects not present in structured_intent.
- Do NOT invent IP addresses, VLAN IDs, or subnets not in structured_intent.
- Do NOT add the 'log' keyword to ACL lines.
- ACL permit/deny lines use wildcard masks (e.g. 0.0.0.255), NOT CIDR (e.g. /24).
- Interface IP addresses use dotted-decimal subnet masks (e.g. 255.255.255.0).
- Every config must start with a hostname line and end with 'end'.
- If a Batfish warning says a line is unrecognized, remove or correct that specific line.
- If a semantic error says an object is missing from the config, add the minimum
  lines needed to represent it based on structured_intent.
- When in doubt about a fix, remove the offending line rather than guessing.

COMMON FIXES:
- "No hostname set" → add: hostname <deployment_target | default NETWORK-DEVICE>
- "ip default-gateway inside interface block" → move it outside or remove it
- "unrecognized syntax" on an ACL line → check wildcard mask format
- "access-list X permit ip any internet" → replace "internet" with "any"
- "access-list X deny ip any internal_servers" → replace label with actual CIDR from intent

DEVICE REJECTION FIXES (when failure_context contains "device rejected lines"):
- "% Invalid input detected" on a vlan line → device may be a router not a switch;
  try removing vlan database commands or use 'vlan <id>' under interface config
- "% Invalid input detected" on an interface vlan line → L3 interfaces may not be
  supported; check if 'ip routing' is enabled
- "% Invalid input detected" on a 'version X.X' line → remove the version line,
  it is informational only and not a valid config command on all platforms
- "% Invalid input detected" on 'hostname' line → should never happen; if it does,
  remove the hostname line and try without it
- General rule: if a whole section (vlan block, interface block) is rejected,
  remove that section entirely and return the minimal working subset

OUTPUT FORMAT (JSON only, no markdown, no explanations):

{
  "status": "repaired" | "unchanged",
  "config": "<full repaired config as a single string with \\n for newlines>",
  "notes": "<one line: what was fixed, or 'no fix possible'>"
}

- "repaired": you made changes, return the fixed config.
- "unchanged": you could not determine a safe fix; return original config unchanged.

Your response must begin with { and end with }.
"""


def repair_agent(state: dict) -> dict:
    """
    Attempts to surgically fix the config based on failure_context.
    Updates state['config'] with the repaired version and increments
    retry_count. Does NOT reset validation_result — ValidationAgent
    will overwrite it on the next pass.
    """
    total_attempts = state.get("total_attempts", 0)
    repair_attempts = state.get("repair_attempts", 0)

    if total_attempts >= MAX_TOTAL_ATTEMPTS:
        print(f"  [repair_agent] max total attempts ({MAX_TOTAL_ATTEMPTS}) reached, passing to escalation")
        state["validation_result"] = (
            f"invalid (escalated after repairs): {state.get('failure_context', 'unknown')}"
        )
        return state

    total_attempts += 1
    state["total_attempts"] = total_attempts

    failure_context = state.get("failure_context", "")
    broken_config = state.get("config", "")
    intent = state["structured_intent"].get("intent", {})

    print(f"  [repair_agent] attempt {total_attempts}/{MAX_TOTAL_ATTEMPTS}")
    print(f"  [repair_agent] fixing: {failure_context[:120]}{'...' if len(failure_context) > 120 else ''}")

    messages = [
        SystemMessage(content=REPAIR_PROMPT),
        HumanMessage(content=json.dumps({
            "broken_config": broken_config,
            "failure_context": failure_context,
            "structured_intent": intent,
            "network_memory_summary": state.get("network_memory_summary", ""),
        }, indent=2))
    ]

    response = invoke_model(messages)
    content = response.content.strip()

    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]

    try:
        result = json.loads(content.strip())
    except json.JSONDecodeError:
        print("  [repair_agent] failed to parse LLM response, keeping original config")
        state["repair_attempts"] = repair_attempts + 1
        return state

    if result.get("status") == "repaired":
        state["config"] = result["config"]
        print(f"  [repair_agent] repaired: {result.get('notes', '')}")
    else:
        print(f"  [repair_agent] no fix applied: {result.get('notes', '')}")

    state["repair_attempts"] = repair_attempts + 1
    return state


def route_after_repair(state: dict) -> str:
    """
    After repair, always go back to ValidationAgent for a fresh check.
    If repair_agent hit its own max and force-escalated, route_after_validation
    in validation_agent.py will catch the 'escalated' string and route to
    human_review on the next ValidationAgent pass.
    """
    return "revalidate"
