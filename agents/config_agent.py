import json

from langchain_core.messages import HumanMessage, SystemMessage
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from agents.common import model
from agents.jinja_filters import register_filters

jinja_env = Environment(
    loader=FileSystemLoader("templates"),
    trim_blocks=True,
    lstrip_blocks=True
)
register_filters(jinja_env)

TEMPLATE_MAP = {
    "create_vlan": "vlan.j2",
    "network_segmentation": "vlan_segmentation.j2",
    "create_acl": "acl.j2",
    "configure_routing": "routin    g.j2",
    "firewall_policy": "firewall.j2",
    "dhcp_configuration": "dhcp.j2",
    "dns_configuration": "dns.j2",
    "mixed_intent": "mixed.j2",
}
CONFIG_VERIFY_PROMPT = """
You are a Configuration Verification Agent for an Autonomous IBN system.

You will be given:
1. The original user request.
2. The structured intent (JSON) extracted from that request.
3. A draft network configuration generated from a Jinja2 template based on that intent.

YOUR TASK:
- Check that the draft configuration correctly and completely implements the structured intent.
- Fix ONLY clear syntax errors (e.g. a missing closing '!', a typo'd keyword).

STRICT RULES — VIOLATING ANY OF THESE IS A FAILURE:
- Do NOT add any line, interface, command, or keyword that is not already present in the draft_config.
- Do NOT invent interface names (e.g. GigabitEthernet0/1) — none was specified in structured_intent.
- Do NOT add 'ip access-group' bindings unless the draft_config already contains one.
- Do NOT replace literal destination/source values (e.g. "internal_servers", "internet") with placeholder keywords like "any" or "log".
- Do NOT add the 'log' keyword to any access-list line.
- If the draft_config has no syntax errors, return it completely unchanged.
- When in doubt, return the draft_config unchanged and set status to "verified".

OUTPUT FORMAT (JSON only, no markdown, no explanations):

{
  "status": "verified" | "corrected" | "invalid",
  "config": "<final corrected or original config as a single string with \\n for newlines>",
  "notes": "<short explanation of any changes made, or 'no changes needed'>"
}

Your response must begin with { and end with }.
"""

def config_agent(state: dict) -> dict:
    intent = state["structured_intent"]["intent"]
    intent_type = intent["intent_type"]

    # 1. Select template
    template_name = TEMPLATE_MAP.get(intent_type, "mixed.j2")
    try:
        template = jinja_env.get_template(template_name)
    except TemplateNotFound:
        state["config"] = ""
        state["validation_result"] = f"error: no template found for intent_type '{intent_type}'"
        return state

    # 2. Render config from intent
    draft_config = template.render(**intent)

    # 3. Verify via LLM
    user_request = next(
        (m.content for m in state["messages"] if isinstance(m, HumanMessage)),
        ""
    )

    verify_messages = [
        SystemMessage(content=CONFIG_VERIFY_PROMPT),
        HumanMessage(content=json.dumps({
            "user_request": user_request,
            "structured_intent": intent,
            "draft_config": draft_config
        }, indent=2))
    ]

    response = model.invoke(verify_messages)
    content = response.content.strip()

    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]

    try:
        verify_result = json.loads(content.strip())
    except json.JSONDecodeError:
        state["config"] = draft_config
        state["validation_result"] = "config_verification_failed_to_parse"
        return state

    if verify_result["status"] == "invalid":
        state["config"] = draft_config
        state["validation_result"] = f"invalid: {verify_result['notes']}"
    else:
        state["config"] = verify_result["config"]
        state["validation_result"] = verify_result["status"]

    return state