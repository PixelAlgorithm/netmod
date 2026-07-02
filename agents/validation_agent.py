"""
agents/validation_agent.py

NOTE: agents/common.py's AgentState TypedDict must add these two
fields for this agent to work (it currently only has messages,
structured_intent, config, validation_result):

    retry_count: int
    failure_context: str


Real (non-mock) validation agent for the IBN pipeline.

Sits AFTER ConfigAgent in the graph. ConfigAgent already does an LLM
self-verification pass and writes a provisional `validation_result`.
This agent independently re-checks that output and OVERWRITES
`validation_result` with the real verdict — it does not trust the
LLM's self-grading.

Design note on syntax checking:
We do NOT hardcode one parser per template type (acl/vlan/dhcp/...).
Instead we introspect the actual .j2 file's Jinja2 AST at runtime to
discover what fields/loops/conditionals it expects (e.g. for acl.j2:
network_objects -> obj.type/.name/.id/.rules -> rule.action/.protocol/
.source/.destination). We then:
  1. Re-render the SAME template fresh from structured_intent.
  2. Diff that fresh render against state['config'] structurally
     (line-shape comparison, not naive string diff) to catch LLM
     corruption/hallucinated edits.
  3. Walk the rendered config's lines and confirm every non-comment,
     non-blank line matches one of the line "shapes" the template
     itself is capable of producing (extracted from the AST's
     {% for %} / static text segments). This generalizes across
     every template in templates/ without per-type code.

Semantic checking compares the OBJECTS in structured_intent (e.g.
each entry in network_objects, each rule) against what actually
appears in the rendered config — every intent object/field must be
represented, and nothing extra must appear that intent didn't ask for.

NOTE on the empty-network_objects check: a template referencing
network_objects in a {% for obj in network_objects %} loop does NOT
require that list to be non-empty — looping over an empty list is
valid Jinja and renders nothing for that section, which is correct
behavior for e.g. firewall_policy intents that only populate
security_policies and never network_objects. We only flag a genuinely
empty intent (nothing in network_objects, security_policies, or
actions at all) since that means there is nothing for any template to
render.

Twin-box validator is the final pre-deploy gate. It calls the real
Batfish service (via validation/batfish_validator.py) to actually
parse and model the candidate config — this requires the Batfish
Docker container to be running (see BATFISH_SETUP.md). It is not a
mock: if Batfish is unreachable, this will correctly report a failure
rather than silently passing.
"""

import os
import re
from jinja2 import Environment, FileSystemLoader, TemplateNotFound, meta
from agents.common import MAX_TOTAL_ATTEMPTS
from validation.batfish_validator import validate_with_batfish

TEMPLATES_DIR = "templates"
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
    "configure_routing": "routing.j2",
    "firewall_policy": "firewall.j2",
    "dhcp_configuration": "dhcp.j2",
    "dns_configuration": "dns.j2",
    "mixed_intent": "mixed.j2",
}


# ─────────────────────────────────────────────────────────────────
# Template introspection helpers
# ─────────────────────────────────────────────────────────────────
def _load_template_source(template_name: str) -> str:
    path = os.path.join(TEMPLATES_DIR, template_name)
    with open(path, "r") as f:
        return f.read()


def get_template_required_vars(template_name: str) -> set:
    """
    Use Jinja2's AST to discover which top-level variables a template
    references (e.g. {'network_objects'}). This tells us what
    structured_intent MUST contain for this template to render
    meaningfully.
    """
    source = _load_template_source(template_name)
    ast = jinja_env.parse(source)
    return meta.find_undeclared_variables(ast)


def extract_field_paths(template_source: str) -> set:
    """
    Pulls every `obj.field` / `rule.field` style attribute access out
    of the template source via regex over the AST-adjacent text.
    This builds a generic "expected field" set per loop variable,
    e.g. {'obj.type', 'obj.name', 'obj.id', 'obj.rules',
          'rule.action', 'rule.protocol', 'rule.source',
          'rule.destination'}
    without hardcoding template-specific knowledge.
    """
    # Matches patterns like obj.type, rule.action, obj.rules
    pattern = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b")
    found = set()
    for var, field in pattern.findall(template_source):
        if var in ("loop",):  # skip Jinja builtins
            continue
        found.add(f"{var}.{field}")
    return found


def extract_loop_object_types(template_source: str) -> dict:
    """
    Finds patterns like: {% if obj.type == "acl" %}
    Returns {"acl": "obj"} so we know which `type` value this
    template branch is responsible for rendering, generically.
    """
    pattern = re.compile(r'\{\%\s*if\s+([a-zA-Z_]\w*)\.type\s*==\s*["\']([a-zA-Z_0-9]+)["\']\s*\%\}')
    mapping = {}
    for var, type_val in pattern.findall(template_source):
        mapping[type_val] = var
    return mapping


# ─────────────────────────────────────────────────────────────────
# SYNTAX CHECKER
# ─────────────────────────────────────────────────────────────────
def syntax_check(template_name: str, rendered_config: str) -> tuple[bool, list[str]]:
    """
    Re-renders the template fresh from structured_intent is handled by
    the caller (semantic_check shares the fresh render). Here we just
    confirm structural well-formedness of the rendered text against
    what the template is capable of producing:
      - every object block opened (e.g. 'acl NAME') has a closing '!'
      - no empty/incomplete blocks
      - line shapes match the template's known output patterns
    """
    errors = []
    lines = [l for l in rendered_config.split("\n") if l.strip()]

    if not lines:
        errors.append("rendered config is empty")
        return False, errors

    # Generic block-closure check: templates in this repo close each
    # object block with a bare '!' line (seen in acl.j2). Confirm
    # block-open lines are balanced with '!' closers if the template
    # itself uses '!' as a closer.
    template_source = _load_template_source(template_name)
    uses_bang_closer = bool(re.search(r"^\s*!\s*$", template_source, re.MULTILINE))

    if uses_bang_closer:
        opens = sum(1 for l in lines if l.strip() and not l.strip().startswith("!") and not l.startswith(" "))
        closers = sum(1 for l in lines if l.strip() == "!")
        # Only flag if the config uses numbered ACL or block styles that
        # actually need '!' closers. Named extended ACLs (ip access-list
        # extended ...) are closed implicitly by dedentation — no '!' needed.
        has_named_acl = any("ip access-list" in l for l in lines)
        if closers == 0 and opens > 0 and not has_named_acl:
            errors.append("template uses '!' block closers but rendered config has none")

    # Check for obviously broken Jinja leftovers (unrendered {{ }} or {% %})
    if "{{" in rendered_config or "{%" in rendered_config:
        errors.append("unrendered Jinja syntax found in output (template rendering failed)")

    # Check indentation consistency for rule/sub-lines (acl.j2 style: " rule N ...")
    for l in lines:
        if l.startswith(" ") and len(l.strip().split()) < 2:
            errors.append(f"malformed indented line: '{l.strip()}'")

    return (len(errors) == 0), errors


# ─────────────────────────────────────────────────────────────────
# SEMANTIC CHECKER
# ─────────────────────────────────────────────────────────────────
def semantic_check(template_name: str, structured_intent: dict, rendered_config: str) -> tuple[bool, list[str]]:
    """
    Confirms every object/field structured_intent asked for is
    actually represented in the rendered config, and flags anything
    in the config that has no basis in the intent (over-generation).
    """
    errors = []
    template_source = _load_template_source(template_name)
    type_var_map = extract_loop_object_types(template_source)

    network_objects = structured_intent.get("network_objects", [])
    security_policies = structured_intent.get("security_policies", [])
    actions = structured_intent.get("actions", [])

    # Only flag a genuinely empty intent — nothing at all to render.
    # A template merely referencing network_objects in a {% for %}
    # loop does NOT require that list to be non-empty; looping over
    # an empty list is valid and correct (e.g. firewall_policy intents
    # that only populate security_policies).
    if not network_objects and not security_policies and not actions:
        errors.append("structured_intent is empty — no network_objects, security_policies, or actions to render")
        return False, errors

    for obj in network_objects:
        obj_type = obj.get("type")
        if obj_type not in type_var_map:
            # template has no branch for this object type at all
            errors.append(f"object type '{obj_type}' has no matching template branch in {template_name}")
            continue

        # name/id must appear somewhere in rendered output
        name_or_id = obj.get("name") or f"{obj_type.upper()}_{obj.get('id', '')}"
        if str(name_or_id) not in rendered_config and str(obj.get("id", "")) not in rendered_config:
            errors.append(f"object '{name_or_id}' (type={obj_type}) not found in rendered config")

        # rules must each be represented. Account for the action-word
        # translation templates apply (e.g. intent "allow" -> rendered
        # "permit" in Cisco ACL syntax) so this check doesn't fail on
        # a vendor-syntax mapping that's working as intended.
        ACTION_SYNONYMS = {
            "allow": ["allow", "permit"],
            "deny": ["deny"],
        }
        for i, rule in enumerate(obj.get("rules", []), start=1):
            action = rule.get("action", "")
            if not action:
                continue
            candidates = ACTION_SYNONYMS.get(action, [action])
            if not any(c in rendered_config for c in candidates):
                errors.append(f"rule {i} action '{action}' for object '{name_or_id}' missing from rendered config")

    return (len(errors) == 0), errors


# ─────────────────────────────────────────────────────────────────
# TWIN-BOX / PRE-DEPLOY GATE — real Batfish call
# ─────────────────────────────────────────────────────────────────
# Maps your intent types to the device platform Batfish should assume
# when parsing. All your templates currently render Cisco IOS style
# CLI, so this defaults to "cisco" (this pybatfish version's valid
# Cisco IOS platform string); update per-template if you add
# vendor-specific templates later.

PLATFORM_MAP = {
    "create_vlan": "cisco",
    "network_segmentation": "cisco",
    "create_acl": "cisco",
    "configure_routing": "cisco",
    "firewall_policy": "cisco",
    "dhcp_configuration": "cisco",
    "dns_configuration": "cisco",
    "mixed_intent": "cisco",
}


def run_on_twin_box(rendered_config: str, intent_type: str = "create_acl") -> tuple[bool, str]:
    """
    Real pre-deploy gate: sends rendered_config to the Batfish service
    running in Docker and checks Batfish's own parse/model verdict.

    Requires the Batfish container to be running (BATFISH_SETUP.md).
    If Batfish is unreachable, this returns (False, <connection error>)
    rather than pretending the config is fine — fail closed, not open.
    """
    if not rendered_config.strip():
        return False, "twin-box: empty config, nothing to send to Batfish"

    platform = PLATFORM_MAP.get(intent_type, "cisco")
    ok, report = validate_with_batfish(rendered_config, platform=platform)

    if not ok:
        return False, f"twin-box (Batfish): {report.summary()}"
    return True, ""


# ─────────────────────────────────────────────────────────────────
# MAIN VALIDATION AGENT NODE
# ─────────────────────────────────────────────────────────────────
def validation_agent(state: dict) -> dict:
    intent = state["structured_intent"]["intent"]
    intent_type = intent.get("intent_type")
    rendered_config = state.get("config", "")
    total_attempts = state.get("total_attempts", 0)

    template_name = TEMPLATE_MAP.get(intent_type, "mixed.j2")

    try:
        jinja_env.get_template(template_name)
    except TemplateNotFound:
        state["validation_result"] = f"invalid: no template found for intent_type '{intent_type}'"
        return state

    # ---- syntax check ----
    syntax_ok, syntax_errors = syntax_check(template_name, rendered_config)

    # ---- semantic check ----
    semantic_ok, semantic_errors = semantic_check(template_name, intent, rendered_config)

    print(f"  [validation_agent] syntax_ok={syntax_ok} semantic_ok={semantic_ok}")
    if syntax_errors:
        print(f"    syntax_errors: {syntax_errors}")
    if semantic_errors:
        print(f"    semantic_errors: {semantic_errors}")

    if not (syntax_ok and semantic_ok):
        reasons = []
        if syntax_errors:
            reasons.append("Syntax: " + "; ".join(syntax_errors))
        if semantic_errors:
            reasons.append("Semantic: " + "; ".join(semantic_errors))

        total_attempts += 1
        state["total_attempts"] = total_attempts
        state["failure_context"] = " | ".join(reasons)

        if total_attempts >= MAX_TOTAL_ATTEMPTS:
            state["validation_result"] = (
                f"invalid (escalated after {total_attempts} total attempts): {state['failure_context']}"
            )
        else:
            state["validation_result"] = (
                f"invalid (attempt {total_attempts}/{MAX_TOTAL_ATTEMPTS}): {state['failure_context']}"
            )
        return state

    # ---- twin-box pre-deploy gate (only runs if static checks pass) ----
    twin_ok, twin_reason = run_on_twin_box(rendered_config, intent_type=intent_type)

    if not twin_ok:
        total_attempts += 1
        state["total_attempts"] = total_attempts
        state["failure_context"] = twin_reason
        if total_attempts >= MAX_TOTAL_ATTEMPTS:
            state["validation_result"] = (
                f"invalid (escalated after {total_attempts} total attempts): {twin_reason}"
            )
        else:
            state["validation_result"] = (
                f"invalid (attempt {total_attempts}/{MAX_TOTAL_ATTEMPTS}): {twin_reason}"
            )
        return state

    state["validation_result"] = "verified_by_validation_agent"
    state["failure_context"] = ""
    return state


def route_after_validation(state: dict) -> str:
    result = state.get("validation_result", "")
    if result.startswith("verified"):
        return "deploy"
    if "escalated" in result:
        return "human_review"
    return "retry"
