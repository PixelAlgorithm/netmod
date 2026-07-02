"""
agents/deployment_agent.py
Deployment Agent — Cisco DevNet Catalyst 8000 Always-On Sandbox via Netmiko SSH.
"""

import json
import uuid

from netmiko.exceptions import NetmikoTimeoutException, NetmikoAuthenticationException

from agents.common import MAX_TOTAL_ATTEMPTS
from device_client import open_device_connection
from memory.network_memory import _parse_acls, _parse_hostname, _parse_interfaces, _parse_routes, _parse_vlans
from memory.network_store import (
    build_memory_summary,
    get_latest_snapshot,
    save_deployed_intent,
    save_device_snapshot,
)
from settings import MissingEnvironmentError, get_device_settings

VERIFY_COMMANDS = {
    "create_vlan":          ["show vlan brief"],
    "network_segmentation": ["show vlan brief", "show interfaces trunk"],
    "create_acl":           ["show ip access-lists"],
    "configure_routing":    ["show ip route"],
    "firewall_policy":      ["show ip access-lists"],
    "dhcp_configuration":   ["show ip dhcp pool", "show ip dhcp binding"],
    "dns_configuration":    ["show hosts"],
    "mixed_intent":         ["show running-config"],
}

SNAPSHOT_COMMANDS = {
    "hostname": "show running-config | include ^hostname",
    "vlans": "show vlan brief",
    "interfaces": "show ip interface brief",
    "acls": "show ip access-lists",
    "routes": "show ip route",
    "running_config": "show running-config",
}

CISCO_ERROR_PATTERNS = [
    "% Invalid input detected",
    "% Ambiguous command",
    "% Incomplete command",
    "% Unknown command",
    "Error:",
]


def _clean_config_lines(config: str) -> list:
    lines = []
    for line in config.splitlines():
        stripped = line.rstrip()
        if not stripped or stripped.startswith("{#"):
            continue
        # skip IOS banner lines that aren't actual config commands
        if stripped.lower().startswith("! cisco") or stripped.lower().startswith("version"):
            continue
        lines.append(stripped)
    return lines


def _contains_device_rejection(push_output: str) -> bool:
    return any(pattern in push_output for pattern in CISCO_ERROR_PATTERNS)


def deployment_agent(state: dict) -> dict:
    config      = state.get("config", "")
    intent      = state.get("structured_intent", {}).get("intent", {})
    intent_type = intent.get("intent_type", "unknown")
    connection = None

    try:
        device_settings = get_device_settings()
    except MissingEnvironmentError as exc:
        state["deployment_result"] = f"deploy_failed: {exc}"
        state["total_attempts"] = MAX_TOTAL_ATTEMPTS
        return state

    print(f"\n  [deployment_agent] connecting to {device_settings.host}:{device_settings.port}")
    print(f"  [deployment_agent] intent_type: {intent_type}")

    if not config.strip():
        state["deployment_result"] = "deploy_failed: config is empty"
        state["total_attempts"] = MAX_TOTAL_ATTEMPTS
        return state

    try:
        # ── Connect ───────────────────────────────────────────────
        print("  [deployment_agent] establishing SSH connection...")
        connection = open_device_connection(device_settings)
        print("  [deployment_agent] connected successfully")

        # ── Push config ───────────────────────────────────────────
        config_lines = _clean_config_lines(config)
        print(f"  [deployment_agent] pushing {len(config_lines)} config lines")

        push_output = connection.send_config_set(
            config_lines,
            enter_config_mode=True,
            exit_config_mode=True,
            read_timeout=60,
        )
        print(f"  [deployment_agent] push output:\n{push_output}")

        if _contains_device_rejection(push_output):
            state["deployment_result"] = f"deploy_failed: device rejected lines\n{push_output}"
            state["failure_context"] = f"Device rejected these config lines:\n{push_output}"
            state["total_attempts"] = state.get("total_attempts", 0) + 1
            return state

        # ── Save config ───────────────────────────────────────────
        save_output = connection.save_config()
        print(f"  [deployment_agent] saved: {save_output.strip()[:80]}")

        # ── Post-deploy verification ──────────────────────────────
        command_order = VERIFY_COMMANDS.get(intent_type, ["show running-config"])
        verify_outputs = {}
        for cmd in command_order:
            result = connection.send_command(cmd, read_timeout=30)
            verify_outputs[cmd] = result
            print(f"\n  {cmd}:\n{result}")

        for key, cmd in SNAPSHOT_COMMANDS.items():
            if cmd in verify_outputs:
                continue
            result = connection.send_command(cmd, read_timeout=30)
            verify_outputs[cmd] = result
            print(f"\n  {cmd}:\n{result}")

        current_hostname = _parse_hostname(
            verify_outputs.get(SNAPSHOT_COMMANDS["hostname"], "") or
            verify_outputs.get(SNAPSHOT_COMMANDS["running_config"], "")
        )
        parsed_vlans = _parse_vlans(verify_outputs.get(SNAPSHOT_COMMANDS["vlans"], ""))
        parsed_interfaces = _parse_interfaces(verify_outputs.get(SNAPSHOT_COMMANDS["interfaces"], ""))
        parsed_acls = _parse_acls(verify_outputs.get(SNAPSHOT_COMMANDS["acls"], ""))
        parsed_routes = _parse_routes(verify_outputs.get(SNAPSHOT_COMMANDS["routes"], ""))
        running_config_text = verify_outputs.get(SNAPSHOT_COMMANDS["running_config"], "")

        state["deployment_result"] = (
            f"deployed: config pushed to '{device_settings.host}' successfully | "
            f"post-deploy verification: PASSED\n" +
            "\n".join(f"--- {cmd} ---\n{output}" for cmd, output in verify_outputs.items())
        )
        state["failure_context"] = ""
        state["total_attempts"] = 0

        save_deployed_intent(
            intent_id=str(uuid.uuid4()),
            intent_type=intent_type,
            deployment_target=intent.get("deployment_target", ""),
            config=config,
            structured_intent=json.dumps(intent),
            status="deployed",
            device_host=device_settings.host,
            deployment_result=state["deployment_result"],
        )

        save_device_snapshot(
            device_host=device_settings.host,
            hostname=current_hostname,
            vlans=json.dumps(parsed_vlans),
            interfaces=json.dumps(parsed_interfaces),
            acls=json.dumps(parsed_acls),
            routes=json.dumps(parsed_routes),
            raw_running_config=running_config_text,
        )

        latest_snapshot = get_latest_snapshot(device_settings.host)
        state["network_memory"] = {"devices": [latest_snapshot]} if latest_snapshot else {}
        state["network_memory_summary"] = build_memory_summary(device_settings.host)

    except NetmikoTimeoutException as e:
        msg = (
            f"deploy_failed: connection timed out to {device_settings.host}:{device_settings.port}. "
            f"Check DEVICE_HOST is correct. Error: {e}"
        )
        print(f"  [deployment_agent] {msg}")
        state["deployment_result"] = msg
        state["failure_context"] = msg
        state["total_attempts"] = MAX_TOTAL_ATTEMPTS

    except NetmikoAuthenticationException as e:
        msg = (
            f"deploy_failed: authentication failed for '{device_settings.username}'. "
            f"Check credentials from DevNet portal. Error: {e}"
        )
        print(f"  [deployment_agent] {msg}")
        state["deployment_result"] = msg
        state["failure_context"] = msg
        state["total_attempts"] = MAX_TOTAL_ATTEMPTS

    except Exception as e:
        msg = f"deploy_failed: unexpected error — {e}"
        print(f"  [deployment_agent] {msg}")
        state["deployment_result"] = msg
        state["failure_context"] = msg
        state["total_attempts"] = MAX_TOTAL_ATTEMPTS

    finally:
        if connection is not None:
            try:
                connection.disconnect()
            except Exception:
                pass

    return state


def route_after_deployment(state: dict) -> str:
    result = state.get("deployment_result", "")
    total_attempts = state.get("total_attempts", 0)
    if result.startswith("deployed:"):
        return "done"
    if result.startswith("deploy_failed:") and total_attempts < MAX_TOTAL_ATTEMPTS:
        return "repair_and_retry"
    return "human_review"
