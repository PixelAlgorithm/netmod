"""
agents/deployment_agent.py
Deployment Agent — Cisco DevNet Catalyst 8000 Always-On Sandbox via Netmiko SSH.
"""

from netmiko.exceptions import NetmikoTimeoutException, NetmikoAuthenticationException

from device_client import open_device_connection
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


def deployment_agent(state: dict) -> dict:
    config      = state.get("config", "")
    intent      = state.get("structured_intent", {}).get("intent", {})
    intent_type = intent.get("intent_type", "unknown")

    try:
        device_settings = get_device_settings()
    except MissingEnvironmentError as exc:
        state["deployment_result"] = f"deploy_failed: {exc}"
        return state

    print(f"\n  [deployment_agent] connecting to {device_settings.host}:{device_settings.port}")
    print(f"  [deployment_agent] intent_type: {intent_type}")

    if not config.strip():
        state["deployment_result"] = "deploy_failed: config is empty"
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

        # ── Save config ───────────────────────────────────────────
        save_output = connection.save_config()
        print(f"  [deployment_agent] saved: {save_output.strip()[:80]}")

        # ── Post-deploy verification ──────────────────────────────
        commands = VERIFY_COMMANDS.get(intent_type, ["show running-config"])
        verify_outputs = []
        for cmd in commands:
            result = connection.send_command(cmd, read_timeout=30)
            verify_outputs.append(f"--- {cmd} ---\n{result}")
            print(f"\n  {cmd}:\n{result}")

        connection.disconnect()

        state["deployment_result"] = (
            f"deployed: config pushed to '{device_settings.host}' successfully | "
            f"post-deploy verification: PASSED\n" +
            "\n".join(verify_outputs)
        )

    except NetmikoTimeoutException as e:
        msg = (
            f"deploy_failed: connection timed out to {device_settings.host}:{device_settings.port}. "
            f"Check DEVICE_HOST is correct. Error: {e}"
        )
        print(f"  [deployment_agent] {msg}")
        state["deployment_result"] = msg

    except NetmikoAuthenticationException as e:
        msg = (
            f"deploy_failed: authentication failed for '{device_settings.username}'. "
            f"Check credentials from DevNet portal. Error: {e}"
        )
        print(f"  [deployment_agent] {msg}")
        state["deployment_result"] = msg

    except Exception as e:
        msg = f"deploy_failed: unexpected error — {e}"
        print(f"  [deployment_agent] {msg}")
        state["deployment_result"] = msg

    return state


def route_after_deployment(state: dict) -> str:
    result = state.get("deployment_result", "")
    if result.startswith("deployed:"):
        return "done"
    return "human_review"
