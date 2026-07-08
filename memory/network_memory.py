import re
from copy import deepcopy
from datetime import datetime, timezone

from langgraph.types import interrupt

from device_client import open_device_connection
from memory.network_store import (
    build_memory_summary,
    get_deployed_intents,
    get_latest_snapshot,
    init_db,
    save_device_snapshot,
)
from settings import MissingEnvironmentError, get_device_settings


def _parse_hostname(output: str) -> str:
    match = re.search(r"^hostname\s+(\S+)", output, re.MULTILINE)
    return match.group(1) if match else "unknown"


def _parse_vlans(output: str) -> list[dict]:
    vlans = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("vlan name") or stripped.startswith("----"):
            continue
        match = re.match(r"^(\d+)\s+(\S+)\s+(\S+)(?:\s+(.*))?$", stripped)
        if not match:
            continue
        ports = []
        if match.group(4):
            ports = [port.strip() for port in match.group(4).split(",") if port.strip()]
        vlans.append({
            "id": int(match.group(1)),
            "name": match.group(2),
            "status": match.group(3),
            "ports": ports,
        })
    return vlans


def _parse_interfaces(output: str) -> list[dict]:
    interfaces = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("interface"):
            continue
        match = re.match(
            r"^(\S+)\s+(\S+)\s+\S+\s+\S+\s+(.+?)\s{2,}(\S+)$",
            line.rstrip(),
        )
        if not match:
            continue
        interface_name, ip_address, status, protocol = match.groups()
        interfaces.append({
            "name": interface_name,
            "ip_address": None if ip_address.lower() == "unassigned" else ip_address,
            "status": status.strip(),
            "protocol": protocol.strip(),
        })
    return interfaces


def _parse_acls(output: str) -> list[dict]:
    acls = []
    current_acl = None

    for raw_line in output.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        header_match = re.match(r"^(Standard|Extended) IP access list (\S+)", stripped)
        if header_match:
            current_acl = {
                "type": header_match.group(1).lower(),
                "name": header_match.group(2),
                "entries": [],
            }
            acls.append(current_acl)
            continue

        if current_acl is not None:
            current_acl["entries"].append(stripped)

    return acls


def _parse_routes(output: str) -> list[dict]:
    routes = []
    for raw_line in output.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("Gateway of last resort") or stripped.startswith("Codes:"):
            continue

        prefix_match = re.match(r"^([A-Z\*]+)\s+(\d+\.\d+\.\d+\.\d+/\d+)", stripped)
        if prefix_match:
            next_hop_match = re.search(r"via\s+(\d+\.\d+\.\d+\.\d+)", stripped)
            interface_match = re.search(r",\s*([A-Za-z][A-Za-z0-9/.\-]+)$", stripped)
            routes.append({
                "code": prefix_match.group(1),
                "prefix": prefix_match.group(2),
                "next_hop": next_hop_match.group(1) if next_hop_match else None,
                "interface": interface_match.group(1) if interface_match else None,
            })
            continue

        connected_match = re.match(
            r"^([A-Z\*]+)\s+(\d+\.\d+\.\d+\.\d+/\d+)\s+is directly connected,\s+(\S+)",
            stripped,
        )
        if connected_match:
            routes.append({
                "code": connected_match.group(1),
                "prefix": connected_match.group(2),
                "next_hop": None,
                "interface": connected_match.group(3),
            })

    return routes


def _parse_topology(output: str) -> list[dict]:
    links = []
    device_id = None
    local_interface = None

    for raw_line in output.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("Device ID:"):
            device_id = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Interface:"):
            local_match = re.search(r"Interface:\s*([^,]+),", stripped)
            remote_match = re.search(r"Port ID \(outgoing port\):\s*(.+)$", stripped)
            local_interface = local_match.group(1).strip() if local_match else None
            remote_interface = remote_match.group(1).strip() if remote_match else None
            if device_id and local_interface and remote_interface:
                links.append({
                    "neighbor": device_id,
                    "local_interface": local_interface,
                    "remote_interface": remote_interface,
                })
                device_id = None
                local_interface = None

    return links


def _summarize_memory(memory: dict) -> str:
    devices = memory.get("devices", [])
    if not devices:
        return "No network memory available yet."

    summary_lines = []
    for device in devices:
        summary_lines.append(
            f"Device {device.get('hostname', 'unknown')} at {device.get('host', 'unknown')}"
        )

        vlans = device.get("vlans", [])
        if vlans:
            vlan_bits = [f"{vlan['id']}:{vlan['name']}" for vlan in vlans]
            summary_lines.append("VLANs: " + ", ".join(vlan_bits))

        interfaces = device.get("interfaces", [])
        if interfaces:
            interface_bits = []
            for interface in interfaces[:10]:
                ip_display = interface.get("ip_address") or "unassigned"
                interface_bits.append(f"{interface['name']}={ip_display}")
            summary_lines.append("Interfaces: " + ", ".join(interface_bits))

        acls = device.get("acls", [])
        if acls:
            summary_lines.append("ACLs: " + ", ".join(acl["name"] for acl in acls))

        routes = device.get("routes", [])
        if routes:
            route_bits = [route["prefix"] for route in routes[:10] if route.get("prefix")]
            if route_bits:
                summary_lines.append("Routes: " + ", ".join(route_bits))

        topology = device.get("topology", {}).get("links", [])
        if topology:
            link_bits = [
                f"{link['local_interface']}->{link['neighbor']}:{link['remote_interface']}"
                for link in topology[:5]
            ]
            summary_lines.append("Topology: " + ", ".join(link_bits))

    history = memory.get("deployment_history", [])
    if history:
        last_deployment = history[-1]
        summary_lines.append(
            "Last deployment: "
            f"{last_deployment.get('timestamp', 'unknown')} "
            f"({last_deployment.get('status', 'unknown')})"
        )

    return "\n".join(summary_lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pull_live_snapshot(device_settings) -> dict | None:
    """
    SSH into device, run show commands, parse, save to SQLite.
    Returns a device_dict on success, None on failure.
    Logs errors but does not raise.
    """
    connection = None
    try:
        connection = open_device_connection(device_settings)
        running_config = connection.send_command("show running-config", read_timeout=30)
        access_lists = connection.send_command("show ip access-lists", read_timeout=30)
        vlan_brief = connection.send_command("show vlan brief", read_timeout=30)
        interface_brief = connection.send_command("show ip interface brief", read_timeout=30)
        routes_output = connection.send_command("show ip route", read_timeout=30)
        connection.send_command("show version", read_timeout=30)
        topology_output = connection.send_command("show cdp neighbors detail", read_timeout=30)

        hostname = _parse_hostname(running_config)
        vlans = _parse_vlans(vlan_brief)
        interfaces = _parse_interfaces(interface_brief)
        acls = _parse_acls(access_lists)
        routes = _parse_routes(routes_output)
        topology_links = _parse_topology(topology_output)
        snapshot_time = _utc_now()

        save_device_snapshot(
            device_host=device_settings.host,
            hostname=hostname,
            vlans=vlans,
            interfaces=interfaces,
            acls=acls,
            routes=routes,
            raw_running_config=running_config,
        )

        device_dict = {
            "host": device_settings.host,
            "hostname": hostname,
            "vlans": vlans,
            "interfaces": interfaces,
            "acls": acls,
            "routes": routes,
            "topology": {"links": topology_links},
            "snapshot_time": snapshot_time,
            "recent_intents": [
                f"{i['timestamp']} {i['intent_type']} ({i['status']})"
                for i in get_deployed_intents(device_settings.host)[:5]
            ],
        }
        return device_dict
    except Exception as exc:
        print(f"  [network_memory] live SSH pull failed: {exc}")
        return None
    finally:
        if connection is not None:
            try:
                connection.disconnect()
            except Exception:
                pass


class NetworkMemory:
    def __init__(self, device_host: str | None = None):
        init_db()
        self.device_host = device_host
        self.data = self.load()

    def load(self) -> dict:
        if not self.device_host:
            return {"devices": []}
        latest = get_latest_snapshot(self.device_host)
        if not latest:
            return {"devices": []}
        return {"devices": [latest]}

    def snapshot(self) -> dict:
        return deepcopy(self.data)

    def summary(self) -> str:
        if self.device_host:
            return build_memory_summary(self.device_host)
        return _summarize_memory(self.data)

    def get_devices(self) -> list[dict]:
        return self.data.get("devices", [])

    def get_primary_device(self) -> dict | None:
        devices = self.get_devices()
        return devices[0] if devices else None

    def find_vlan(self, vlan_id: int) -> dict | None:
        for device in self.get_devices():
            for vlan in device.get("vlans", []):
                if vlan.get("id") == vlan_id:
                    return vlan
        return None

    def find_interface(self, interface_name: str) -> dict | None:
        for device in self.get_devices():
            for interface in device.get("interfaces", []):
                if interface.get("name") == interface_name:
                    return interface
        return None

    def find_acl(self, acl_name: str) -> dict | None:
        for device in self.get_devices():
            for acl in device.get("acls", []):
                if acl.get("name") == acl_name:
                    return acl
        return None


def load_network_memory_node(state: dict) -> dict:
    try:
        device_settings = get_device_settings()
    except MissingEnvironmentError:
        state["network_memory"] = state.get("network_memory", {})
        state["network_memory_summary"] = state.get("network_memory_summary", "")
        return state

    live_device = _pull_live_snapshot(device_settings)
    if live_device is not None:
        state["network_memory"] = {"devices": [live_device]}
        state["network_memory_summary"] = _summarize_memory(state["network_memory"])
        return state

    latest_snapshot = get_latest_snapshot(device_settings.host)
    last_snapshot_time = latest_snapshot.get("snapshot_time") if latest_snapshot else None
    warning = (
        f"Device unreachable: unable to pull live state from {device_settings.host}\n"
        f"Last snapshot: {last_snapshot_time or 'never'}\n"
        "Options: (r)retry  (c)continue with cached data  (a)abort\n"
        "Your choice: "
    )
    print(
        f"  [network_memory] warning: device unreachable, "
        f"last snapshot={last_snapshot_time or 'never'}"
    )
    choice = interrupt(warning).strip().lower()

    if choice in {"r", "retry"}:
        live_device = _pull_live_snapshot(device_settings)
        if live_device is not None:
            state["network_memory"] = {"devices": [live_device]}
            state["network_memory_summary"] = _summarize_memory(state["network_memory"])
            return state

    if choice in {"a", "abort"}:
        raise SystemExit(1)

    if latest_snapshot:
        state["network_memory"] = {"devices": [latest_snapshot]}
        state["network_memory_summary"] = build_memory_summary(device_settings.host)
    else:
        state["network_memory"] = {}
        state["network_memory_summary"] = "No network memory available."

    return state


def update_network_memory_node(state: dict) -> dict:
    try:
        device_settings = get_device_settings()
    except MissingEnvironmentError:
        return state

    live_device = _pull_live_snapshot(device_settings)
    if live_device is not None:
        state["network_memory"] = {"devices": [live_device]}
        state["network_memory_summary"] = _summarize_memory(state["network_memory"])
        return state

    latest_snapshot = get_latest_snapshot(device_settings.host)
    if latest_snapshot:
        state["network_memory"] = {"devices": [latest_snapshot]}
        state["network_memory_summary"] = build_memory_summary(device_settings.host)
    else:
        state["network_memory"] = {}
        state["network_memory_summary"] = "No network memory available."
    return state
