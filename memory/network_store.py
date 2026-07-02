import json
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = "memory/ibn_network.db"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    os.makedirs("memory", exist_ok=True)
    return sqlite3.connect(DB_PATH)


def _ensure_intent_store_file() -> None:
    path = "memory/intent_store.json"
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as file_obj:
            json.dump([], file_obj)


def _json_dump(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _json_load(value: str | None):
    if not value:
        return [] if value == "[]" else {}
    return json.loads(value)


def init_db():
    """Create tables if they don't exist."""
    _ensure_intent_store_file()
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS deployed_intents (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                intent_type TEXT,
                deployment_target TEXT,
                config TEXT,
                structured_intent TEXT,
                status TEXT,
                device_host TEXT,
                deployment_result TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS device_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_host TEXT,
                hostname TEXT,
                snapshot_time TEXT,
                vlans TEXT,
                interfaces TEXT,
                acls TEXT,
                routes TEXT,
                raw_running_config TEXT
            )
            """
        )
        conn.commit()


def save_deployed_intent(intent_id, intent_type, deployment_target,
                         config, structured_intent, status,
                         device_host, deployment_result):
    """Insert or replace a deployed intent record."""
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO deployed_intents (
                id, timestamp, intent_type, deployment_target,
                config, structured_intent, status, device_host, deployment_result
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                intent_id,
                _utc_now(),
                intent_type,
                deployment_target,
                config,
                _json_dump(structured_intent),
                status,
                device_host,
                deployment_result,
            ),
        )
        conn.commit()


def save_device_snapshot(device_host, hostname, vlans,
                         interfaces, acls, routes, raw_running_config):
    """Insert a new device snapshot row."""
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO device_snapshots (
                device_host, hostname, snapshot_time,
                vlans, interfaces, acls, routes, raw_running_config
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                device_host,
                hostname,
                _utc_now(),
                _json_dump(vlans),
                _json_dump(interfaces),
                _json_dump(acls),
                _json_dump(routes),
                raw_running_config,
            ),
        )
        conn.commit()


def get_latest_snapshot(device_host: str) -> dict | None:
    """Return the most recent snapshot for a device as a dict, or None."""
    init_db()
    with _connect() as conn:
        cursor = conn.execute(
            """
            SELECT device_host, hostname, snapshot_time, vlans, interfaces,
                   acls, routes, raw_running_config
            FROM device_snapshots
            WHERE device_host = ?
            ORDER BY snapshot_time DESC, id DESC
            LIMIT 1
            """,
            (device_host,),
        )
        row = cursor.fetchone()

    if not row:
        return None

    return {
        "host": row[0],
        "device_host": row[0],
        "hostname": row[1],
        "snapshot_time": row[2],
        "vlans": _json_load(row[3]),
        "interfaces": _json_load(row[4]),
        "acls": _json_load(row[5]),
        "routes": _json_load(row[6]),
        "raw_running_config": row[7],
    }


def get_deployed_intents(device_host: str = None) -> list[dict]:
    """Return all deployed intents, optionally filtered by device_host."""
    init_db()
    query = """
        SELECT id, timestamp, intent_type, deployment_target, config,
               structured_intent, status, device_host, deployment_result
        FROM deployed_intents
    """
    params = ()
    if device_host:
        query += " WHERE device_host = ?"
        params = (device_host,)
    query += " ORDER BY timestamp DESC"

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()

    results = []
    for row in rows:
        results.append({
            "id": row[0],
            "timestamp": row[1],
            "intent_type": row[2],
            "deployment_target": row[3],
            "config": row[4],
            "structured_intent": _json_load(row[5]),
            "status": row[6],
            "device_host": row[7],
            "deployment_result": row[8],
        })
    return results


def build_memory_summary(device_host: str) -> str:
    """
    Build a human-readable summary string of the current network state
    for a device, combining the latest snapshot + recent deployed intents.
    Format it the same way network_memory_summary is currently formatted
    in the existing memory agent.
    """
    latest = get_latest_snapshot(device_host)
    intents = get_deployed_intents(device_host=device_host)

    if not latest and not intents:
        return "No network memory available yet."

    summary_lines = []
    if latest:
        summary_lines.append(
            f"Device {latest.get('hostname', 'unknown')} at {latest.get('host', 'unknown')}"
        )

        vlans = latest.get("vlans", [])
        if vlans:
            vlan_bits = [f"{vlan['id']}:{vlan['name']}" for vlan in vlans]
            summary_lines.append("VLANs: " + ", ".join(vlan_bits))

        interfaces = latest.get("interfaces", [])
        if interfaces:
            interface_bits = []
            for interface in interfaces[:10]:
                ip_display = interface.get("ip_address") or "unassigned"
                interface_bits.append(f"{interface['name']}={ip_display}")
            summary_lines.append("Interfaces: " + ", ".join(interface_bits))

        acls = latest.get("acls", [])
        if acls:
            summary_lines.append("ACLs: " + ", ".join(acl["name"] for acl in acls))

        routes = latest.get("routes", [])
        if routes:
            route_bits = [route["prefix"] for route in routes[:10] if route.get("prefix")]
            if route_bits:
                summary_lines.append("Routes: " + ", ".join(route_bits))

    if intents:
        last_deployment = intents[0]
        summary_lines.append(
            "Last deployment: "
            f"{last_deployment.get('timestamp', 'unknown')} "
            f"({last_deployment.get('status', 'unknown')})"
        )
        recent_intents = [
            f"{intent['intent_type']}->{intent['deployment_target'] or 'unknown'}"
            for intent in intents[:5]
        ]
        summary_lines.append("Recent intents: " + ", ".join(recent_intents))

    return "\n".join(summary_lines)
