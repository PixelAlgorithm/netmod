"""
validation/batfish_validator.py

Real pybatfish integration for the twin-box / pre-deploy gate.

This talks to an actual Batfish service running in Docker (see
BATFISH_SETUP.md). It is NOT a mock — it sends the rendered config
text to Batfish, asks Batfish to parse and model it, and reports back
whether Batfish considers it valid, with structured error/warning
detail you can feed into the retry loop's failure_context.

Requires:
    pip install pybatfish
    Docker container running:
        docker run -d --name batfish -p 9996:9996 -p 9997:9997 batfish/allinone

Usage:
    from validation.batfish_validator import validate_with_batfish

    ok, report = validate_with_batfish(rendered_config, platform="cisco-ios")
    if not ok:
        print(report.summary())
"""

from dataclasses import dataclass, field
from typing import Optional
import uuid

from pybatfish.client.session import Session


# ─────────────────────────────────────────────────────────────────
# Report object — structured result handed back to validation_agent
# ─────────────────────────────────────────────────────────────────
@dataclass
class BatfishReport:
    parsed_ok: bool
    file_status: str               # PASSED / PARTIALLY_PARSED / FAILED
    parse_warnings: list = field(default_factory=list)
    init_issues: list = field(default_factory=list)
    error: Optional[str] = None    # set if Batfish/connection itself failed

    def summary(self) -> str:
        if self.error:
            return f"Batfish connection/error: {self.error}"
        lines = [f"file_status={self.file_status}"]
        if self.parse_warnings:
            lines.append("parse_warnings: " + "; ".join(self.parse_warnings))
        if self.init_issues:
            lines.append("init_issues: " + "; ".join(self.init_issues))
        return " | ".join(lines)


# ─────────────────────────────────────────────────────────────────
# Session management — reuse one session across calls
# ─────────────────────────────────────────────────────────────────
_session: Optional[Session] = None


def get_session(host: str = "localhost") -> Session:
    global _session
    if _session is None:
        _session = Session(host=host)
    return _session


# ─────────────────────────────────────────────────────────────────
# Core validation call
# ─────────────────────────────────────────────────────────────────
def validate_with_batfish(
    rendered_config: str,
    platform: str = "cisco",
    host: str = "localhost",
    network_name: Optional[str] = None,
) -> tuple[bool, BatfishReport]:
    """
    Sends rendered_config to a running Batfish service and checks
    whether it parses cleanly.

    Returns (ok, BatfishReport). ok is True only if file parse status
    is PASSED and there are no parse warnings or init issues.
    """
    try:
        bf = get_session(host=host)
    except Exception as e:
        return False, BatfishReport(
            parsed_ok=False, file_status="ERROR",
            error=f"could not create Batfish session: {e}"
        )

    # Use a fresh network per validation run so snapshots don't collide
    # across concurrent/parallel validation calls.
    net_name = network_name or f"ibn_validation_{uuid.uuid4().hex[:8]}"

    try:
        bf.set_network(net_name)

        # NOTE: in this pybatfish version, init_snapshot_from_text takes
        # the config as a single string via `text`, plus a `filename`
        # arg — it does NOT accept a {filename: content} dict. Passing
        # a dict here previously caused:
        #   "memoryview: a bytes-like object is required, not 'dict'"
        bf.init_snapshot_from_text(
            rendered_config,
            filename="candidate.cfg",
            platform=platform,
            snapshot_name="candidate",
            overwrite=True,
        )
    except Exception as e:
        return False, BatfishReport(
            parsed_ok=False, file_status="ERROR",
            error=f"snapshot initialization failed: {e}"
        )

    # ---- 1. File parse status (PASSED / PARTIALLY_PARSED / FAILED) ----
    try:
        parse_status_df = bf.q.fileParseStatus().answer().frame()
        file_status = "UNKNOWN"
        if not parse_status_df.empty:
            file_status = parse_status_df.iloc[0]["Status"]
    except Exception as e:
        return False, BatfishReport(
            parsed_ok=False, file_status="ERROR",
            error=f"fileParseStatus question failed: {e}"
        )

    # ---- 2. Parse warnings (lines Batfish couldn't fully understand) ----
    parse_warnings = []
    try:
        warn_df = bf.q.parseWarning().answer().frame()
        for _, row in warn_df.iterrows():
            detail = row.get("Text") or row.get("Comment") or str(row.to_dict())
            parse_warnings.append(str(detail))
    except Exception as e:
        parse_warnings.append(f"(could not retrieve parse warnings: {e})")

    # ---- 3. Init issues (deeper conversion/model-building problems) ----
    init_issues = []
    try:
        issues_df = bf.q.initIssues().answer().frame()
        for _, row in issues_df.iterrows():
            detail = row.get("Details") or str(row.to_dict())
            init_issues.append(str(detail))
    except Exception as e:
        init_issues.append(f"(could not retrieve init issues: {e})")

    parsed_ok = (
        file_status == "PASSED"
        and len(parse_warnings) == 0
        and len(init_issues) == 0
    )

    report = BatfishReport(
        parsed_ok=parsed_ok,
        file_status=str(file_status),
        parse_warnings=parse_warnings,
        init_issues=init_issues,
    )
    return parsed_ok, report


# ─────────────────────────────────────────────────────────────────
# Optional: reachability / ACL behavior check (deeper than parse-only)
# ─────────────────────────────────────────────────────────────────
def check_acl_behavior(
    acl_name: str,
    source_ip: str,
    destination_ip: str,
    host: str = "localhost",
) -> Optional[dict]:
    """
    Runs Batfish's filterLineReachability / searchFilters-style check
    to confirm a named ACL actually behaves as expected for a given
    flow. This goes beyond "does it parse" into "does it do what the
    intent asked for" — use this for the semantic side of twin-box
    testing once parsing passes.

    Returns None if the question itself failed to run (e.g. ACL name
    not found, connection issue) so callers can fall back gracefully.
    """
    try:
        bf = get_session(host=host)
        result = bf.q.searchFilters(
            filters=acl_name,
            headers={"srcIps": source_ip, "dstIps": destination_ip},
        ).answer().frame()
        if result.empty:
            return None
        return result.iloc[0].to_dict()
    except Exception:
        return None


if __name__ == "__main__":
    # Quick manual test — requires the Docker Batfish container running.
    sample_config = """! Cisco IOS Configuration
version 15.2
hostname VALIDATION-NODE
!
ip access-list extended BLOCK_VLAN30
 deny ip any 30.0.0.0 0.0.0.255
!
end
"""
    ok, report = validate_with_batfish(sample_config, platform="cisco")
    print("parsed_ok:", ok)
    print(report.summary())