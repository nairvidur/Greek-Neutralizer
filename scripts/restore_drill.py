#!/usr/bin/env python3
"""Exercise backup, restore, and crash recovery against a real service process.

Usage: restore_drill.py [--out results.json] [--proposals 20] [--crashes 5]

Follows docs/operations/runbook.md, and fails if any step does not behave as the runbook says:

1. Write proposals and approvals through HTTP, then stop the service.
2. Back up the log and its sidecar, recording checksums, and destroy the originals.
3. Restore both to a new path, verify the checksums and the chain, and start a service on them.
   The chain must continue: a new submission extends it, and it survives another restart.
4. Restore damaged backups: the sidecar missing, the sidecar from another time, the log truncated,
   and a record edited. Each must report a broken chain and refuse to submit.
5. SIGKILL the service during concurrent submissions, repeatedly. After each crash, record what
   verification reports, and check that every acknowledged submission is still in the log.

This runs on one machine against a local disk. It does not exercise a remote backup target, a real
power loss, or a disk that lies about fsync.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx2

sys.path.insert(0, str(Path(__file__).parent))
from load_test import client, running_service, start_on_a_free_port

from greeks_neutralizer._version import __version__
from greeks_neutralizer.workflow import verify_audit_log

BROKEN_OR_NOT = frozenset({200, 503})
ACKNOWLEDGE_TIMEOUT_SECONDS = 60.0


def _sidecar(log: Path) -> Path:
    return log.with_name(log.name + ".head")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hedge_request(http: httpx2.Client) -> dict[str, Any]:
    sample: dict[str, Any] = http.get("/v1/example").json()
    return sample


def _record_ids(log: Path) -> list[str]:
    return [
        record["detail"]["id"]
        for record in (json.loads(line) for line in log.read_text().splitlines() if line.strip())
        if record["event"] == "submitted"
    ]


def write_records(base: str, proposals: int) -> int:
    """Proposals and approvals through the API. Returns how many proposals were approved."""
    approved = 0
    with client(base, "alice") as alice, client(base, "bob") as bob:
        request = _hedge_request(alice)
        for _ in range(proposals):
            created = alice.post("/v1/proposals", json=request)
            created.raise_for_status()
            record_id = created.json()["id"]
            bob.post(
                f"/v1/proposals/{record_id}/approve", json={"reason": "drill"}
            ).raise_for_status()
            approved += 1
    return approved


def backup_and_restore(work: Path, proposals: int) -> dict[str, Any]:
    original = work / "live" / "audit.jsonl"
    original.parent.mkdir()
    with running_service(original) as base:
        write_records(base, proposals)
    records_before = verify_audit_log(original).records

    # The runbook's backup: both files, while stopped.
    backup = work / "backup"
    backup.mkdir()
    manifest = {}
    for source in (original, _sidecar(original)):
        shutil.copy2(source, backup / source.name)
        manifest[source.name] = _sha256(source)
    shutil.rmtree(original.parent)

    started = time.monotonic()
    restored = work / "restored" / "audit.jsonl"
    restored.parent.mkdir()
    for name in manifest:
        shutil.copy2(backup / name, restored.parent / name)
    checksums_match = all(_sha256(restored.parent / n) == h for n, h in manifest.items())
    check = verify_audit_log(restored)
    with running_service(restored) as base, client(base) as http:
        health = http.get("/healthz")
        restore_seconds = time.monotonic() - started
        created = http.post("/v1/proposals", json=_hedge_request(http))
    extended = verify_audit_log(restored)

    with running_service(restored) as base, client(base) as http:
        after_restart = http.get("/healthz")

    return {
        "records_backed_up": records_before,
        "checksums_match": checksums_match,
        "chain_ok_after_restore": check.ok,
        "records_after_restore": check.records,
        "healthz_after_restore": health.status_code,
        "seconds_from_restore_to_healthy": restore_seconds,
        "new_submission_status": created.status_code,
        "chain_ok_after_new_submission": extended.ok,
        "records_after_new_submission": extended.records,
        "healthz_after_second_restart": after_restart.status_code,
        "passed": bool(
            checksums_match
            and check.ok
            and check.records == records_before
            and health.status_code == 200
            and created.status_code == 201
            and extended.ok
            and extended.records == records_before + 1
            and after_restart.status_code == 200
        ),
    }


def _drop_sidecar(log: Path) -> None:
    _sidecar(log).unlink()


def _old_sidecar(log: Path) -> None:
    head = json.loads(_sidecar(log).read_text())
    previous = json.loads(log.read_text().splitlines()[-2])["hash"]
    _sidecar(log).write_text(json.dumps({"head": previous, "count": head["count"] - 1}))


def _truncate(log: Path) -> None:
    lines = log.read_text().splitlines(keepends=True)
    log.write_text("".join(lines[:-3]))


def _edit_record(log: Path) -> None:
    lines = log.read_text().splitlines(keepends=True)
    record = json.loads(lines[1])
    record["actor"] = "mallory"
    lines[1] = json.dumps(record, sort_keys=True) + "\n"
    log.write_text("".join(lines))


DAMAGE: dict[str, Callable[[Path], None]] = {
    "sidecar_missing": _drop_sidecar,
    "sidecar_from_an_earlier_time": _old_sidecar,
    "log_truncated": _truncate,
    "record_edited": _edit_record,
}


def damaged_restores(work: Path) -> dict[str, Any]:
    source = work / "source" / "audit.jsonl"
    source.parent.mkdir()
    with running_service(source) as base:
        write_records(base, 3)

    cases = {}
    for name, damage in DAMAGE.items():
        target = work / name / "audit.jsonl"
        target.parent.mkdir()
        shutil.copy2(source, target)
        shutil.copy2(_sidecar(source), _sidecar(target))
        damage(target)
        with running_service(target, BROKEN_OR_NOT) as base, client(base) as http:
            health = http.get("/healthz")
            refused = http.post("/v1/proposals", json=_hedge_request(http))
        cases[name] = {
            "healthz": health.status_code,
            "healthz_status": health.json()["status"],
            "submit": refused.status_code,
            "detail": health.json()["detail"],
            "passed": health.status_code == 503 and refused.status_code == 409,
        }
    return {"cases": cases, "passed": all(case["passed"] for case in cases.values())}


def _submit_until_dead(
    base: str, request: dict[str, Any], acknowledged: list[str], lock: threading.Lock
) -> None:
    with client(base) as http:
        while True:
            try:
                response = http.post("/v1/proposals", json=request)
            except httpx2.TransportError:
                return
            if response.status_code == 201:
                with lock:
                    acknowledged.append(response.json()["id"])


def crash_recovery(work: Path, crashes: int) -> dict[str, Any]:
    """SIGKILL during concurrent submissions. Acknowledged means durable, or the drill fails."""
    outcomes = []
    for attempt in range(crashes):
        log = work / f"crash_{attempt}" / "audit.jsonl"
        log.parent.mkdir()
        with tempfile.TemporaryFile() as output:
            process, base = start_on_a_free_port(log, output, frozenset({200}))
            try:
                with client(base) as http:
                    request = _hedge_request(http)

                acknowledged: list[str] = []
                lock = threading.Lock()
                with ThreadPoolExecutor(8) as pool:
                    futures = [
                        pool.submit(_submit_until_dead, base, request, acknowledged, lock)
                        for _ in range(8)
                    ]
                    # A crash before anything was acknowledged would prove nothing. The count
                    # varies, so that the kill lands in different places. The submitters stop only
                    # when the service dies, so it must die before the pool waits for them.
                    try:
                        deadline = time.monotonic() + ACKNOWLEDGE_TIMEOUT_SECONDS
                        while len(acknowledged) < 5 + 3 * attempt and time.monotonic() < deadline:
                            time.sleep(0.01)
                    finally:
                        process.kill()
                        process.wait()
                    for future in futures:
                        future.result()
            finally:
                process.kill()
                process.wait()

        check = verify_audit_log(log)
        recorded = set(_record_ids(log))
        lost = [record_id for record_id in acknowledged if record_id not in recorded]
        outcomes.append(
            {
                "acknowledged": len(acknowledged),
                "recorded": len(recorded),
                "acknowledged_but_lost": len(lost),
                "chain_ok": check.ok,
                "reason": check.reason,
            }
        )
    return {
        "attempts": outcomes,
        "acknowledged_but_lost": sum(o["acknowledged_but_lost"] for o in outcomes),
        "crashes_leaving_a_broken_chain": sum(not o["chain_ok"] for o in outcomes),
        "passed": all(o["acknowledged"] > 0 and o["acknowledged_but_lost"] == 0 for o in outcomes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--proposals", type=int, default=20)
    parser.add_argument("--crashes", type=int, default=5)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        for name in ("restore", "damage", "crash"):
            (root / name).mkdir()
        results = {
            "code_version": __version__,
            "note": "One machine and a local disk. The script's docstring says what is not run.",
            "restore": backup_and_restore(root / "restore", args.proposals),
            "damaged_restores": damaged_restores(root / "damage"),
            "crash_recovery": crash_recovery(root / "crash", args.crashes),
        }

    text = json.dumps(results, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)
    sections = ("restore", "damaged_restores", "crash_recovery")
    passed = all(results[key]["passed"] for key in sections)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
