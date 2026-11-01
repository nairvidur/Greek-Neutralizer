"""Proposal lifecycle: propose, four-eyes approval, export, and a tamper-evident audit trail.

This software never routes an order. The furthest a proposal can go is an exported ticket, and
sending it to a broker is a separate human or system action outside this package.

The audit log is an append-only JSON Lines file where each record carries the hash of the previous
one, so editing or removing a line breaks the chain and `verify_audit_log` finds it. This detects
tampering. It does not prevent it: production retention needs write-once storage as well.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from greeks_neutralizer.domain import LinearInstrument, OptionContract, OptionRight
from greeks_neutralizer.errors import GNError, ValidationError
from greeks_neutralizer.hedging.problem import HedgeProposal, ProposalStatus

_GENESIS = "0" * 64


class WorkflowError(GNError):
    """An illegal lifecycle action, such as approving your own proposal or exporting a stale one."""


class State(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPORTED = "exported"


@dataclass(frozen=True, slots=True)
class WorkflowPolicy:
    require_second_approver: bool = True
    approval_valid_for: timedelta = timedelta(minutes=5)
    allow_unstable: bool = False


@dataclass(frozen=True, slots=True)
class ProposalRecord:
    id: str
    proposal: HedgeProposal
    created_by: str
    created_at: datetime
    state: State = State.PROPOSED
    decided_by: str | None = None
    decided_at: datetime | None = None
    reason: str = ""


class AuditLog:
    """Hash-chained append-only log. Safe across threads in one process."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._head = self._read_head()

    def _read_head(self) -> str:
        if not self.path.exists():
            return _GENESIS
        head = _GENESIS
        with self.path.open() as handle:
            for line in handle:
                if line.strip():
                    head = json.loads(line)["hash"]
        return head

    def append(self, event: str, actor: str, detail: dict[str, Any], at: datetime) -> str:
        with self._lock:
            body = {
                "at": at.isoformat(),
                "event": event,
                "actor": actor,
                "detail": detail,
                "prev": self._head,
            }
            digest = _record_hash(body)
            line = json.dumps({**body, "hash": digest}, sort_keys=True)
            with self.path.open("a") as handle:
                handle.write(line + "\n")
            self._head = digest
            return digest


def _record_hash(body: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class AuditVerification:
    ok: bool
    records: int
    first_bad_line: int | None
    reason: str = ""


def verify_audit_log(path: Path) -> AuditVerification:
    """Recompute the chain from the start. Any edit, removal, or reorder is reported by line."""
    previous = _GENESIS
    count = 0
    if not path.exists():
        return AuditVerification(True, 0, None)
    with path.open() as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                digest = record.pop("hash")
            except (json.JSONDecodeError, KeyError) as exc:
                return AuditVerification(False, count, number, f"unreadable record: {exc}")
            if record.get("prev") != previous:
                return AuditVerification(False, count, number, "chain link broken")
            if _record_hash(record) != digest:
                return AuditVerification(False, count, number, "record contents changed")
            previous = digest
            count += 1
    return AuditVerification(True, count, None)


@dataclass(slots=True)
class ProposalStore:
    """In-process proposal store with an audit trail for every action.

    `clock` is injected so that expiry is testable and nothing depends on wall time directly.
    """

    audit: AuditLog
    policy: WorkflowPolicy = field(default_factory=WorkflowPolicy)
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    _records: dict[str, ProposalRecord] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def submit(self, proposal: HedgeProposal, actor: str) -> ProposalRecord:
        _require_actor(actor)
        if not proposal.verification.passed:
            raise WorkflowError("a proposal that failed verification cannot be submitted")
        now = self.clock()
        record = ProposalRecord(str(uuid.uuid4()), proposal, actor, now)
        with self._lock:
            self._records[record.id] = record
        self.audit.append(
            "submitted",
            actor,
            {
                "id": record.id,
                "status": proposal.status.value,
                "legs": [
                    {"instrument": leg.instrument.key, "quantity": leg.quantity}
                    for leg in proposal.legs
                ],
                "problem_fingerprint": proposal.manifest.problem_fingerprint,
                "market_fingerprint": proposal.manifest.market_fingerprint,
                "code_version": proposal.manifest.code_version,
            },
            now,
        )
        return record

    def get(self, record_id: str) -> ProposalRecord:
        with self._lock:
            record = self._records.get(record_id)
        if record is None:
            raise WorkflowError(f"no proposal with id {record_id}")
        return record

    def list(self) -> list[ProposalRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda r: r.created_at, reverse=True)

    def _transition(self, record: ProposalRecord, **changes: Any) -> ProposalRecord:
        updated = replace(record, **changes)
        with self._lock:
            self._records[record.id] = updated
        return updated

    def approve(self, record_id: str, actor: str, reason: str = "") -> ProposalRecord:
        _require_actor(actor)
        record = self.get(record_id)
        if record.state is not State.PROPOSED:
            if record.state is State.APPROVED and record.decided_by == actor:
                return record  # Idempotent: the state the caller asked for already holds.
            raise WorkflowError(f"cannot approve a proposal that is {record.state.value}")
        if self.policy.require_second_approver and actor == record.created_by:
            raise WorkflowError("four-eyes: the proposer cannot approve their own proposal")
        if record.proposal.status is ProposalStatus.UNSTABLE and not self.policy.allow_unstable:
            raise WorkflowError(
                "this proposal is unstable under small input changes, and policy blocks approval"
            )
        now = self.clock()
        self._check_fresh(record, now)
        updated = self._transition(
            record, state=State.APPROVED, decided_by=actor, decided_at=now, reason=reason
        )
        self.audit.append("approved", actor, {"id": record_id, "reason": reason}, now)
        return updated

    def reject(self, record_id: str, actor: str, reason: str) -> ProposalRecord:
        _require_actor(actor)
        if not reason.strip():
            raise ValidationError("a rejection needs a reason")
        record = self.get(record_id)
        if record.state is State.REJECTED:
            return record
        if record.state is not State.PROPOSED:
            raise WorkflowError(f"cannot reject a proposal that is {record.state.value}")
        now = self.clock()
        updated = self._transition(
            record, state=State.REJECTED, decided_by=actor, decided_at=now, reason=reason
        )
        self.audit.append("rejected", actor, {"id": record_id, "reason": reason}, now)
        return updated

    def _check_fresh(self, record: ProposalRecord, now: datetime) -> None:
        age = now - record.created_at
        if age > self.policy.approval_valid_for:
            raise WorkflowError(
                f"proposal is {age.total_seconds():.0f}s old, past the "
                f"{self.policy.approval_valid_for.total_seconds():.0f}s limit: the market has "
                "moved, so re-propose it"
            )

    def export_tickets(self, record_id: str, actor: str, fmt: str = "csv") -> str:
        """Tickets for an approved, still-fresh proposal. Limit prices are the quoted touch."""
        _require_actor(actor)
        record = self.get(record_id)
        if record.state not in (State.APPROVED, State.EXPORTED):
            raise WorkflowError(f"only an approved proposal can be exported: {record.state.value}")
        now = self.clock()
        self._check_fresh(record, now)
        text = {"csv": tickets_csv, "fix": tickets_fix}.get(fmt)
        if text is None:
            raise ValidationError(f"unknown ticket format {fmt!r}; valid: csv, fix")
        output = text(record.proposal, record.id)
        self._transition(record, state=State.EXPORTED)
        self.audit.append(
            "exported",
            actor,
            {"id": record_id, "format": fmt, "sha256": hashlib.sha256(output.encode()).hexdigest()},
            now,
        )
        return output


def _require_actor(actor: str) -> None:
    if not actor or not actor.strip():
        raise ValidationError("an authenticated actor is required")


def tickets_csv(proposal: HedgeProposal, proposal_id: str) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "proposal_id",
            "side",
            "quantity",
            "symbol",
            "underlying",
            "right",
            "strike",
            "expiry",
            "limit_price",
            "rationale",
        ]
    )
    for leg in proposal.legs:
        instrument = leg.instrument
        if isinstance(instrument, OptionContract):
            detail = [
                instrument.underlying,
                instrument.underlying,
                instrument.right.value,
                instrument.strike,
                instrument.expiry.isoformat(),
            ]
        else:
            detail = [instrument.symbol, instrument.underlying, "", "", ""]
        writer.writerow(
            [proposal_id, leg.side, abs(leg.quantity), *detail, leg.expected_price, leg.rationale]
        )
    return buffer.getvalue()


def tickets_fix(proposal: HedgeProposal, proposal_id: str) -> str:
    """FIX 4.4 NewOrderSingle message bodies, pipe-delimited, one per leg.

    Application fields only: session headers, sequence numbers, and checksums belong to the sender's
    FIX engine. Review them against your broker's rules of engagement; none has certified this.
    """
    lines: list[str] = []
    for index, leg in enumerate(proposal.legs, start=1):
        instrument = leg.instrument
        fields: list[tuple[int, str]] = [
            (35, "D"),
            (11, f"{proposal_id[:8]}-{index}"),
            (54, "1" if leg.quantity > 0 else "2"),
            (38, str(abs(leg.quantity))),
            (40, "2"),
            (44, f"{leg.expected_price:.4f}"),
            (59, "0"),
        ]
        if isinstance(instrument, OptionContract):
            fields += [
                (55, instrument.underlying),
                (167, "OPT"),
                (541, instrument.expiry.strftime("%Y%m%d")),
                (201, "1" if instrument.right is OptionRight.CALL else "0"),
                (202, f"{instrument.strike:.4f}"),
                (231, f"{instrument.multiplier:g}"),
            ]
        else:
            assert isinstance(instrument, LinearInstrument)
            fields += [(55, instrument.symbol), (167, "FUT" if instrument.expiry else "CS")]
        lines.append("|".join(f"{tag}={value}" for tag, value in fields) + "|")
    return "\n".join(lines) + ("\n" if lines else "")
