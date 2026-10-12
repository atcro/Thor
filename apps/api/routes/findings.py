"""Repair findings over HTTP: "What did the technician find?" (Phase A.3).

`POST /findings` records a person's finding on a completed work order (insert-only; a
correction is a new row that supersedes the old one). `GET /findings` lists every row, so
superseded findings stay visible. `GET /findings/trust` is the deterministic alarm-trust
count shown on Fleet and ModelOps; `GET /findings/ledger` lists the confirmed findings the
next training run will label. Nothing here reads simulator ground truth.
"""

from __future__ import annotations

from fastapi import APIRouter

from apps.api import db, findings
from apps.api.schemas import AlarmTrust, LabelLedger, RepairFinding, RepairFindingRequest

router = APIRouter(tags=["findings"])


@router.get("/findings", response_model=list[RepairFinding])
def list_findings(
    asset_id: str | None = None, contract_id: str | None = None
) -> list[RepairFinding]:
    """Every finding row, oldest first, superseded rows included (optionally filtered)."""
    return db.list_repair_findings(
        asset_id=asset_id, contract_id=contract_id, engine=db.get_engine()
    )


@router.post("/findings", response_model=RepairFinding, status_code=201)
def post_finding(req: RepairFindingRequest) -> RepairFinding:
    """Record what the technician found on a completed work order (insert-only).

    404 when the contract is not an approved work order; 409 when the work order is not
    completed yet or `supersedes` does not name the current finding.
    """
    return findings.record_finding(req, engine=db.get_engine())


@router.get("/findings/trust", response_model=AlarmTrust)
def get_alarm_trust() -> AlarmTrust:
    """Alarms acted on / confirmed / no fault found / awaiting finding (counts, sandbox excluded)."""
    return findings.alarm_trust(engine=db.get_engine())


@router.get("/findings/ledger", response_model=LabelLedger)
def get_label_ledger() -> LabelLedger:
    """Confirmed findings queued for the next `mode=train` run, with the labelling rule."""
    return findings.label_ledger(engine=db.get_engine())
