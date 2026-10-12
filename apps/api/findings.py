"""Repair findings: "What did the technician find?" (Phase A.3, CLAUDE.md sections 7 and 9).

Thor used to stop at the repair: the work order completed and Thor never learned whether its
alarm was right. A finding closes that loop. It is entered by a person on a COMPLETED work
order, stored insert-only (`db.repair_findings`; a correction is a new row that supersedes the
old one, both stay visible) and consumed three ways, all deterministic Python:

* **alarm trust** -- counts over approved maintenance contracts and their findings (Fleet,
  ModelOps);
* **Bolt's memory** -- `plant_precedent()` matches findings to a symptom question by failure
  mode and asset family (no embedding), for `search_field_history`'s `this_plant` list;
* **training labels** -- `field_failure_events()` turns every effective "Confirmed: bearing
  damage" into a `FieldFailureEvent` for the next `mode=train` run (see
  `agents/ml_architect/labels.py`).

Nothing here reads the simulator's `health` or `failure_within_h`. Sandbox-tagged findings are
excluded everywhere.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pandas as pd
from fastapi import HTTPException
from sqlalchemy.engine import Engine

from apps.api import db
from apps.api.schemas import (
    FINDING_LABEL,
    AlarmTrust,
    DecisionContract,
    FieldFailureEvent,
    LabelLedger,
    LabelLedgerEntry,
    RepairFinding,
    RepairFindingRequest,
    WorkOrder,
)

log = logging.getLogger("thor.findings")

#: Below this many acted-on alarms the UI shows counts only, never a percentage.
MIN_FOR_RATE = 5

LABEL_RULE = (
    "A confirmed bearing-damage finding labels the motor's rows from its previous completed "
    "repair (or the start of its history) up to the moment it went offline for this repair "
    "with the hours until that intervention; the motor joins the train group only. No fault "
    "found and Other change nothing in training."
)

#: Words a symptom question shares with a bearing-damage finding (the plant's one failure
#: mode so far). Deterministic keyword matching, not an embedding.
_BEARING_MODE_WORDS = frozenset(
    {
        "bearing",
        "bearings",
        "vibration",
        "vibrating",
        "vibrate",
        "kurtosis",
        "crest",
        "noise",
        "noisy",
        "hot",
        "heat",
        "temperature",
        "overheating",
        "damage",
        "damaged",
        "wear",
        "worn",
        "spall",
        "spalling",
        "rumble",
        "grinding",
        "drive",
        "end",
    }
)
_GENERIC_WORDS = frozenset(
    {"seen", "before", "similar", "happened", "precedent", "history", "past", "again"}
)


def _wo_by_contract(engine: Engine | None) -> dict[str, WorkOrder]:
    return {wo.contract_id: wo for wo in db.list_work_orders(engine=engine)}


def record_finding(req: RepairFindingRequest, engine: Engine | None = None) -> RepairFinding:
    """Insert a finding for a completed work order (the only way a finding is created).

    Inputs: the request (contract, kind, note, the person's name, optional `supersedes`,
    sandbox flag). Rules: 404 when the contract is not an approved work order; 409 when the
    work order is not completed yet (the plant has not reported back) or when `supersedes`
    names a finding that is not the one currently standing. When a finding already exists and
    `supersedes` is omitted, the new row supersedes the current one. `plant_ts` is the work
    order's completion time in plant time. Output: the inserted row.
    """
    engine = engine or db.get_engine()
    wo = _wo_by_contract(engine).get(req.contract_id)
    if wo is None:
        raise HTTPException(
            status_code=404, detail=f"no approved work order for contract {req.contract_id}"
        )
    if wo.status != "completed" or wo.completed_ts is None:
        raise HTTPException(
            status_code=409,
            detail=f"work order {req.contract_id} is {wo.status}; a finding needs a completed repair",
        )
    current = db.effective_findings(engine=engine, include_sandbox=True).get(req.contract_id)
    supersedes = req.supersedes
    if current is not None:
        if supersedes is None:
            supersedes = current.finding_id
        elif supersedes != current.finding_id:
            raise HTTPException(
                status_code=409,
                detail=f"finding {supersedes} is not the current finding for {req.contract_id} ({current.finding_id})",
            )
    elif supersedes is not None:
        raise HTTPException(status_code=409, detail=f"no finding to supersede on {req.contract_id}")
    row = RepairFinding(
        finding_id=f"rf_{uuid4().hex[:12]}",
        contract_id=req.contract_id,
        asset_id=wo.asset_id,
        finding=req.finding,
        note=req.note.strip(),
        recorded_by=req.recorded_by,
        recorded_at=datetime.now(UTC),
        plant_ts=db._as_utc(wo.completed_ts),
        sandbox=bool(req.sandbox),
        supersedes=supersedes,
    )
    db.insert_repair_finding(row, engine=engine)
    log.info(
        "finding %s on %s: %s by %s", row.finding_id, row.asset_id, row.finding, row.recorded_by
    )
    return row


def alarm_trust(engine: Engine | None = None) -> AlarmTrust:
    """Deterministic alarm-trust counts (never computed by the LLM).

    Output: `AlarmTrust` with acted_on = approved maintain_now / maintain_later contracts,
    confirmed / no_fault_found / other = their effective (non-sandbox) findings,
    awaiting_finding = acted on without one, completed_without_finding = those whose work
    order is already completed.
    """
    engine = engine or db.get_engine()
    orders = db.list_work_orders(engine=engine)
    counts = {"confirmed_bearing_damage": 0, "no_fault_found": 0, "other": 0}
    awaiting = completed_awaiting = 0
    for wo in orders:
        f = wo.finding
        if f is None or f.sandbox:
            awaiting += 1
            if wo.status == "completed":
                completed_awaiting += 1
            continue
        counts[f.finding] = counts.get(f.finding, 0) + 1
    return AlarmTrust(
        acted_on=len(orders),
        confirmed=counts["confirmed_bearing_damage"],
        no_fault_found=counts["no_fault_found"],
        other=counts["other"],
        awaiting_finding=awaiting,
        completed_without_finding=completed_awaiting,
        min_for_rate=MIN_FOR_RATE,
    )


def field_failure_events(engine: Engine | None = None) -> list[FieldFailureEvent]:
    """Every effective, non-sandbox "Confirmed: bearing damage" finding as a training event.

    Output: events oldest first. `event_ts` = the work order's `started_ts` (the plant's
    `in_progress` report), else its `completed_ts`; `labelled_from` = the motor's previous
    completed work order (its completion time) or None. Findings on work orders that no longer
    resolve are skipped. NFF and Other never produce an event.
    """
    engine = engine or db.get_engine()
    orders = db.list_work_orders(engine=engine)
    by_contract = {wo.contract_id: wo for wo in orders}
    completed_by_asset: dict[str, list[datetime]] = {}
    for wo in orders:
        if wo.completed_ts is not None:
            completed_by_asset.setdefault(wo.asset_id, []).append(db._as_utc(wo.completed_ts))
    events: list[FieldFailureEvent] = []
    for contract_id, f in db.effective_findings(engine=engine).items():
        if f.finding != "confirmed_bearing_damage":
            continue
        wo = by_contract.get(contract_id)
        if wo is None:
            continue
        event_ts = wo.started_ts or wo.completed_ts
        if event_ts is None:
            continue
        event_ts = db._as_utc(event_ts)
        earlier = [t for t in completed_by_asset.get(f.asset_id, []) if t < event_ts]
        events.append(
            FieldFailureEvent(
                asset_id=f.asset_id,
                contract_id=contract_id,
                finding_id=f.finding_id,
                recorded_by=f.recorded_by,
                event_ts=event_ts,
                labelled_from=max(earlier) if earlier else None,
                note=f.note,
            )
        )
    events.sort(key=lambda e: (e.event_ts, e.asset_id))
    return events


def training_rows(event: FieldFailureEvent, engine: Engine | None = None) -> pd.DataFrame:
    """The telemetry a finding may label: the motor's stored rows (seeded history AND live
    replayed rows, which carry no ground truth) up to the intervention, after its previous
    repair. Output: a telemetry frame sorted by `ts`; the label itself is applied by
    `agents.ml_architect.labels.label_rows_for_event`."""
    engine = engine or db.get_engine()
    return db.read_telemetry(
        asset_id=event.asset_id,
        start=event.labelled_from,
        end=event.event_ts,
        engine=engine,
    )


def label_ledger(engine: Engine | None = None) -> LabelLedger:
    """GET /findings/ledger: the confirmed findings the next `mode=train` run will label."""
    return LabelLedger(
        queued=[
            LabelLedgerEntry(
                finding_id=ev.finding_id,
                asset_id=ev.asset_id,
                contract_id=ev.contract_id,
                recorded_by=ev.recorded_by,
                recorded_at=next(
                    (
                        f.recorded_at
                        for f in db.list_repair_findings(engine=engine)
                        if f.finding_id == ev.finding_id
                    ),
                    ev.event_ts,
                ),
                event_ts=ev.event_ts,
                note=ev.note,
            )
            for ev in field_failure_events(engine)
        ],
        rule=LABEL_RULE,
    )


# --------------------------------------------------------------------------------------
# Bolt: "Have we seen this before?" -- this plant's findings, matched deterministically
# --------------------------------------------------------------------------------------


def _mode_words(f: RepairFinding, dc: DecisionContract | None) -> set[str]:
    """Words that describe a finding's failure mode: the bearing vocabulary for a confirmed
    bearing damage, the note's words for Other, and the contract's risk-raising driver
    phrases for every finding."""
    from agents.reliability import rag

    words: set[str] = set()
    if f.finding == "confirmed_bearing_damage":
        words |= _BEARING_MODE_WORDS
    if f.note:
        words |= rag._content_tokens(rag.normalize_symptom_text(f.note) or f.note.lower())
    if dc is not None:
        feats = [x.feature for x in dc.top_features if x.direction == "raises_risk"] or [
            x.feature for x in dc.top_features
        ]
        words |= rag._content_tokens(rag.history_query_from_features(feats))
    return words


def plant_precedent(
    query: str, engine: Engine | None = None, asset_id: str | None = None, k: int = 5
) -> list[dict[str, Any]]:
    """This plant's effective findings that match a symptom question.

    Matching is deterministic: a finding matches when the question's content words overlap
    its failure-mode words (`_mode_words`), or when the question names / is asked in the
    context of the same asset. With only generic words ("have we seen this before") and an
    asset in context, every finding of that asset family (same `asset_type`) matches, same
    asset first. Output: up to `k` dicts, same asset first, then by overlap, newest first:
    {finding_id, asset_id, contract_id, finding, finding_label, note, recorded_by,
    recorded_at, plant_ts, event_ts, recommendation, failure_probability, top_drivers,
    source}. Sandbox and superseded rows never appear. Never raises.
    """
    from agents.reliability import rag

    engine = engine or db.get_engine()
    try:
        effective = db.effective_findings(engine=engine)
    except Exception as e:  # noqa: BLE001 - precedent enriches an answer, never blocks it
        log.warning("plant_precedent: findings unavailable: %s", e)
        return []
    if not effective:
        return []
    q_tokens = rag._content_tokens(rag.normalize_symptom_text(query) or (query or "").lower())
    specific = q_tokens - _GENERIC_WORDS
    assets = {a.asset_id: a for a in db.list_assets(engine)}
    family = assets[asset_id].asset_type if asset_id and asset_id in assets else None
    orders = _wo_by_contract(engine)
    out: list[tuple[tuple[int, int, str], dict[str, Any]]] = []
    for contract_id, f in effective.items():
        dc = db.get_decision_contract(contract_id, engine=engine)
        overlap = len(specific & _mode_words(f, dc))
        same_asset = asset_id is not None and f.asset_id == asset_id.upper()
        same_family = (
            family is not None
            and assets.get(f.asset_id) is not None
            and assets[f.asset_id].asset_type == family
        )
        if not (overlap or same_asset or (same_family and not specific)):
            continue
        wo = orders.get(contract_id)
        drivers: list[str] = []
        if dc is not None:
            from agents.reliability.explain import FEATURE_PHRASES

            drivers = [
                FEATURE_PHRASES.get(x.feature, x.feature)
                for x in dc.top_features
                if x.direction == "raises_risk"
            ][:3]
        entry = {
            "finding_id": f.finding_id,
            "asset_id": f.asset_id,
            "contract_id": contract_id,
            "finding": f.finding,
            "finding_label": FINDING_LABEL.get(f.finding, f.finding),
            "note": f.note,
            "recorded_by": f.recorded_by,
            "recorded_at": f.recorded_at.isoformat(),
            "plant_ts": f.plant_ts.isoformat(),
            "event_ts": (db._as_utc(wo.started_ts).isoformat() if wo and wo.started_ts else None),
            "recommendation": dc.recommendation if dc else None,
            "failure_probability": dc.failure_probability if dc else None,
            "top_drivers": drivers,
            "source": f"this plant: work order {contract_id}, recorded by {f.recorded_by}",
        }
        out.append(((0 if same_asset else 1, -overlap, f.recorded_at.isoformat()), entry))
    # newest first within ties (stable sorts: recency first, then the ranking keys)
    out.sort(key=lambda t: t[0][2], reverse=True)
    out.sort(key=lambda t: (t[0][0], t[0][1]))
    return [e for _, e in out][: max(1, int(k))]
